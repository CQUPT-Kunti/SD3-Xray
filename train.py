import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from diffusers import StableDiffusion3Pipeline, DDPMScheduler
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from PIL import Image
import os
from pathlib import Path
from tqdm import tqdm
import argparse
import numpy as np
import datetime
import math

logger = get_logger(__name__)


class XrayDatasetDual(Dataset):
    """
    双文件夹X光片数据集 - 直接使用预处理好的数据
    - data/train/X-ray: 原图
    - data/train/X-ray_transform_padding512512: 增强图片
    """
    def __init__(self, instance_data_root, tokenizer_1, tokenizer_2, resolution=1024):
        self.resolution = resolution
        self.tokenizer_1 = tokenizer_1
        self.tokenizer_2 = tokenizer_2
        
        # 原图文件夹
        self.original_dir = os.path.join(instance_data_root, 'train', 'X-ray')
        self.original_files = sorted([f for f in os.listdir(self.original_dir) 
                                      if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        self.num_original = len(self.original_files)
        
        # 增强图片文件夹
        self.augmented_dir = os.path.join(instance_data_root, 'train', 'X-ray_transform_padding512512')
        self.augmented_files = sorted([f for f in os.listdir(self.augmented_dir)
                                       if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        self.num_augmented = len(self.augmented_files)
        
        # 总长度 = 原图 + 增强图
        self._length = self.num_original + self.num_augmented
        
        logger.info(f"\n{'='*60}")
        logger.info(f"📁 Dataset Configuration:")
        logger.info(f"   Original images: {self.num_original}")
        logger.info(f"   Augmented images: {self.num_augmented}")
        logger.info(f"   Total samples: {self._length}")
        logger.info(f"   Data expansion: {self._length / self.num_original:.1f}x")
        logger.info(f"{'='*60}\n")
        
        if self._length == 0:
            raise ValueError(f"No images found in {instance_data_root}")
    
    def __len__(self):
        return self._length
    
    def __getitem__(self, idx):
        # 前半部分是原图，后半部分是增强图
        if idx < self.num_original:
            # 加载原图
            img_path = os.path.join(self.original_dir, self.original_files[idx])
            image = Image.open(img_path).convert('RGB')
            prompt = "a high quality X-ray image of scoliosis with nopadding"
        else:
            # 加载增强图
            aug_idx = idx - self.num_original
            img_path = os.path.join(self.augmented_dir, self.augmented_files[aug_idx])
            image = Image.open(img_path).convert('RGB')
            prompt = "a high quality X-ray image of scoliosis with padding"
        
        # Resize 到目标分辨率
        image = image.resize((self.resolution, self.resolution), Image.LANCZOS)
        
        # 转换为tensor，归一化到 [-1, 1]
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 127.5 - 1.0
        
        # Tokenize prompt
        tokens_1 = self.tokenizer_1(
            prompt, padding="max_length", max_length=77,
            truncation=True, return_tensors="pt"
        )
        tokens_2 = self.tokenizer_2(
            prompt, padding="max_length", max_length=77,
            truncation=True, return_tensors="pt"
        )
        
        return {
            "pixel_values": image,
            "input_ids_1": tokens_1.input_ids[0],
            "attention_mask_1": tokens_1.attention_mask[0],
            "input_ids_2": tokens_2.input_ids[0],
            "attention_mask_2": tokens_2.attention_mask[0],
        }


def encode_prompt(text_encoder_1, text_encoder_2,
                  input_ids_1, input_ids_2,
                  attention_mask_1=None, attention_mask_2=None):
    """编码文本 prompt"""
    out1 = text_encoder_1(input_ids_1, attention_mask=attention_mask_1, 
                          output_hidden_states=False, return_dict=True)
    out2 = text_encoder_2(input_ids_2, attention_mask=attention_mask_2,
                          output_hidden_states=False, return_dict=True)

    prompt_embeds_1 = out1.last_hidden_state
    prompt_embeds_2 = out2.last_hidden_state

    if hasattr(out1, "text_embeds") and out1.text_embeds is not None:
        pooled_1 = out1.text_embeds
    elif hasattr(out1, "pooler_output") and out1.pooler_output is not None:
        pooled_1 = out1.pooler_output
    else:
        pooled_1 = prompt_embeds_1[:, 0, :]

    if hasattr(out2, "text_embeds") and out2.text_embeds is not None:
        pooled_2 = out2.text_embeds
    elif hasattr(out2, "pooler_output") and out2.pooler_output is not None:
        pooled_2 = out2.pooler_output
    else:
        pooled_2 = prompt_embeds_2[:, 0, :]

    prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)
    pooled_prompt_embeds = torch.cat([pooled_1, pooled_2], dim=-1)

    return prompt_embeds, pooled_prompt_embeds


def adapt_transformer_for_two_encoders(transformer, new_in_features=2048):
    """适配 transformer 的 context_embedder"""
    old_embedder = transformer.context_embedder
    old_out_features = old_embedder.out_features
    old_in_features = old_embedder.in_features
    
    if old_in_features == new_in_features:
        logger.info(f"⚠️  context_embedder already adapted to {new_in_features}")
        return transformer
    
    logger.info(f"🔧 Adapting context_embedder: {old_in_features} → {new_in_features}")
    
    new_embedder = torch.nn.Linear(
        new_in_features, old_out_features,
        bias=old_embedder.bias is not None,
        dtype=old_embedder.weight.dtype,
        device=old_embedder.weight.device
    )
    
    with torch.no_grad():
        if old_in_features > new_in_features:
            new_embedder.weight.data = old_embedder.weight.data[:, :new_in_features].clone()
        else:
            new_embedder.weight.data[:, :old_in_features] = old_embedder.weight.data.clone()
            torch.nn.init.xavier_uniform_(new_embedder.weight.data[:, old_in_features:])
        
        if new_embedder.bias is not None and old_embedder.bias is not None:
            new_embedder.bias.data = old_embedder.bias.data.clone()
    
    transformer.context_embedder = new_embedder
    logger.info(f"✅ Context embedder adapted successfully")
    return transformer


def apply_lora_to_transformer(transformer, lora_rank=16, lora_alpha=16):
    """应用 LoRA 到 transformer"""
    logger.info(f"🎯 Applying LoRA (rank={lora_rank}, alpha={lora_alpha})")
    
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.1,
    )
    
    transformer = get_peft_model(transformer, lora_config)
    
    trainable = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    total = sum(p.numel() for p in transformer.parameters())
    
    logger.info(f"   Trainable: {trainable:,} ({100*trainable/total:.2f}%)")
    logger.info(f"   Total: {total:,}")
    
    return transformer


def generate_validation_image(vae, text_encoder_1, text_encoder_2, transformer, 
                             tokenizer_1, tokenizer_2, scheduler_config,
                             validation_prompt, output_path, step, epoch, 
                             device, weight_dtype, current_loss=None):
    """生成验证图片"""
    logger.info(f"📸 Generating validation image at step {step}...")
    
    transformer.eval()
    inference_scheduler = DDPMScheduler.from_config(scheduler_config)
    
    try:
        with torch.no_grad():
            tokens_1 = tokenizer_1(validation_prompt, padding="max_length", 
                                  max_length=77, truncation=True, return_tensors="pt")
            tokens_2 = tokenizer_2(validation_prompt, padding="max_length",
                                  max_length=77, truncation=True, return_tensors="pt")
            
            prompt_embeds, pooled_prompt_embeds = encode_prompt(
                text_encoder_1, text_encoder_2,
                tokens_1.input_ids.to(device), tokens_2.input_ids.to(device),
                tokens_1.attention_mask.to(device), tokens_2.attention_mask.to(device)
            )
            
            # 每次使用随机种子生成不同的验证图片
            generator = torch.Generator(device=device)
            latents = torch.randn((1, 16, 128, 128), generator=generator,
                                 device=device, dtype=weight_dtype)
            
            inference_scheduler.set_timesteps(28)
            latents = latents * inference_scheduler.init_noise_sigma
            
            for t in inference_scheduler.timesteps:
                noise_pred = transformer(
                    hidden_states=latents, timestep=t.unsqueeze(0).to(device),
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    return_dict=False
                )[0]
                latents = inference_scheduler.step(noise_pred, t, latents).prev_sample
            
            # 🔧 修复：解码到像素空间，确保数据类型匹配
            latents = latents / vae.config.scaling_factor
            # 将 latents 转换为与 VAE 相同的数据类型
            latents = latents.to(dtype=weight_dtype)
            image = vae.decode(latents).sample
            
            # 转换到 PIL Image
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()
            image = (image[0] * 255).round().astype("uint8")
            image = Image.fromarray(image)
        
        timestamp = datetime.datetime.now().strftime("%H%M%S")
        if current_loss is not None:
            filename = f"step{step:06d}_epoch{epoch:03d}_loss{current_loss:.4f}_{timestamp}.png"
        else:
            filename = f"step{step:06d}_epoch{epoch:03d}_{timestamp}.png"
        
        save_path = os.path.join(output_path, filename)
        image.save(save_path)
        logger.info(f"✓ Saved: {save_path}")
    
    except Exception as e:
        logger.error(f"✗ Failed to generate validation image: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        transformer.train()


def train(args):
    # 🚀 使用 Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_dir=os.path.join(args.output_dir, "logs"),
    )
    
    # 设置随机种子（可选）
    if args.seed is not None:
        set_seed(args.seed)
        logger.info(f"✓ Using fixed seed: {args.seed}")
    else:
        logger.info(f"✓ Using random seed")
    
    # 加载模型
    logger.info("Loading SD3 model...")
    
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32
    
    pipe = StableDiffusion3Pipeline.from_single_file(
        args.safetensors_path,
        config=args.sd3_config_dir,
        torch_dtype=weight_dtype,
        local_files_only=True,
        text_encoder_3=None,
        tokenizer_3=None,
    )
    
    # 提取组件
    vae = pipe.vae
    text_encoder_1 = pipe.text_encoder
    text_encoder_2 = pipe.text_encoder_2
    transformer = pipe.transformer
    
    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    
    logger.info(f"✓ Using DDPMScheduler")
    logger.info(f"✓ VAE scaling_factor: {vae.config.scaling_factor}")
    
    # 适配 transformer
    transformer = adapt_transformer_for_two_encoders(transformer, new_in_features=2048)
    
    # 应用 LoRA
    if args.use_lora:
        transformer = apply_lora_to_transformer(
            transformer, 
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha
        )
    else:
        transformer.requires_grad_(True)
        logger.info(f"Training full transformer")
    
    # 冻结不训练的模型
    vae.requires_grad_(False)
    text_encoder_1.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    
    # 梯度检查点
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
        logger.info("✓ Gradient checkpointing enabled")
    
    # 创建数据集
    dataset = XrayDatasetDual(
        args.instance_data_dir,
        pipe.tokenizer,
        pipe.tokenizer_2,
        resolution=args.resolution
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # 优化器
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_class = bnb.optim.AdamW8bit
            logger.info("✓ Using 8-bit Adam optimizer")
        except ImportError:
            logger.warning("bitsandbytes not found, using standard AdamW")
            optimizer_class = torch.optim.AdamW
    else:
        optimizer_class = torch.optim.AdamW
    
    trainable_params = [p for p in transformer.parameters() if p.requires_grad]
    optimizer = optimizer_class(
        trainable_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    # 计算训练步数
    num_update_steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.epochs * num_update_steps_per_epoch
    else:
        args.epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    
    # 学习率调度器
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )
    
    # 使用 Accelerator 准备模型
    transformer, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, dataloader, lr_scheduler
    )
    
    # 移动其他模型到设备
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder_1.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2.to(accelerator.device, dtype=weight_dtype)
    
    # 打印训练配置
    total_batch_size = args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    
    logger.info("\n" + "="*60)
    logger.info("***** Training Configuration *****")
    logger.info(f"  Num samples = {len(dataset)}")
    logger.info(f"  Num epochs = {args.epochs}")
    logger.info(f"  Instantaneous batch size = {args.batch_size}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Learning rate = {args.learning_rate}")
    logger.info(f"  LR scheduler = {args.lr_scheduler}")
    logger.info(f"  Use LoRA = {args.use_lora}")
    if args.use_lora:
        logger.info(f"  LoRA rank = {args.lora_rank}")
    logger.info("="*60 + "\n")
    
    # 创建输出目录
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        validation_dir = os.path.join(args.output_dir, "validation_images")
        os.makedirs(validation_dir, exist_ok=True)
    
    # 初始化 tracker
    if accelerator.is_main_process:
        accelerator.init_trackers("sd3_xray_training")
    
    # 生成初始验证图片
    if accelerator.is_main_process:
        generate_validation_image(
            vae, text_encoder_1, text_encoder_2,
            accelerator.unwrap_model(transformer),
            pipe.tokenizer, pipe.tokenizer_2, noise_scheduler.config,
            args.validation_prompt, validation_dir, 0, 0,
            accelerator.device, weight_dtype
        )
    
    # 训练循环
    global_step = 0
    first_epoch = 0
    
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=0,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    
    for epoch in range(first_epoch, args.epochs):
        transformer.train()
        epoch_loss = 0
        
        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(transformer):
                pixel_values = batch["pixel_values"].to(dtype=weight_dtype)
                
                # 编码
                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                    
                    prompt_embeds, pooled_prompt_embeds = encode_prompt(
                        text_encoder_1, text_encoder_2,
                        batch["input_ids_1"], batch["input_ids_2"],
                        batch["attention_mask_1"], batch["attention_mask_2"]
                    )
                
                # 添加噪声
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (bsz,), device=latents.device
                ).long()
                
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # 预测噪声
                model_pred = transformer(
                    hidden_states=noisy_latents,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    return_dict=False
                )[0]
                
                # 计算损失
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                
                # 反向传播
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # 更新进度
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                # 记录日志
                epoch_loss += loss.detach().item()
                
                logs = {
                    "loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "epoch": epoch,
                }
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                
                # 生成验证图片
                if global_step % args.validation_steps == 0 and accelerator.is_main_process:
                    generate_validation_image(
                        vae, text_encoder_1, text_encoder_2,
                        accelerator.unwrap_model(transformer),
                        pipe.tokenizer, pipe.tokenizer_2, noise_scheduler.config,
                        args.validation_prompt, validation_dir, global_step, epoch + 1,
                        accelerator.device, weight_dtype,
                        current_loss=epoch_loss / (step + 1)
                    )
                
                # 保存检查点
                if global_step % args.save_steps == 0 and accelerator.is_main_process:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(save_path, exist_ok=True)
                    
                    unwrapped_transformer = accelerator.unwrap_model(transformer)
                    unwrapped_transformer.save_pretrained(save_path)
                    
                    logger.info(f"\n✓ Saved checkpoint to {save_path}")
                
                if global_step >= args.max_train_steps:
                    break
        
        # Epoch 结束
        avg_epoch_loss = epoch_loss / len(dataloader)
        logger.info(f"\nEpoch {epoch + 1}/{args.epochs} - Avg Loss: {avg_epoch_loss:.6f}")
    
    # 保存最终模型
    if accelerator.is_main_process:
        final_path = os.path.join(args.output_dir, "final_lora" if args.use_lora else "final_transformer")
        os.makedirs(final_path, exist_ok=True)
        
        unwrapped_transformer = accelerator.unwrap_model(transformer)
        unwrapped_transformer.save_pretrained(final_path)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"✓ Training complete!")
        logger.info(f"✓ Final model saved to: {final_path}")
        logger.info(f"{'='*60}")
    
    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SD3 Fine-tuning with pre-augmented data")
    
    # 模型和数据
    parser.add_argument("--safetensors_path", type=str, required=True,
                       help="Path to SD3 safetensors checkpoint")
    parser.add_argument("--sd3_config_dir", type=str, required=True,
                       help="Path to SD3 config directory")
    parser.add_argument("--instance_data_dir", type=str, required=True,
                       help="Root dir containing train/X-ray and train/X-ray_transform_padding512512")
    parser.add_argument("--output_dir", type=str, default="./output",
                       help="Output directory for checkpoints")
    parser.add_argument("--validation_prompt", type=str, 
                       default="a high quality X-ray image of scoliosis",
                       help="Prompt for validation image generation")
    
    # 训练设置
    parser.add_argument("--resolution", type=int, default=1024,
                       help="Image resolution")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Training batch size")
    parser.add_argument("--epochs", type=int, default=500,
                       help="Number of training epochs")
    parser.add_argument("--max_train_steps", type=int, default=None,
                       help="Total number of training steps (overrides epochs)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                       help="Number of gradient accumulation steps")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                       help="Enable gradient checkpointing to save memory")
    parser.add_argument("--mixed_precision", type=str, default="bf16",
                       choices=["no", "fp16", "bf16"],
                       help="Mixed precision training")
    
    # 优化器设置
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--lr_scheduler", type=str, default="cosine",
                       choices=["linear", "cosine", "cosine_with_restarts", "constant", "constant_with_warmup"],
                       help="Learning rate scheduler type")
    parser.add_argument("--lr_warmup_steps", type=int, default=500,
                       help="Number of warmup steps for learning rate scheduler")
    parser.add_argument("--use_8bit_adam", action="store_true",
                       help="Use 8-bit Adam optimizer to save memory")
    parser.add_argument("--adam_beta1", type=float, default=0.9,
                       help="Adam beta1")
    parser.add_argument("--adam_beta2", type=float, default=0.999,
                       help="Adam beta2")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2,
                       help="Adam weight decay")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8,
                       help="Adam epsilon")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                       help="Max gradient norm for clipping")
    
    # LoRA 设置
    parser.add_argument("--use_lora", action="store_true",
                       help="Use LoRA instead of full fine-tuning")
    parser.add_argument("--lora_rank", type=int, default=16,
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16,
                       help="LoRA alpha")
    
    # 其他
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for reproducibility (optional)")
    parser.add_argument("--num_workers", type=int, default=2,
                       help="Number of dataloader workers")
    parser.add_argument("--save_steps", type=int, default=4000,
                       help="Save checkpoint every X steps")
    parser.add_argument("--validation_steps", type=int, default=500,
                       help="Generate validation image every X steps")
    
    args = parser.parse_args()
    train(args)
