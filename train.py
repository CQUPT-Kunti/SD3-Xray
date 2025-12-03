import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from diffusers import StableDiffusion3Pipeline, FlowMatchEulerDiscreteScheduler  # ✅ 修改1
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
    X-ray dataset:
    - data/train/X-ray
    - data/train/X-ray/X-ray_transform_padding512512  # ✅ 修改了路径
    """
    def __init__(self, instance_data_root, tokenizer1, tokenizer2, resolution=1024):
        self.resolution = resolution
        self.tokenizer1 = tokenizer1
        self.tokenizer2 = tokenizer2
        
        # Original images
        self.original_dir = os.path.join(instance_data_root, "train", "X-ray")
        self.original_files = sorted([f for f in os.listdir(self.original_dir) 
                                     if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        self.num_original = len(self.original_files)
        
        # ✅ Augmented images - 修改路径（去掉transform）
        self.augmented_dir = os.path.join(instance_data_root, "train", "X-ray", "X-ray_transform_padding512512")
        
        # 如果X-ray_transform_padding512512文件夹不存在，就只用原始图像
        if os.path.exists(self.augmented_dir):
            self.augmented_files = sorted([f for f in os.listdir(self.augmented_dir) 
                                          if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
            self.num_augmented = len(self.augmented_files)
        else:
            logger.warning(f"Augmented directory not found: {self.augmented_dir}")
            logger.warning("Using only original images")
            self.augmented_files = []
            self.num_augmented = 0
        
        self.length = self.num_original + self.num_augmented
        
        logger.info("=" * 60)
        logger.info("Dataset Configuration")
        logger.info(f"Original images: {self.num_original}")
        logger.info(f"Augmented images: {self.num_augmented}")
        logger.info(f"Total samples: {self.length}")
        if self.num_original > 0:
            logger.info(f"Data expansion: {(self.length / self.num_original):.1f}x")
        logger.info("=" * 60)
        
        print("=" * 60)
        print("Dataset Configuration")
        print(f"Original images (X-ray): {self.num_original}")
        print(f"Augmented images (X-ray/X-ray_transform_padding512512): {self.num_augmented}")
        print(f"Total samples: {self.length}")
        if self.num_original > 0:
            print(f"Data expansion: {(self.length / self.num_original):.1f}x")
        print("=" * 60)
        
        if self.length == 0:
            raise ValueError(f"No images found in {instance_data_root}")
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # First half: original images
        if idx < self.num_original:
            img_path = os.path.join(self.original_dir, self.original_files[idx])
            image = Image.open(img_path).convert("RGB")
            prompt = "a high quality X-ray image of scoliosis with no padding"
        else:
            # Second half: augmented images
            aug_idx = idx - self.num_original
            img_path = os.path.join(self.augmented_dir, self.augmented_files[aug_idx])
            image = Image.open(img_path).convert("RGB")
            prompt = "a high quality X-ray image of scoliosis with padding"
        
        # Resize
        image = image.resize((self.resolution, self.resolution), Image.LANCZOS)
        
        # ✅ 正确的归一化: [0,255] -> [0,1] -> [-1,1]
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        image = image * 2.0 - 1.0  # [-1,1]
        
        # Tokenize prompt
        tokens1 = self.tokenizer1(prompt, padding="max_length", max_length=77, 
                                  truncation=True, return_tensors="pt")
        tokens2 = self.tokenizer2(prompt, padding="max_length", max_length=77, 
                                  truncation=True, return_tensors="pt")
        
        return {
            "pixel_values": image,
            "input_ids1": tokens1.input_ids[0],
            "attention_mask1": tokens1.attention_mask[0],
            "input_ids2": tokens2.input_ids[0],
            "attention_mask2": tokens2.attention_mask[0],
        }


def encode_prompt(text_encoder1, text_encoder2, input_ids1, input_ids2, 
                  attention_mask1=None, attention_mask2=None):
    """Tokenize prompt"""
    out1 = text_encoder1(input_ids1, attention_mask=attention_mask1, 
                        output_hidden_states=False, return_dict=True)
    out2 = text_encoder2(input_ids2, attention_mask=attention_mask2, 
                        output_hidden_states=False, return_dict=True)
    
    prompt_embeds1 = out1.last_hidden_state
    prompt_embeds2 = out2.last_hidden_state
    
    if hasattr(out1, 'text_embeds') and out1.text_embeds is not None:
        pooled1 = out1.text_embeds
    elif hasattr(out1, 'pooler_output') and out1.pooler_output is not None:
        pooled1 = out1.pooler_output
    else:
        pooled1 = prompt_embeds1[:, 0, :]
    
    if hasattr(out2, 'text_embeds') and out2.text_embeds is not None:
        pooled2 = out2.text_embeds
    elif hasattr(out2, 'pooler_output') and out2.pooler_output is not None:
        pooled2 = out2.pooler_output
    else:
        pooled2 = prompt_embeds2[:, 0, :]
    
    prompt_embeds = torch.cat([prompt_embeds1, prompt_embeds2], dim=-1)
    pooled_prompt_embeds = torch.cat([pooled1, pooled2], dim=-1)
    
    return prompt_embeds, pooled_prompt_embeds


def adapt_transformer_for_two_encoders(transformer, new_in_features=2048):
    """Adapt transformer context_embedder"""
    old_embedder = transformer.context_embedder
    old_out_features = old_embedder.out_features
    old_in_features = old_embedder.in_features
    
    if old_in_features == new_in_features:
        logger.info(f"context_embedder already adapted to {new_in_features}")
        return transformer
    
    logger.info(f"Adapting context_embedder: {old_in_features} -> {new_in_features}")
    
    new_embedder = torch.nn.Linear(
        new_in_features, old_out_features,
        bias=(old_embedder.bias is not None),
        dtype=old_embedder.weight.dtype,
        device=old_embedder.weight.device
    )
    
    with torch.no_grad():
        if old_in_features < new_in_features:
            new_embedder.weight.data[:, :old_in_features] = old_embedder.weight.data.clone()
        else:
            new_embedder.weight.data = old_embedder.weight.data[:, :new_in_features].clone()
        
        torch.nn.init.xavier_uniform_(new_embedder.weight.data[:, old_in_features:])
        
        if new_embedder.bias is not None and old_embedder.bias is not None:
            new_embedder.bias.data = old_embedder.bias.data.clone()
    
    transformer.context_embedder = new_embedder
    logger.info(f"Context embedder adapted successfully")
    return transformer


def apply_lora_to_transformer(transformer, lora_rank=16, lora_alpha=16):
    """Apply LoRA to transformer"""
    logger.info(f"Applying LoRA (rank={lora_rank}, alpha={lora_alpha})")
    
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
    logger.info(f"Trainable: {trainable} ({100*trainable/total:.2f}%)")
    logger.info(f"Total: {total}")
    
    return transformer


def generate_validation_image(vae, text_encoder1, text_encoder2, transformer, 
                             tokenizer1, tokenizer2, scheduler_config, 
                             validation_prompt, output_path, step, epoch, 
                             device, weight_dtype, resolution=1024, current_loss=None):
    """Generate validation image"""
    logger.info(f"Generating validation image at step {step}...")
    transformer.eval()
    
    # ✅ 修改2: 使用FlowMatchEulerDiscreteScheduler
    inference_scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
    
    try:
        with torch.no_grad():
            # Tokenize prompt
            tokens1 = tokenizer1(validation_prompt, padding="max_length", max_length=77, 
                               truncation=True, return_tensors="pt")
            tokens2 = tokenizer2(validation_prompt, padding="max_length", max_length=77, 
                               truncation=True, return_tensors="pt")
            
            prompt_embeds, pooled_prompt_embeds = encode_prompt(
                text_encoder1, text_encoder2,
                tokens1.input_ids.to(device),
                tokens2.input_ids.to(device),
                tokens1.attention_mask.to(device),
                tokens2.attention_mask.to(device)
            )
            
            # Prepare latent shape
            latent_channels = getattr(vae.config, "latent_channels", 
                                     getattr(vae.config, "out_channels", 16))
            vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
            latent_size = resolution // vae_scale_factor
            logger.info(f"Latent shape: {latent_channels}, {latent_size}, {latent_size}")
            
            # Initialize latents
            generator = torch.Generator(device=device)
            latents = torch.randn(
                1, latent_channels, latent_size, latent_size,
                generator=generator, device=device, dtype=weight_dtype
            )
            
            # ✅ Flow Matching采样
            inference_scheduler.set_timesteps(28)
            
            for i, t in enumerate(inference_scheduler.timesteps):
                # 将timestep归一化到[0,1]
                timestep_tensor = t.unsqueeze(0).to(device).float() / 1000.0
                
                noise_pred = transformer(
                    hidden_states=latents,
                    timestep=timestep_tensor,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    return_dict=False
                )[0]
                
                latents = inference_scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                
                if i == 0 or i == len(inference_scheduler.timesteps) - 1:
                    logger.debug(f"Denoising step {i}/{len(inference_scheduler.timesteps)}, "
                               f"latents range [{latents.min():.3f}, {latents.max():.3f}]")
            
            # Decode
            latents = latents / vae.config.scaling_factor
            latents = latents.to(dtype=weight_dtype)
            image = vae.decode(latents).sample
            
            # Post-process
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()
            image = (image[0] * 255).round().astype("uint8")
            image = Image.fromarray(image)
            
            # Save
            timestamp = datetime.datetime.now().strftime("%H%M%S")
            if current_loss is not None:
                filename = f"step{step:06d}_epoch{epoch:03d}_loss{current_loss:.4f}_{timestamp}.png"
            else:
                filename = f"step{step:06d}_epoch{epoch:03d}_{timestamp}.png"
            
            save_path = os.path.join(output_path, filename)
            image.save(save_path)
            logger.info(f"Saved validation image: {save_path}")
            
    except Exception as e:
        logger.error(f"Failed to generate validation image: {e}")
        import traceback
        traceback.print_exc()
    finally:
        transformer.train()


def train(args):
    # Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_dir=os.path.join(args.output_dir, "logs"),
    )
    
    if args.seed is not None:
        set_seed(args.seed)
        logger.info(f"Using fixed seed: {args.seed}")
    else:
        logger.info(f"Using random seed")
    
    # Load model
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
    
    vae = pipe.vae
    text_encoder1 = pipe.text_encoder
    text_encoder2 = pipe.text_encoder_2
    transformer = pipe.transformer
    
    # ✅ 修改3: 使用FlowMatchEulerDiscreteScheduler
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
    logger.info(f"✅ Using FlowMatchEulerDiscreteScheduler for SD3")
    logger.info(f"VAE scaling_factor: {vae.config.scaling_factor}")
    
    # Adapt transformer
    transformer = adapt_transformer_for_two_encoders(transformer, new_in_features=2048)
    
    # LoRA
    if args.use_lora:
        transformer = apply_lora_to_transformer(
            transformer, 
            lora_rank=args.lora_rank, 
            lora_alpha=args.lora_alpha
        )
    else:
        transformer.requires_grad_(True)
        logger.info(f"Training full transformer")
    
    vae.requires_grad_(False)
    text_encoder1.requires_grad_(False)
    text_encoder2.requires_grad_(False)
    
    # Gradient checkpointing
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
        logger.info("Gradient checkpointing enabled")
    
    # Dataset
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
    
    # Optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_class = bnb.optim.AdamW8bit
            logger.info("Using 8-bit Adam optimizer")
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
    
    # LR scheduler
    num_update_steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.epochs * num_update_steps_per_epoch
    else:
        args.epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )
    
    # Prepare with accelerator
    transformer, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, dataloader, lr_scheduler
    )
    
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder1.to(accelerator.device, dtype=weight_dtype)
    text_encoder2.to(accelerator.device, dtype=weight_dtype)
    
    # Training info
    total_batch_size = args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    
    logger.info("=" * 60)
    logger.info("Training Configuration")
    logger.info(f"Num samples: {len(dataset)}")
    logger.info(f"Num epochs: {args.epochs}")
    logger.info(f"Instantaneous batch size: {args.batch_size}")
    logger.info(f"Total train batch size: {total_batch_size}")
    logger.info(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    logger.info(f"Total optimization steps: {args.max_train_steps}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"LR scheduler: {args.lr_scheduler}")
    logger.info(f"Use LoRA: {args.use_lora}")
    if args.use_lora:
        logger.info(f"LoRA rank: {args.lora_rank}")
    logger.info("=" * 60)
    
    # Output directories
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        validation_dir = os.path.join(args.output_dir, "validation_images")
        os.makedirs(validation_dir, exist_ok=True)
    
    # Tracker
    if accelerator.is_main_process:
        accelerator.init_trackers("sd3_xray_training")
    
    # Generate initial validation image
    if accelerator.is_main_process:
        generate_validation_image(
            vae, text_encoder1, text_encoder2,
            accelerator.unwrap_model(transformer),
            pipe.tokenizer, pipe.tokenizer_2,
            noise_scheduler.config,
            args.validation_prompt,
            validation_dir, 0, 0,
            accelerator.device, weight_dtype,
            resolution=args.resolution
        )
    
    # Training loop
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
                
                # Encode images to latents
                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                    
                    prompt_embeds, pooled_prompt_embeds = encode_prompt(
                        text_encoder1, text_encoder2,
                        batch['input_ids1'], batch['input_ids2'],
                        batch['attention_mask1'], batch['attention_mask2']
                    )
                
                # ✅ 修改4: SD3 Flow Matching训练
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                
                # 采样连续时间步 u ~ Uniform(0, 1)
                u = torch.rand(bsz, device=latents.device, dtype=latents.dtype)
                
                # Flow Matching: 线性插值 x_t = (1-t)*x_0 + t*x_1
                sigmas = u.view(-1, 1, 1, 1)
                noisy_latents = (1 - sigmas) * latents + sigmas * noise
                
                # 目标是velocity field: v = x_1 - x_0
                target = noise - latents
                
                # ✅ 修改5: Transformer预测（timestep用连续的u）
                model_pred = transformer(
                    hidden_states=noisy_latents,
                    timestep=u,  # ⚠️ 这里是连续时间步，不是离散的
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    return_dict=False
                )[0]
                
                # 计算loss
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Update progress
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                epoch_loss += loss.detach().item()
            
            # Logging
            if global_step % 100 == 0:
                current_lr = lr_scheduler.get_last_lr()[0]
                logger.info(f"Step {global_step} | LR={current_lr:.2e}, Loss={loss.item():.6f}")
            
            # Log to tensorboard
            current_lr = lr_scheduler.get_last_lr()[0]
            logs = {"loss": loss.detach().item(), "lr": current_lr, "epoch": epoch}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            
            # Validation
            if global_step % args.validation_steps == 0 and accelerator.is_main_process:
                generate_validation_image(
                    vae, text_encoder1, text_encoder2,
                    accelerator.unwrap_model(transformer),
                    pipe.tokenizer, pipe.tokenizer_2,
                    noise_scheduler.config,
                    args.validation_prompt,
                    validation_dir, global_step, epoch + 1,
                    accelerator.device, weight_dtype,
                    resolution=args.resolution,
                    current_loss=epoch_loss / (step + 1)
                )
            
            # Save checkpoint
            if global_step % args.save_steps == 0 and accelerator.is_main_process:
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                os.makedirs(save_path, exist_ok=True)
                unwrapped_transformer = accelerator.unwrap_model(transformer)
                unwrapped_transformer.save_pretrained(save_path)
                logger.info(f"Saved checkpoint to {save_path}")
            
            if global_step >= args.max_train_steps:
                break
        
        # Epoch summary
        avg_epoch_loss = epoch_loss / len(dataloader)
        logger.info(f"epoch {epoch + 1}/{args.epochs} - Avg Loss: {avg_epoch_loss:.6f}")
    
    # Save final model
    if accelerator.is_main_process:
        final_path = os.path.join(args.output_dir, "final_lora" if args.use_lora else "final_transformer")
        os.makedirs(final_path, exist_ok=True)
        unwrapped_transformer = accelerator.unwrap_model(transformer)
        unwrapped_transformer.save_pretrained(final_path)
        
        logger.info("=" * 60)
        logger.info("Training complete!")
        logger.info(f"Final model saved to {final_path}")
        logger.info("=" * 60)
    
    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SD3 Fine-tuning with pre-augmented data")
    
    # Model paths
    parser.add_argument("--safetensors_path", type=str, required=True, 
                       help="Path to SD3 safetensors checkpoint")
    parser.add_argument("--sd3_config_dir", type=str, required=True, 
                       help="Path to SD3 config directory")
    parser.add_argument("--instance_data_dir", type=str, required=True, 
                       help="Root dir containing train/X-ray and train/X-ray_transform_padding512512")
    parser.add_argument("--output_dir", type=str, default="./output", 
                       help="Output directory for checkpoints")
    parser.add_argument("--validation_prompt", type=str, default="a high quality X-ray image of scoliosis", 
                       help="Prompt for validation image generation")
    
    # Training hyperparameters
    parser.add_argument("--resolution", type=int, default=1024, help="Image resolution")
    parser.add_argument("--batch_size", type=int, default=1, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs")
    parser.add_argument("--max_train_steps", type=int, default=None, 
                       help="Total number of training steps (overrides epochs)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, 
                       help="Number of gradient accumulation steps")
    parser.add_argument("--gradient_checkpointing", action="store_true", 
                       help="Enable gradient checkpointing to save memory")
    parser.add_argument("--mixed_precision", type=str, default="bf16", 
                       choices=["no", "fp16", "bf16"], help="Mixed precision training")
    
    # Optimizer
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lr_scheduler", type=str, default="cosine", 
                       choices=["linear", "cosine", "cosine_with_restarts", "constant", "constant_with_warmup"],
                       help="Learning rate scheduler type")
    parser.add_argument("--lr_warmup_steps", type=int, default=500, 
                       help="Number of warmup steps for learning rate scheduler")
    parser.add_argument("--use_8bit_adam", action="store_true", 
                       help="Use 8-bit Adam optimizer to save memory")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="Adam beta2")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Adam weight decay")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="Adam epsilon")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, 
                       help="Max gradient norm for clipping")
    
    # LoRA
    parser.add_argument("--use_lora", action="store_true", 
                       help="Use LoRA instead of full fine-tuning")
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    
    # Misc
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
