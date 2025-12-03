import torch
import torch.nn as nn
from diffusers import StableDiffusion3Pipeline, FlowMatchEulerDiscreteScheduler
from safetensors.torch import load_file


# ==================== 适配函数（独立版本）====================
def adapt_transformer_for_two_encoders(transformer, new_in_features=2048):
    """
    适配 transformer 的 context_embedder
    将SD3从3个文本编码器（CLIP-L + CLIP-G + T5）适配为2个（CLIP-L + CLIP-G）
    """
    old_embedder = transformer.context_embedder
    old_out_features = old_embedder.out_features
    old_in_features = old_embedder.in_features
    
    if old_in_features == new_in_features:
        print(f"⚠️  context_embedder already adapted to {new_in_features}")
        return transformer
    
    print(f"🔧 Adapting context_embedder: {old_in_features} → {new_in_features}")
    
    new_embedder = torch.nn.Linear(
        new_in_features, old_out_features,
        bias=old_embedder.bias is not None,
        dtype=old_embedder.weight.dtype,
        device=old_embedder.weight.device
    )
    
    with torch.no_grad():
        if old_in_features > new_in_features:
            # 从3编码器降到2编码器：截取前2048维
            new_embedder.weight.data = old_embedder.weight.data[:, :new_in_features].clone()
        else:
            # 从2编码器升到3编码器（不太可能）：复制现有权重+初始化新维度
            new_embedder.weight.data[:, :old_in_features] = old_embedder.weight.data.clone()
            torch.nn.init.xavier_uniform_(new_embedder.weight.data[:, old_in_features:])
        
        if new_embedder.bias is not None and old_embedder.bias is not None:
            new_embedder.bias.data = old_embedder.bias.data.clone()
    
    transformer.context_embedder = new_embedder
    print(f"✅ Context embedder adapted successfully")
    return transformer


# ==================== 主程序 ====================
print("="*60)
print("SD3 Fine-tuned Model Inference")
print("="*60)


# 1. 加载基础模型
print("\n[1/5] Loading base SD3 model...")
pipe = StableDiffusion3Pipeline.from_single_file(
    "../checkpoint/sd3_medium_incl_clips.safetensors",
    config="sd3_config_cache",
    torch_dtype=torch.float16,
    local_files_only=True,
    text_encoder_3=None,
    tokenizer_3=None
)
pipe = pipe.to("cuda")
print(f"✓ Base model loaded")
print(f"  Scheduler: {pipe.scheduler.__class__.__name__}")


# 2. 设置正确的 scheduler（关键！）
print("\n[2/5] Setting up FlowMatchEulerDiscreteScheduler...")
pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
    pipe.scheduler.config,
    shift=3.0  # SD3 Medium 的推荐值
)
print(f"✓ Scheduler configured: {pipe.scheduler.__class__.__name__}")




# 4. 加载微调权重
print("\n[4/5] Loading fine-tuned weights...")
ft_path = "/CSTemp/yjl/R-SD/4/output_full_fast/checkpoint-10000/diffusion_pytorch_model.safetensors"
state_dict = load_file(ft_path)


missing, unexpected = pipe.transformer.load_state_dict(state_dict, strict=False)
print(f"  Missing keys: {len(missing)} - {missing[:3] if missing else '[]'}")
print(f"  Unexpected keys: {len(unexpected)} - {unexpected[:3] if unexpected else '[]'}")


# 验证权重确实改变了
finetuned_weight_sample = pipe.transformer.transformer_blocks[0].attn.to_q.weight[0, 0].item()
print(f"\n  Weight verification:")
print(f"    Original: {original_weight_sample.item():.6f}")
print(f"    Fine-tuned: {finetuned_weight_sample:.6f}")
print(f"    Changed: {'✓ YES' if abs(original_weight_sample.item() - finetuned_weight_sample) > 1e-6 else '✗ NO - WARNING!'}")


if len(missing) == 0 and len(unexpected) == 0:
    print("✓ Fine-tuned weights loaded successfully!")
else:
    print("⚠️ Warning: Key mismatch detected")



# 3. 适配 transformer
print("\n[3/5] Adapting transformer for 2-encoder input...")
# 保存原始权重用于对比
original_weight_sample = pipe.transformer.transformer_blocks[0].attn.to_q.weight[0, 0].clone()
pipe.transformer = adapt_transformer_for_two_encoders(pipe.transformer, new_in_features=2048)


# 5. 推理生成
print("\n[5/5] Generating image...")
print("="*60)


prompt = "a high quality X-ray image of scoliosis"
negative_prompt = "blurry, low quality, distorted"


# 使用更多步数和更高的 guidance
image = pipe(
    prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=50,  # 增加步数
    guidance_scale=7.0,
    generator=torch.Generator(device="cuda").manual_seed(42)  # 固定随机种子
).images[0]


image.save("gene.png")
print("\n✅ Image saved as gene.png")


# 额外：生成多张对比
print("\nGenerating comparison images...")
for i, steps in enumerate([28, 50, 100]):
    img = pipe(
        prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=7.0,
        generator=torch.Generator(device="cuda").manual_seed(42)
    ).images[0]
    img.save(f"gene_steps{steps}.png")
    print(f"  Saved gene_steps{steps}.png")


print("\n" + "="*60)
print("Done! Check the generated images.")
print("="*60)
