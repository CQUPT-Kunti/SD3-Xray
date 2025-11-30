import torch
from diffusers import StableDiffusion3Pipeline, FlowMatchEulerDiscreteScheduler
from safetensors.torch import load_file
from train import adapt_transformer_for_two_encoders

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

# 3. 适配 transformer
print("\n[3/5] Adapting transformer for 2-encoder input...")
# 保存原始权重用于对比
original_weight_sample = pipe.transformer.transformer_blocks[0].attn.to_q.weight[0, 0].clone()
pipe.transformer = adapt_transformer_for_two_encoders(pipe.transformer, new_in_features=2048)

# 4. 加载微调权重
print("\n[4/5] Loading fine-tuned weights...")
ft_path = "/CSTemp/yjl/R-SD/4/output/checkpoint-20000/diffusion_pytorch_model.safetensors"
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
