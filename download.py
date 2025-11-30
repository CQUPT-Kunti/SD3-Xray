from huggingface_hub import snapshot_download
import os

# 下载 SD3 的配置文件到本地目录
print("开始下载 SD3 配置文件...")

snapshot_download(
    repo_id="stabilityai/stable-diffusion-3-medium-diffusers",
    local_dir="sd3_config_cache1",  # 保存到这个目录
    ignore_patterns=[
        "*.safetensors",  # 忽略权重文件
        "*.bin",          # 忽略模型权重
        "*.fp16.*",       # 忽略 fp16 文件
        "*.msgpack",      # 忽略其他大文件
    ],
    allow_patterns=[
        "*.json",         # 只下载配置文件
        "*.txt",          # 文本文件
        "tokenizer/**",   # tokenizer 文件
    ]
)

print("配置文件下载完成！保存在 sd3_config_cache 目录")
