#!/usr/bin/env python3
"""
下载SD3配置文件到本地
只需运行一次，之后可完全离线使用
"""

import os
from huggingface_hub import snapshot_download

def main():
    cache_dir = "./sd3_config_cache"

    print("="*70)
    print("下载SD3配置文件")
    print("="*70)
    print(f"目标目录: {cache_dir}")
    print("这个过程只需要进行一次，之后可以完全离线使用")
    print()

    # 需要的配置文件
    patterns = [
        "tokenizer/*",
        "tokenizer_2/*",
        "tokenizer_3/*",
        "transformer/config.json",
        "vae/config.json",
        "text_encoder/config.json",
        "text_encoder_2/config.json",
        "text_encoder_3/config.json",
        "scheduler/scheduler_config.json",
    ]

    try:
        print("开始下载...")
        snapshot_download(
            repo_id="stabilityai/stable-diffusion-3-medium-diffusers",
            local_dir=cache_dir,
            allow_patterns=patterns,
        )

        print()
        print("="*70)
        print("✅ 配置文件下载完成！")
        print("="*70)
        print(f"保存在: {cache_dir}")
        print()
        print("现在可以使用以下命令训练（完全离线）:")
        print()
        print("python train_sd3_full_finetune_offline.py \\")
        print("    --pretrained_model_path sd3_medium_incl_clips.safetensors \\")
        print(f"    --sd3_config_dir {cache_dir} \\")
        print("    --instance_data_dir ./data \\")
        print("    --instance_prompt 'your prompt' \\")
        print("    ...")
        print("="*70)

    except Exception as e:
        print(f"❌ 下载失败: {e}")
        print()
        print("如果无法下载，请尝试:")
        print("1. 检查网络连接")
        print("2. 登录Hugging Face: huggingface-cli login")
        print("3. 或从其他地方获取完整的SD3模型目录")

if __name__ == "__main__":
    main()
