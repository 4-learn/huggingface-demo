"""
Hugging Face Hub 基礎操作

學習目標：
1. 了解 Hugging Face Hub 是什麼
2. 搜尋和瀏覽模型
3. 下載模型和資料集
"""

from huggingface_hub import (
    list_models,
    model_info,
    hf_hub_download,
    snapshot_download,
)


def demo_list_models():
    """列出熱門模型"""
    print("=== 列出熱門文本生成模型 ===\n")

    # 搜尋文本生成模型，按下載數排序
    models = list_models(
        task="text-generation",
        sort="downloads",
        direction=-1,
        limit=5
    )

    for model in models:
        print(f"模型: {model.id}")
        print(f"  下載數: {model.downloads:,}")
        print(f"  按讚數: {model.likes:,}")
        print()


def demo_model_info():
    """查看模型資訊"""
    print("=== 查看模型詳細資訊 ===\n")

    # 查看 GPT-2 的資訊
    info = model_info("gpt2")

    print(f"模型 ID: {info.id}")
    print(f"作者: {info.author}")
    print(f"下載數: {info.downloads:,}")
    print(f"標籤: {info.tags[:5]}...")
    print(f"Pipeline 標籤: {info.pipeline_tag}")
    print(f"建立時間: {info.created_at}")


def demo_download_file():
    """下載單一檔案"""
    print("\n=== 下載模型設定檔 ===\n")

    # 只下載 config.json（不下載整個模型）
    config_path = hf_hub_download(
        repo_id="gpt2",
        filename="config.json"
    )

    print(f"檔案下載至: {config_path}")

    # 讀取內容
    import json
    with open(config_path) as f:
        config = json.load(f)

    print(f"模型類型: {config.get('model_type')}")
    print(f"詞彙大小: {config.get('vocab_size'):,}")
    print(f"隱藏層大小: {config.get('n_embd')}")


def demo_search_chinese_models():
    """搜尋中文模型"""
    print("\n=== 搜尋中文 NLP 模型 ===\n")

    # 搜尋中文相關模型
    models = list_models(
        search="chinese",
        task="text-classification",
        sort="downloads",
        direction=-1,
        limit=5
    )

    for model in models:
        print(f"模型: {model.id}")
        print(f"  下載數: {model.downloads:,}")
        print()


# === 主程式 ===
if __name__ == "__main__":
    print("=" * 50)
    print("Hugging Face Hub 基礎操作")
    print("=" * 50)
    print()

    # 1. 列出熱門模型
    demo_list_models()

    # 2. 查看模型資訊
    demo_model_info()

    # 3. 下載單一檔案
    demo_download_file()

    # 4. 搜尋中文模型
    demo_search_chinese_models()

    print("\n完成！")
