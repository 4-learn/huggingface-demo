"""
模型載入與互動

學習目標：
1. 手動載入模型和 tokenizer
2. 理解 tokenization 過程
3. 直接與模型互動（不用 Pipeline）
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
)


def demo_tokenizer_basics():
    """Tokenizer 基礎"""
    print("=== Tokenizer 基礎 ===\n")

    # 載入 tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # 原始文字
    text = "Workers must wear safety helmets."

    # Tokenization
    tokens = tokenizer.tokenize(text)
    print(f"原文: {text}")
    print(f"Tokens: {tokens}")

    # 轉換為 ID
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    print(f"Token IDs: {token_ids}")

    # 使用 encode（一步完成）
    encoded = tokenizer.encode(text, add_special_tokens=True)
    print(f"Encoded (含特殊 token): {encoded}")

    # 解碼回文字
    decoded = tokenizer.decode(encoded)
    print(f"Decoded: {decoded}")


def demo_tokenizer_batch():
    """批次處理與 Padding"""
    print("\n=== 批次 Tokenization ===\n")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # 多個句子（長度不同）
    texts = [
        "Safety first.",
        "Workers must wear helmets.",
        "Personal Protective Equipment is required in all construction zones.",
    ]

    # 批次 tokenization（自動 padding）
    batch = tokenizer(
        texts,
        padding=True,      # 補齊到最長
        truncation=True,   # 截斷超長文字
        max_length=20,
        return_tensors="pt"  # 回傳 PyTorch tensor
    )

    print("批次處理結果:")
    print(f"  input_ids shape: {batch['input_ids'].shape}")
    print(f"  attention_mask shape: {batch['attention_mask'].shape}")

    print("\n各句子 token 數:")
    for i, text in enumerate(texts):
        # attention_mask 中 1 的數量 = 實際 token 數
        actual_tokens = batch['attention_mask'][i].sum().item()
        print(f"  {i+1}. {text[:30]}... -> {actual_tokens} tokens")


def demo_model_embeddings():
    """取得文字嵌入"""
    print("\n=== 模型嵌入 (Embeddings) ===\n")

    # 載入模型和 tokenizer
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # 設為評估模式（不計算梯度）
    model.eval()

    # 準備輸入
    text = "Safety helmet violation detected."
    inputs = tokenizer(text, return_tensors="pt")

    # 前向傳播
    with torch.no_grad():
        outputs = model(**inputs)

    # outputs 包含:
    # - last_hidden_state: 每個 token 的向量 (batch, seq_len, hidden_size)
    # - pooler_output: [CLS] token 的向量 (batch, hidden_size)

    print(f"輸入文字: {text}")
    print(f"last_hidden_state shape: {outputs.last_hidden_state.shape}")
    print(f"pooler_output shape: {outputs.pooler_output.shape}")

    # 取得句子向量（使用 [CLS] token）
    sentence_embedding = outputs.pooler_output[0]
    print(f"句子向量維度: {sentence_embedding.shape}")
    print(f"向量前 5 維: {sentence_embedding[:5].tolist()}")


def demo_classification_model():
    """分類模型互動"""
    print("\n=== 分類模型 ===\n")

    # 載入預訓練的情感分類模型
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    model.eval()

    # 測試句子
    texts = [
        "This safety system is excellent!",
        "The equipment failed again.",
    ]

    for text in texts:
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt")

        # 推理
        with torch.no_grad():
            outputs = model(**inputs)

        # 取得預測
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

        # 模型的標籤
        labels = ["NEGATIVE", "POSITIVE"]

        print(f"文字: {text}")
        print(f"預測: {labels[predicted_class]}")
        print(f"機率: NEGATIVE={probabilities[0][0]:.2%}, POSITIVE={probabilities[0][1]:.2%}")
        print()


def demo_model_comparison():
    """比較不同模型大小"""
    print("=== 模型大小比較 ===\n")

    models_to_compare = [
        "prajjwal1/bert-tiny",      # 4.4M 參數
        "prajjwal1/bert-mini",      # 11.3M 參數
        "bert-base-uncased",         # 110M 參數
    ]

    for model_name in models_to_compare:
        try:
            model = AutoModel.from_pretrained(model_name)

            # 計算參數數量
            num_params = sum(p.numel() for p in model.parameters())

            print(f"模型: {model_name}")
            print(f"  參數數量: {num_params:,}")
            print(f"  約: {num_params / 1e6:.1f}M")
            print()

            del model  # 釋放記憶體

        except Exception as e:
            print(f"模型 {model_name} 載入失敗: {e}")


# === 主程式 ===
if __name__ == "__main__":
    print("=" * 50)
    print("模型載入與互動")
    print("=" * 50)
    print()

    # 1. Tokenizer 基礎
    demo_tokenizer_basics()

    # 2. 批次處理
    demo_tokenizer_batch()

    # 3. 模型嵌入
    demo_model_embeddings()

    # 4. 分類模型
    demo_classification_model()

    # 5. 模型比較
    demo_model_comparison()

    print("\n完成！")
