"""
文本向量化（Embedding）

學習目標：
1. 理解什麼是 Embedding
2. 使用不同的 Embedding 模型
3. 計算文本相似度
4. Embedding 的應用場景
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


def mean_pooling(model_output, attention_mask):
    """
    Mean Pooling：將所有 token 的向量平均

    這是取得句子向量的常用方法
    """
    token_embeddings = model_output[0]  # (batch, seq_len, hidden_size)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def demo_basic_embedding():
    """基礎 Embedding"""
    print("=== 基礎 Embedding ===\n")

    # 使用專門為 sentence embedding 設計的模型
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    # 文字
    text = "Workers must wear safety helmets."

    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    # 取得 embedding
    with torch.no_grad():
        outputs = model(**inputs)

    # Mean pooling
    embedding = mean_pooling(outputs, inputs['attention_mask'])

    # L2 正規化（讓向量長度為 1）
    embedding = F.normalize(embedding, p=2, dim=1)

    print(f"文字: {text}")
    print(f"Embedding 維度: {embedding.shape}")
    print(f"向量前 10 維: {embedding[0][:10].tolist()}")
    print(f"向量長度 (應為 1): {torch.norm(embedding[0]).item():.4f}")


def demo_similarity():
    """計算文本相似度"""
    print("\n=== 文本相似度 ===\n")

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    # 待比較的句子
    sentences = [
        "Safety helmet is required in construction zone.",  # 基準句
        "Workers must wear hard hats on building sites.",   # 語義相近
        "The weather is sunny today.",                       # 完全不相關
        "PPE includes helmets and safety vests.",            # 部分相關
    ]

    # 批次 tokenize
    inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)

    # 取得 embeddings
    with torch.no_grad():
        outputs = model(**inputs)

    embeddings = mean_pooling(outputs, inputs['attention_mask'])
    embeddings = F.normalize(embeddings, p=2, dim=1)

    # 計算第一句與其他句子的相似度（Cosine Similarity）
    base_embedding = embeddings[0]

    print(f"基準句: {sentences[0]}\n")
    print("相似度比較:")
    for i, sentence in enumerate(sentences[1:], 1):
        # Cosine similarity = 內積（因為已正規化）
        similarity = torch.dot(base_embedding, embeddings[i]).item()

        bar = "█" * int(similarity * 20)
        print(f"  {similarity:.4f} {bar} {sentence}")


def demo_semantic_search():
    """語義搜尋範例"""
    print("\n=== 語義搜尋 ===\n")

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    # 模擬「知識庫」（工安相關法規/規定）
    knowledge_base = [
        "Workers in construction zones must wear safety helmets at all times.",
        "High visibility vests are required in areas with vehicle traffic.",
        "Safety glasses must be worn when operating power tools.",
        "Fire extinguishers should be placed at visible and accessible locations.",
        "Emergency exits must be clearly marked and unobstructed.",
        "First aid kits should be available within 100 meters of work areas.",
        "Ladders must be inspected before each use.",
        "Electrical equipment should be grounded properly.",
    ]

    # 取得知識庫 embeddings
    inputs = tokenizer(knowledge_base, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    kb_embeddings = mean_pooling(outputs, inputs['attention_mask'])
    kb_embeddings = F.normalize(kb_embeddings, p=2, dim=1)

    # 使用者查詢
    queries = [
        "What should I wear on my head?",
        "Where to put fire safety equipment?",
        "How to stay safe when using electric tools?",
    ]

    for query in queries:
        print(f"查詢: {query}")

        # 取得查詢 embedding
        query_inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            query_outputs = model(**query_inputs)
        query_embedding = mean_pooling(query_outputs, query_inputs['attention_mask'])
        query_embedding = F.normalize(query_embedding, p=2, dim=1)

        # 計算與所有文件的相似度
        similarities = torch.matmul(query_embedding, kb_embeddings.T)[0]

        # 取得最相關的結果
        top_k = 2
        top_indices = torch.topk(similarities, k=top_k).indices

        print(f"最相關的 {top_k} 條規定:")
        for idx in top_indices:
            print(f"  [{similarities[idx]:.3f}] {knowledge_base[idx]}")
        print()


def demo_clustering():
    """Embedding 用於聚類"""
    print("=== Embedding 聚類 ===\n")

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    # 違規事件描述
    violations = [
        # 安全帽相關
        "Worker detected without helmet",
        "Missing hard hat in zone A",
        "No helmet violation at entrance",
        # 背心相關
        "High visibility vest not worn",
        "No safety vest detected",
        "Reflective clothing missing",
        # 其他
        "Fire exit blocked",
        "Emergency door obstructed",
    ]

    # 取得 embeddings
    inputs = tokenizer(violations, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = mean_pooling(outputs, inputs['attention_mask'])
    embeddings = F.normalize(embeddings, p=2, dim=1)

    # 計算相似度矩陣
    similarity_matrix = torch.matmul(embeddings, embeddings.T)

    print("違規事件:")
    for i, v in enumerate(violations):
        print(f"  {i}: {v}")

    print("\n相似度矩陣 (簡化顯示):")
    print("事件之間語義相似度高的可以歸為同一類別")
    print()

    # 顯示高相似度配對
    print("高相似度配對 (>0.7):")
    for i in range(len(violations)):
        for j in range(i + 1, len(violations)):
            sim = similarity_matrix[i][j].item()
            if sim > 0.7:
                print(f"  [{sim:.3f}] {i} <-> {j}")
                print(f"    {violations[i]}")
                print(f"    {violations[j]}")
                print()


def demo_different_models():
    """比較不同 Embedding 模型"""
    print("=== 比較不同 Embedding 模型 ===\n")

    models_to_compare = [
        ("sentence-transformers/all-MiniLM-L6-v2", 384),   # 小型，快速
        ("sentence-transformers/all-mpnet-base-v2", 768),  # 中型，品質較高
    ]

    test_sentences = [
        "Safety helmet is mandatory.",
        "Wearing a hard hat is required.",
    ]

    for model_name, expected_dim in models_to_compare:
        print(f"模型: {model_name}")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()

        # 計算參數量
        num_params = sum(p.numel() for p in model.parameters())

        inputs = tokenizer(test_sentences, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = mean_pooling(outputs, inputs['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # 計算兩句話的相似度
        similarity = torch.dot(embeddings[0], embeddings[1]).item()

        print(f"  參數量: {num_params:,} ({num_params/1e6:.1f}M)")
        print(f"  向量維度: {embeddings.shape[1]}")
        print(f"  測試相似度: {similarity:.4f}")
        print()

        del model  # 釋放記憶體


# === 主程式 ===
if __name__ == "__main__":
    print("=" * 50)
    print("文本向量化（Embedding）")
    print("=" * 50)
    print()

    # 1. 基礎 Embedding
    demo_basic_embedding()

    # 2. 文本相似度
    demo_similarity()

    # 3. 語義搜尋
    demo_semantic_search()

    # 4. 聚類應用
    demo_clustering()

    # 5. 比較不同模型
    demo_different_models()

    print("\n完成！")
