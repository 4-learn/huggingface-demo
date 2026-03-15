"""
Demo：Hugging Face 基礎

老師示範用：Pipeline 推論 + 零樣本分類 + 語義搜尋

執行前先安裝：
pip install transformers torch sentence-transformers
"""

from transformers import pipeline

# ========================
# 1. Pipeline 基本使用
# ========================
print("=== 1. Pipeline 基本使用（情感分析）===\n")

classifier = pipeline("sentiment-analysis")
result = classifier("I love this product!")
print(result)
# [{'label': 'POSITIVE', 'score': 0.9998}]

print()

# ========================
# 2. 零樣本分類
# ========================
print("=== 2. 零樣本分類（不用訓練就能分類）===\n")

classifier = pipeline("zero-shot-classification")
text = "工人沒有配戴安全帽"
labels = ["safety_violation", "equipment_issue", "normal_operation"]
result = classifier(text, labels)

print(f"文字：{text}")
print(f"分類：{result['labels'][0]}")
print(f"信心度：{result['scores'][0]:.2%}")

print()

# ========================
# 3. 語義搜尋（Embedding）
# ========================
print("=== 3. 語義搜尋 ===\n")

from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

regulations = [
    "雇主應使勞工確實使用安全帽",
    "有車輛出入場所應穿著反光衣",
    "緊急出口應保持暢通",
]

query = "沒有戴安全帽"

reg_embeddings = model.encode(regulations)
query_embedding = model.encode(query)

scores = util.cos_sim(query_embedding, reg_embeddings)[0]

print(f"查詢：「{query}」\n")
results = sorted(zip(regulations, scores), key=lambda x: x[1], reverse=True)
for text, score in results:
    print(f"  {score:.2f}  {text}")
