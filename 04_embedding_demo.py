"""
Demo：Embedding 概念

老師示範用：展示什麼是 Embedding、怎麼算相似度

執行前先安裝：
pip install sentence-transformers
"""

from sentence_transformers import SentenceTransformer, util

# 1. 載入模型
print("載入模型...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("完成！\n")

# 2. 準備 5 個違規描述
violations = [
    "工人未配戴安全帽",
    "員工沒有穿戴頭盔",
    "緊急出口被貨物阻擋",
    "電線裸露在地面上",
    "員工在禁煙區吸煙",
]

# 3. 轉成 Embedding（向量）
embeddings = model.encode(violations)

print(f"每個句子變成一個 {embeddings.shape[1]} 維的向量")
print(f"例如「{violations[0]}」的前 5 維：{embeddings[0][:5].tolist()}\n")

# 4. 查詢：輸入一句話，找最像的
query = "工人沒戴安全帽在高處作業"
query_embedding = model.encode(query)

# 計算相似度
scores = util.cos_sim(query_embedding, embeddings)[0]

print(f"查詢：「{query}」\n")
print("相似度排名：")

# 排序顯示
results = sorted(zip(violations, scores), key=lambda x: x[1], reverse=True)
for text, score in results:
    bar = "█" * int(score * 20)
    print(f"  {score:.2f} {bar} {text}")
