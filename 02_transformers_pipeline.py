"""
Demo：Hugging Face Transformers 進階控制

老師示範用：指定模型 + 調整參數 + 不同 task 類型

執行前先安裝：
pip install transformers torch
"""

from transformers import pipeline

# ========================
# 1. 指定模型：同一個 task，換模型結果不同
# ========================
print("=== 1. 指定模型 ===\n")

# 預設模型
classifier_default = pipeline("sentiment-analysis")
result1 = classifier_default("I love this product!")
print(f"預設模型: {result1[0]['label']} ({result1[0]['score']:.2%})")

# 指定其他模型
classifier_custom = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
result2 = classifier_custom("I love this product!")
print(f"指定模型: {result2[0]['label']} ({result2[0]['score']:.2%})")
# 這個模型用 1~5 星評分，不是 POSITIVE/NEGATIVE

print()

# ========================
# 2. 調整參數：temperature 影響生成結果
# ========================
print("=== 2. 調整參數（文字生成）===\n")

generator = pipeline("text-generation", model="gpt2")
prompt = "Safety helmets are important because"

# 低 temperature → 保守、重複
result_low = generator(prompt, max_length=40, temperature=0.1, do_sample=True)
print(f"temperature=0.1: {result_low[0]['generated_text']}")

# 高 temperature → 多樣、發散
result_high = generator(prompt, max_length=40, temperature=1.5, do_sample=True)
print(f"temperature=1.5: {result_high[0]['generated_text']}")

print()

# ========================
# 3. 不同 task：遮罩填充（fill-mask）
# ========================
print("=== 3. 遮罩填充（BERT 怎麼理解語言）===\n")

unmasker = pipeline("fill-mask", model="bert-base-uncased")
text = "Workers must wear a [MASK] on the construction site."

results = unmasker(text)

print(f"原句: {text}\n")
print("BERT 預測 [MASK] 是什麼：")
for r in results[:5]:
    print(f"  {r['token_str']:12s} ({r['score']:.2%})")
