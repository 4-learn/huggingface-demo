"""
Transformers Pipeline 快速入門

學習目標：
1. 了解 Pipeline 是什麼
2. 使用不同任務的 Pipeline
3. 理解 Pipeline 的便利性與限制
"""

from transformers import pipeline


def demo_sentiment_analysis():
    """情感分析 Pipeline"""
    print("=== 情感分析 ===\n")

    # 建立情感分析 Pipeline
    classifier = pipeline("sentiment-analysis")

    # 測試句子
    texts = [
        "I love this product! It's amazing!",
        "This is terrible. I want a refund.",
        "It's okay, nothing special.",
    ]

    results = classifier(texts)

    for text, result in zip(texts, results):
        print(f"文字: {text}")
        print(f"結果: {result['label']} (信心度: {result['score']:.2%})")
        print()


def demo_text_generation():
    """文本生成 Pipeline"""
    print("=== 文本生成 ===\n")

    # 建立文本生成 Pipeline（使用較小的 GPT-2）
    generator = pipeline("text-generation", model="gpt2")

    # 生成文本
    prompt = "The safety helmet is important because"

    results = generator(
        prompt,
        max_length=50,
        num_return_sequences=2,
        do_sample=True,
        temperature=0.7,
    )

    print(f"提示: {prompt}\n")
    for i, result in enumerate(results, 1):
        print(f"生成 {i}: {result['generated_text']}")
        print()


def demo_fill_mask():
    """遮罩填充 Pipeline"""
    print("=== 遮罩填充 (BERT) ===\n")

    # 建立遮罩填充 Pipeline
    unmasker = pipeline("fill-mask", model="bert-base-uncased")

    # 測試句子（[MASK] 是要預測的位置）
    text = "Workers must wear a [MASK] on the construction site."

    results = unmasker(text)

    print(f"原句: {text}\n")
    print("預測結果:")
    for result in results[:5]:
        print(f"  {result['token_str']}: {result['score']:.2%}")


def demo_question_answering():
    """問答 Pipeline"""
    print("\n=== 問答系統 ===\n")

    # 建立問答 Pipeline
    qa = pipeline("question-answering")

    # 上下文和問題
    context = """
    Personal Protective Equipment (PPE) includes safety helmets,
    high-visibility vests, safety glasses, and protective gloves.
    Workers in construction zones must wear appropriate PPE at all times.
    Failure to comply may result in serious injuries or fatalities.
    """

    questions = [
        "What does PPE include?",
        "Who must wear PPE?",
        "What happens if workers don't wear PPE?",
    ]

    print(f"上下文: {context.strip()}\n")

    for question in questions:
        result = qa(question=question, context=context)
        print(f"問: {question}")
        print(f"答: {result['answer']} (信心度: {result['score']:.2%})")
        print()


def demo_summarization():
    """摘要 Pipeline"""
    print("=== 文本摘要 ===\n")

    # 建立摘要 Pipeline
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    # 長文本
    article = """
    Workplace safety is a critical concern in industrial environments.
    According to recent statistics, thousands of workers are injured
    annually due to inadequate safety measures. The most common causes
    of workplace injuries include falls from heights, being struck by
    objects, and exposure to harmful substances.

    To mitigate these risks, employers must implement comprehensive
    safety programs that include regular training sessions, proper
    equipment maintenance, and strict adherence to safety protocols.
    Personal Protective Equipment (PPE) such as helmets, gloves, and
    safety glasses should be provided and their use enforced.

    Regular safety audits and inspections can help identify potential
    hazards before they cause harm. Workers should also be encouraged
    to report unsafe conditions without fear of retaliation.
    """

    result = summarizer(
        article,
        max_length=60,
        min_length=20,
        do_sample=False
    )

    print(f"原文長度: {len(article)} 字元")
    print(f"摘要: {result[0]['summary_text']}")


def demo_zero_shot_classification():
    """零樣本分類 Pipeline"""
    print("\n=== 零樣本分類 ===\n")

    # 建立零樣本分類 Pipeline
    classifier = pipeline("zero-shot-classification")

    # 待分類文本
    text = "Worker detected without safety helmet in construction zone"

    # 候選標籤（不需要訓練！）
    candidate_labels = ["safety violation", "equipment malfunction", "normal operation"]

    result = classifier(text, candidate_labels)

    print(f"文字: {text}\n")
    print("分類結果:")
    for label, score in zip(result['labels'], result['scores']):
        bar = "█" * int(score * 20)
        print(f"  {label}: {bar} {score:.2%}")


# === 主程式 ===
if __name__ == "__main__":
    print("=" * 50)
    print("Transformers Pipeline 快速入門")
    print("=" * 50)
    print()
    print("注意：首次執行會下載模型，請稍候...\n")

    # 1. 情感分析
    demo_sentiment_analysis()

    # 2. 文本生成
    demo_text_generation()

    # 3. 遮罩填充
    demo_fill_mask()

    # 4. 問答
    demo_question_answering()

    # 5. 摘要
    demo_summarization()

    # 6. 零樣本分類
    demo_zero_shot_classification()

    print("\n完成！")
