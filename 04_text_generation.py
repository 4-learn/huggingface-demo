"""
文本生成（Inference）

學習目標：
1. 使用 Causal LM 生成文本
2. 理解生成參數（temperature, top_k, top_p）
3. 控制生成品質
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig


def demo_basic_generation():
    """基礎文本生成"""
    print("=== 基礎文本生成 ===\n")

    # 載入 GPT-2（小型模型，方便示範）
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    model.eval()

    # 設定 pad token（GPT-2 沒有預設）
    tokenizer.pad_token = tokenizer.eos_token

    # 提示詞
    prompt = "Workplace safety is important because"

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt")

    # 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,  # 貪婪解碼（確定性）
        )

    # 解碼
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"提示: {prompt}")
    print(f"生成: {generated_text}")


def demo_sampling_parameters():
    """採樣參數實驗"""
    print("\n=== 採樣參數比較 ===\n")

    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    prompt = "A safety helmet protects"
    inputs = tokenizer(prompt, return_tensors="pt")

    # 不同的採樣策略
    strategies = [
        {
            "name": "Greedy (do_sample=False)",
            "params": {"do_sample": False},
        },
        {
            "name": "Temperature=0.3 (保守)",
            "params": {"do_sample": True, "temperature": 0.3},
        },
        {
            "name": "Temperature=0.9 (創意)",
            "params": {"do_sample": True, "temperature": 0.9},
        },
        {
            "name": "Top-K=50",
            "params": {"do_sample": True, "top_k": 50, "temperature": 0.7},
        },
        {
            "name": "Top-P=0.9 (Nucleus)",
            "params": {"do_sample": True, "top_p": 0.9, "temperature": 0.7},
        },
    ]

    print(f"提示: {prompt}\n")

    for strategy in strategies:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                **strategy["params"],
            )

        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"{strategy['name']}:")
        print(f"  {text}")
        print()


def demo_generation_config():
    """使用 GenerationConfig"""
    print("=== GenerationConfig 設定 ===\n")

    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    # 建立生成設定
    gen_config = GenerationConfig(
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,  # 避免重複
        no_repeat_ngram_size=2,  # 禁止重複 2-gram
    )

    prompt = "The construction site safety rules include:"
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(**inputs, generation_config=gen_config)

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"提示: {prompt}")
    print(f"\n生成: {text}")


def demo_batch_generation():
    """批次生成"""
    print("\n=== 批次生成 ===\n")

    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # 生成時從左邊 padding
    model.eval()

    # 多個提示
    prompts = [
        "Safety helmets are designed to",
        "High visibility vests help workers",
        "Construction zone accidents can be prevented by",
    ]

    # 批次 tokenize
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)

    # 批次生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
        )

    # 解碼
    print("批次生成結果:\n")
    for i, output in enumerate(outputs):
        text = tokenizer.decode(output, skip_special_tokens=True)
        print(f"{i+1}. {text}")
        print()


def demo_stop_sequences():
    """停止序列"""
    print("=== 停止序列 ===\n")

    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    prompt = "List three safety rules:\n1."
    inputs = tokenizer(prompt, return_tensors="pt")

    # 定義停止 token（遇到換行就停）
    stop_token_id = tokenizer.encode("\n")[0]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=True,
            temperature=0.7,
            eos_token_id=stop_token_id,  # 自訂停止 token
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"提示: {prompt}")
    print(f"生成（到換行停止）: {text}")


def demo_multiple_generations():
    """生成多個版本"""
    print("\n=== 生成多個版本 ===\n")

    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    prompt = "A safety inspection found that"
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=25,
            do_sample=True,
            temperature=0.8,
            num_return_sequences=3,  # 生成 3 個版本
        )

    print(f"提示: {prompt}\n")
    print("生成的 3 個版本:")
    for i, output in enumerate(outputs, 1):
        text = tokenizer.decode(output, skip_special_tokens=True)
        print(f"\n版本 {i}: {text}")


# === 主程式 ===
if __name__ == "__main__":
    print("=" * 50)
    print("文本生成（Inference）")
    print("=" * 50)
    print()

    # 1. 基礎生成
    demo_basic_generation()

    # 2. 採樣參數
    demo_sampling_parameters()

    # 3. GenerationConfig
    demo_generation_config()

    # 4. 批次生成
    demo_batch_generation()

    # 5. 停止序列
    demo_stop_sequences()

    # 6. 多版本生成
    demo_multiple_generations()

    print("\n完成！")
