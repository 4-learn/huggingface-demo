# Hugging Face Demo

模型層（Model Responsibility）的 Demo 程式碼。

## 安裝

```bash
pip install -r requirements.txt
```

## 檔案說明

| 檔案 | 說明 |
|------|------|
| `01_hub_basics.py` | Hugging Face Hub 基礎操作 |
| `02_transformers_pipeline.py` | Transformers Pipeline 快速入門 |
| `03_model_interaction.py` | 模型載入與互動 |
| `04_text_generation.py` | 文本生成（Inference） |
| `05_embeddings.py` | 文本向量化（Embedding） |

## 執行

```bash
# 基礎範例
python 01_hub_basics.py

# Pipeline 範例
python 02_transformers_pipeline.py

# 文本生成
python 04_text_generation.py
```

## 注意事項

- 首次執行會下載模型，需要網路連線
- 某些模型需要較大記憶體（建議 8GB+ RAM）
- 可使用 `TRANSFORMERS_CACHE` 環境變數指定快取目錄
