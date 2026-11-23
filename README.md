# ✨ 6-Class Emotion Classification Studio

An end-to-end NLP emotion classification web application built with **RoBERTa-Large (LoRA)** fine-tuned on **GoEmotions** (`checkpoint-4310`), featuring an interactive **Streamlit** interface and automatic model loading from **Hugging Face Hub**.

## 📌 Overview

This project classifies English text into **6 core emotion categories**:
- 😄 **Joy**
- 😢 **Sadness**
- 😠 **Anger**
- 😨 **Fear**
- 😲 **Surprise**
- 🤢 **Disgust**

## 📊 Model & Evaluation Metrics

- **Base Architecture:** `roberta-large` with **LoRA** fine-tuning on **GoEmotions** (`checkpoint-4310`)
- **Accuracy:** `82.88%`
- **Macro F1 Score:** `0.7512`
- **Model Repository:** [`mo123essam/emotion-roberta-large-6class`](https://huggingface.co/mo123essam/emotion-roberta-large-6class)

## 🗂️ Project Structure

```text
├── app.py              # Streamlit application & inference pipeline
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
