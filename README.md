# 💬 Emotion Classification Using Large Language Models

This is a final group project submitted for the course **DTSC 5082: Data Science Capstone** at the University of North Texas.  
We built an end-to-end pipeline to classify **emotions from text data** using **Large Language Models (LLMs)** like **Mistral-7B** and **LLaMA 3.1 8B**.  
We evaluated the performance of these models under **zero-shot**, **few-shot**, and **fine-tuned** settings on both labeled datasets and real-world YouTube comments.

---

## 🔍 Project Summary

This project evaluates the ability of modern Large Language Models (LLMs) to classify emotions in text using various learning strategies: **zero-shot**, **few-shot**, and **fine-tuned** approaches.

We experimented with two cutting-edge open-source LLMs — **Mistral-7B** and **LLaMA 3.1 8B** — to classify text into six core emotions: **joy, sadness, anger, fear, love, and surprise**. 

Key highlights:
- Used the **Kaggle emotion dataset** for training and benchmarking
- Fine-tuned models using **QLoRA + HuggingFace Transformers** on a constrained GPU environment
- Applied the best models to **10,000+ real YouTube comments** for real-world evaluation
- Performed detailed analysis of **accuracy, F1-score, invalid predictions**, and **confusion matrices**
- Explored model agreement and disagreement between LLaMA and Mistral

> Fine-tuned LLaMA 3.1 achieved over **90% accuracy**, showing the strength of efficient finetuning even in noisy, emoji-rich social media data.


### 🧠 Goal
To assess how well LLMs can understand and classify human emotions from natural language text — especially in noisy, user-generated content.

### 📊 Emotions Predicted
- Joy
- Sadness
- Anger
- Fear
- Love
- Surprise

### 🛠 Techniques Used
- **Zero-shot & Few-shot prompting**
- **Fine-tuning using QLoRA/LoRA**
- **Open-source LLMs**: Mistral-7B, LLaMA 3.1
- **Evaluation**: Accuracy, F1 Score, Confusion Matrix
- **Real-world inference** on 10,000+ YouTube comments

---

## 👨‍💻 My Role (Varun Kumar Atkuri)

I actively contributed to the design, implementation, and evaluation of the full project:

- ✅ Conducted experiments using zero-shot, few-shot, and fine-tuned models
- ✅ Fine-tuned LLaMA 3.1 and Mistral using QLoRA (low-rank adaptation)
- ✅ Developed reusable Python modules:
  - `config.py`, `data_loader.py`, `metrics.py`, `model_utils.py`
- ✅ Designed the inference pipeline for 10,000+ YouTube comments
- ✅ Generated and validated confusion matrices and model reports
- ✅ Performed manual analysis of model predictions and error types
- ✅ Created structured project layout and `README.md` for GitHub
- ✅ Collaborated with team for final report documentation

> 🧪 This project improved my practical understanding of prompt engineering, fine-tuning LLMs, and real-world evaluation using transformer models.

---

## 🧪 Project Files

| Folder | Contents |
|--------|----------|
| `notebooks/` | Jupyter notebooks for zero-shot, few-shot, fine-tuning, and inference |
| `scripts/` | Python modules for training, evaluation, and data handling |
| `results/` | Evaluation results, confusion matrices, and output files |
| `report/` | Final project report (PDF) |
| `assets/` | (Optional) Visuals for README or presentation |
| `README.md` | This documentation file |

---

## 🧾 Key Results

| Model        | Mode        | Accuracy | F1 Score | Invalids |
|--------------|-------------|----------|----------|----------|
| Mistral-7B   | Zero-shot   | 53.9%    | 52.7%    | 93       |
| Mistral-7B   | Few-shot    | 46.8%    | 49.4%    | 147      |
| Mistral-7B   | Fine-tuned  | 89.3%    | 89.0%    | 0        |
| **LLaMA 3.1**| Fine-tuned  | **90.9%**| **90.7%**| 0        |

- LLaMA 3.1 achieved **state-of-the-art accuracy** after fine-tuning with QLoRA
- Fine-tuned models outperformed prompting-only settings significantly
- The models handled real-world, emoji-rich, and noisy YouTube comments effectively

---

## 📄 Final Report

The full project paper is available in `report/DTSC5082.401_Group8_Final_Report.pdf`.

It includes:
- Literature review
- Model architectures
- Training and testing approach
- Result visualizations
- Error analysis
- Conclusions and future work


## 👨‍👩‍👧‍👦 Team Members

This project was completed as part of a group submission for DTSC 5082 at the University of North Texas.

- Irfan Ahmed Shaik  
- Varun Kumar Atkuri (me) 
- Bhavyaraj Nadimipalli  
