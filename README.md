---
title: Medical Specialty AI Triage System
emoji: 🩺
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# 🩺 Multilingual AI Medical Specialty Predictor & Triage System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformers-yellow)](https://huggingface.co/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end, production-ready AI medical triage system designed to process patient symptoms in both **Arabic** and **English**, mapping them to **19 medical specialties** with **96% validation accuracy**.

---

## 🚀 Key Features

* **Dual-Stream Hybrid Architecture:** Combines `BioClinicalBERT` (for clinical semantics) and `AraBERT v02` (for Arabic morphological structure) via Late Fusion (Averaging).
* **Multilingual Capability:** Dynamically processes queries in Arabic and English, utilizing asynchronous translation fallback mechanisms.
* **Production-Grade Backend:** Powered by **FastAPI** featuring asynchronous endpoints, **Pydantic** schema validation, and **SlowAPI** rate limiting.
* **Deterministic Fallback Engine:** Features a secondary rule-based keyword matching algorithm to ensure reliability in ambiguous edge-case scenarios.
* **Interactive Web Interface:** A lightweight, responsive UI (Bootstrap 5) providing real-time specialty predictions and confidence scores.

---

## 🧠 Model Architecture & Pipeline

```text
               +---------------------------------+
               |     User Input Symptom Query    |
               +---------------------------------+
                                |
               +---------------------------------+
               |  Preprocessing & Translation    |
               +---------------------------------+
                                |
                 +--------------+--------------+
                 |                             |
                 v                             v
       +-------------------+         +--------------------+
       | BioClinicalBERT   |         | AraBERT v02        |
       | (Clinical Domain) |         | (Morphology/Arabic)|
       +-------------------+         +--------------------+
                 |                             |
                 +--------------+--------------+
                                |
                                v
               +---------------------------------+
               |   Late Fusion (Vector Avg 768)  |
               +---------------------------------+
                                |
                                v
               +---------------------------------+
               | Dropout (0.3) + Linear Classifier|
               +---------------------------------+
                                |
                                v
               +---------------------------------+
               |  Predict Specialty (19 Classes) |
               +---------------------------------+
Technical Specs:
Total Parameters: ~245M Parameters

Base Architecture: Transformer Encoders (BERT-Base, 12 Layers, 12 Attention Heads, 768 Hidden Size)

Optimization: AdamW Optimizer, Cross-Entropy Loss, Dropout = 0.3

Validation Accuracy: 96.2%

🛠️ Tech Stack
| Domain | Technologies Used |
| :--- | :--- |
| **Deep Learning & NLP** | PyTorch, Hugging Face Transformers, BioClinicalBERT, AraBERT |
| **Backend & API** | FastAPI, Uvicorn, Pydantic, SlowAPI |
| **Data Processing** | Pandas, NumPy, ThreadPoolExecutor, GoogleTranslator |
| **Frontend** | HTML5, CSS3, JavaScript, Bootstrap 5 |
| **Deployment** | Docker, Hugging Face Spaces |

⚡ Quick Start & Local Setup
Prerequisites
Python 3.10 or higher
Git

1. Clone the Repository
git clone [https://github.com/YOUR_USERNAME/med-triage-ai.git](https://github.com/YOUR_USERNAME/med-triage-ai.git)
cd med-triage-ai
2. Set Up Virtual Environment
Bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On Linux/macOS
python3 -m venv venv
source venv/bin/activate

3. Install Dependencies
Bash
pip install -r requirements.txt
4. Run the Application
Bash
uvicorn app.main:app --reload
Open your browser and navigate to http://127.0.0.1:8000 or check the interactive API docs at http://127.0.0.1:8000/docs.

🐳 Running with Docker
Bash
# Build the Docker image
docker build -t medical-triage-app .

# Run the container
docker run -p 7860:7860 medical-triage-app

🔮 Future Improvements
Transition to Multi-turn Dialogue system using Med-LLMs (e.g., Llama-3 fine-tuned on medical notes).

Integrate Voice-to-Text input support via OpenAI Whisper.

Connect with Google Maps API for geographic doctor recommendation based on location.

Full CI/CD pipeline deployment to AWS / Google Cloud Platform.

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
