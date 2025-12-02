# Eat Smart! – AI Recipe Recommendation Service

This project is a recipe recommendation system built with
React (frontend) – Express (Node.js backend) – Flask (model server).

It uses graph-based RAG, embedding search, and LLM-based recommendation
to suggest recipes tailored to the user’s natural-language prompt and current situation.

---

## Project Structure
```
mini_project2/
├─ client/          # React frontend
├─ backend/         # Node.js (Express) server
├─ model-server/    # Flask model server (conda environment)
└─ codes/           # Data preprocessing & utility Python scripts
```


---

## 🚀 Quick Start (Run Order)

1. **client (React)** – User Interface
2. **backend (Express)** – Receives requests from the frontend and calls the model server
3. **model-server (Flask)** – Handles models, embeddings, and RAG-based search

---

# 1️⃣ Client (React)

### 📌 Setup

```bash
cd client
npm install
npm start
```

---

# 2️⃣ Backend (Express)

### 📌 Setup

```bash
cd backend
npm install
npm start
```

---

# 3️⃣ Model Server (Flask)

### 📌 Setup

```bash
cd model-server
conda create -n recipe-model python=3.10 -y
conda activate recipe-model
pip install -r requirements.txt
python app.py
```

※ The model server is recommended to run inside a conda environment.
* We use the Qwen-14B-Instruct model from Hugging Face. It is recommended to download the model inside the model-server/ directory.
---

### .env File (Required when using OpenAI API)
Create a file at model-server/.env and add:
```
OPENAI_API_KEY=your_openai_api_key
```

🗂 Tech Stack

Frontend
	•	React
	•	Fetch API
	•	React Hooks

Backend
	•	Node.js
	•	Express.js
	•	REST API
	•	Flask 연동

Model Server
	•	Python 3.10+
	•	Flask
	•	Transformers / FAISS / Graph RAG 관련 라이브러리
	•	Conda Environment

Others
	•	Preprocessing scripts (Python)
	•	Embedding / Keyword extraction
	•	Graph construction
