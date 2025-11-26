# 해먹고말지 – 레시피 추천 서비스

본 프로젝트는 **React(프론트)** – **Express(Node.js 백엔드)** – **Flask 모델 서버**로 구성된  
레시피 추천 시스템입니다. 그래프 기반 RAG, Embedding 검색, LLM 기반 추천을 활용하여  
사용자의 프롬프트와 상황에 맞는 레시피를 제공합니다.

---

## 📁 폴더 구조
미니프로젝트2
├─ client/          # React 프론트엔드
├─ backend/         # Node.js(Express) 서버
├─ model-server/    # Flask 모델 서버 (conda 환경)
└─ codes/           # 데이터 전처리 및 유틸 파이썬 코드

---

## 🚀 실행 순서 요약

1. **client (React)** – 사용자 인터페이스
2. **backend (Express)** – 프론트 요청 → 모델 서버 연동  
3. **model-server (Flask)** – 모델 / 임베딩 / RAG 검색 담당  

---

# 1️⃣ Client (React)

### 📌 설치

```bash
cd client
npm install
npm start
```

---

# 2️⃣ Backend (Express)

### 📌 설치

```bash
cd backend
npm install
npm start
```

---

# 3️⃣ Model Server (Flask)

### 📌 설치

```bash
cd model-server
conda create -n recipe-model python=3.10 -y
conda activate recipe-model
pip install -r requirements.txt
python app.py
```

※ 모델 서버는 conda 환경에서만 실행하는 것을 권장합니다.
* Huggingface에서 Qwen-14B-Instruct 모델을 사용합니다. model-server/ 디렉토리 안에 다운로드 받는 것을 권장합니다.
---

### .env 파일 (openAI API 활용시 필요)
"model-server/.env" 파일을 생성하고, 아래 내용을 추가합니다.
```
OPENAI_API_KEY=your_openai_api_key
```

🗂 기술 스택

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
	•	Conda 환경 기반

Others
	•	Preprocessing scripts (Python)
	•	Embedding / Keyword extraction
	•	Graph construction
