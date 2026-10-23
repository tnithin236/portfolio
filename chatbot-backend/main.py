import asyncio
import os
import time
from collections import defaultdict

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]          # set on your host, never in the HTML
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
# Tried in order if the main model is busy (503) or not found (404)
FALLBACK_MODELS = [m.strip() for m in os.getenv("GEMINI_FALLBACK_MODELS", "gemini-3.8-flash").split(",") if m.strip()]
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")  # set to your portfolio URL in production

SYSTEM_PROMPT = """You are the AI assistant on Nithin T's portfolio website. Answer questions about Nithin
using ONLY the facts below. Be concise, friendly and professional. If something is not covered, say you don't
know and suggest emailing nithint236@gmail.com. Never invent facts.

FACTS
- AI & Data Science B.Tech (2022-2026), United Institute of Technology, Tamil Nadu. CGPA 7.44. Based in Coimbatore.
- Open to full-time roles, internships and research collaborations in AI/ML, data analytics, full-stack.
- Published: Multi-class brain tumor MRI classification with ResNet50V2, Grad-CAM, two-stage augmentation
  (IJRASET Vol 14 Issue IV, April 2026, DOI 10.22214/ijraset.2026.79816). Accuracy 89.68%, macro-AUC 0.9820.
- Projects: Hospital Management System (ASP.NET Core, React, PostgreSQL, Docker, JWT roles, 11 entities);
  Construction PPE Detection (YOLO, DETR, OpenCV, 92% real-world accuracy); RAG Knowledge Assistant (FastAPI,
  Pinecone, Gemini, React); Vehicle Number Plate Detection (DETR, OpenCV, OCR); Predictive Maintenance & Anomaly Detection System (XGBoost, Isolation Forest, SHAP, FastAPI, Streamlit).
- More GitHub projects (see the All Projects list): Solar Filament Segmentation, Car Damage Detection System,
  Predictive Maintenance for Jet Engines, Mall Customer Segmentation, Pneumonia Detection from Chest X-rays,
  Construction Analytics, Construction Site Helmet Detection, Delivery Delay Prediction, HealthRisk AI,
  AI-Powered Resume Screening and Job Matching, HotelGuard, Breast Cancer Detection with ResNet-50 V2,
  FraudShield, AI Spam Email Detection, Helmet Detection, Loan Default Prediction, Medical Insurance Cost
  Prediction, Bank Customer Credit Analysis. Only the names are known, so do not invent details about them.
- Experience: Data Science Trainee at Q Spiders (Feb-Aug 2026); AI/ML Intern at Neuronest AI (Jun-Aug 2025);
  Data Science Intern at UptoSkills (May-Jul 2025).
- Skills: Python, C#, JavaScript, TensorFlow, Scikit-learn, CNNs, Generative AI, RAG, FastAPI, ASP.NET Core,
  React, EF Core, PostgreSQL, Oracle SQL, Pinecone, Docker, Power BI, DAX, Pandas, NumPy, YOLO, DETR, OpenCV, OCR.
- Leadership: Vice President, Rotaract Club of United Institute of Technology (2024-25); member, Rotaract Club of
  R.S. Puram. Presented at ICIRES 2025.
- Contact: nithint236@gmail.com, GitHub tnithin236, LinkedIn nithin-t-59b58727b.
"""

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_methods=["POST"], allow_headers=["*"])

class Turn(BaseModel):
    role: str
    text: str = Field(max_length=2000)

class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=500)
    history: list[Turn] = []

# Simple per-IP rate limit (20 requests / 10 min) to protect your API quota
hits: dict[str, list[float]] = defaultdict(list)

@app.post("/chat")
async def chat(body: ChatIn, request: Request):
    ip = request.client.host if request.client else "unknown"
    now = time.time()
    hits[ip] = [t for t in hits[ip] if now - t < 600]
    if len(hits[ip]) >= 20:
        raise HTTPException(429, "Too many messages, please try again later.")
    hits[ip].append(now)

    contents = [
        {"role": "model" if t.role == "model" else "user", "parts": [{"text": t.text}]}
        for t in body.history[-8:]
    ]
    contents.append({"role": "user", "parts": [{"text": body.message}]})

    payload = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": contents,
        "generationConfig": {"maxOutputTokens": 1024, "temperature": 0.4},
    }
    r = None
    # Short timeout per try so a slow or busy model never hangs the chat; every failure falls through to the next model
    async with httpx.AsyncClient(timeout=httpx.Timeout(15.0, connect=8.0)) as client:
        for model in [GEMINI_MODEL] + FALLBACK_MODELS:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
            for attempt in range(3):
                try:
                    r = await client.post(url, params={"key": GEMINI_API_KEY}, json=payload)
                except httpx.HTTPError as e:           # timeout, network error, etc.
                    print("GEMINI NETWORK ERROR", model, type(e).__name__)
                    r = None
                    break                              # don't wait again on a slow model; try the next one
                if r.status_code == 200:
                    break
                print("GEMINI ERROR", model, r.status_code, r.text[:300])  # never includes your key
                if r.status_code in (429, 500, 503) and attempt < 2:
                    await asyncio.sleep(1.5 * (attempt + 1))   # busy: wait a little longer each time, then move on
                else:
                    break                              # 404 / 400 / 403: go straight to the next model
            if r is not None and r.status_code == 200:
                break
    if r is None or r.status_code != 200:
        raise HTTPException(503, "The AI assistant is busy right now. Please try again in a moment.")
    try:
        reply = r.json()["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        reply = "I couldn't generate an answer to that. Please email nithint236@gmail.com."
    return {"reply": reply}

@app.get("/health")
def health():
    return {"ok": True}
