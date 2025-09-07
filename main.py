from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from openai import OpenAI
from dotenv import load_dotenv
import os, re, json, math
import fitz  # pymupdf
import docx
from io import BytesIO

load_dotenv()

# Initialize OpenAI client (expects OPENAI_API_KEY in env)
client = OpenAI()

app = FastAPI(title="theAgent (embeddings)", version="0.3")

# -----------------------
# Utilities
# -----------------------
def normalize_text(text: str) -> str:
  text = (text or "").lower()
  text = re.sub(r"[^a-z0-9\s]", " ", text)
  text = re.sub(r"\s+", " ", text).strip()
  return text

def extract_text_from_bytes(raw_bytes: bytes, filename: str = "") -> str:
  if not raw_bytes:
    return ""

  if raw_bytes.startswith(b"%PDF-"):
    try:
      text_parts = []
      doc = fitz.open(stream=raw_bytes, filetype="pdf")
      for page in doc:
        page_text = page.get_text()
        if page_text:
          text_parts.append(page_text)
      doc.close()
      return normalize_text("\n".join(text_parts))
    except Exception as e:
      raise HTTPException(status_code=415, detail=f"PDF parse error for {filename}: {e}")

  if raw_bytes[:2] == b"PK":
    try:
      doc = docx.Document(BytesIO(raw_bytes))
      paragraphs = [p.text for p in doc.paragraphs if p.text]
      return normalize_text("\n".join(paragraphs))
    except Exception as e:
      raise HTTPException(status_code=415, detail=f"DOCX parse error for {filename}: {e}")

  try:
    text = raw_bytes.decode("utf-8", errors="ignore")
    return normalize_text(text)
  except Exception:
    raise HTTPException(status_code=415, detail=f"Unsupported or corrupted file: {filename}")

def safe_parse_llm_json(content: str) -> Dict[str, Any]:
  if content is None:
    return {"raw_output": None}
  s = content.strip()
  s = re.sub(r"^```[a-zA-Z]*\n", "", s)
  s = re.sub(r"\n```$", "", s)
  s = s.strip()
  try:
    return json.loads(s)
  except json.JSONDecodeError:
    return {"raw_output": s}

# -----------------------
# Embeddings helpers
# -----------------------
def get_embedding(text: str, model: str = "text-embedding-3-small") -> Optional[List[float]]:
  if not text:
    return None
  try:
    resp = client.embeddings.create(model=model, input=text)
    # New OpenAI SDK: resp.data is a list, first item has 'embedding'
    emb = resp.data[0].embedding
    return emb
  except Exception as e:
    # don't raise; return None so caller can fall back
    return None

def cosine_similarity(a: List[float], b: List[float]) -> float:
  if not a or not b or len(a) != len(b):
    return 0.0
  dot = sum(x*y for x,y in zip(a,b))
  na = math.sqrt(sum(x*x for x in a))
  nb = math.sqrt(sum(x*x for x in b))
  if na == 0 or nb == 0:
    return 0.0
  return dot / (na * nb)

# -----------------------
# Endpoint
# -----------------------
@app.post("/score_resume/")
async def score_resume(file: UploadFile = File(...), job_description: str = Form(...)):
  raw = await file.read()
  resume_text = extract_text_from_bytes(raw, file.filename)

  # Compute embeddings for job description and resume (best-effort)
  jd_clean = job_description[:20000]  # limit length for embeddings
  resume_clean = resume_text[:20000]
  jd_emb = get_embedding(jd_clean)
  resume_emb = get_embedding(resume_clean)
  embedding_score = None
  if jd_emb and resume_emb:
    sim = cosine_similarity(jd_emb, resume_emb)
    embedding_score = round(sim * 100, 2)  # 0-100

  prompt = f"""
  You are an AI hiring assistant.
  Compare this resume against the job description.
  Respond ONLY in JSON with the following structure:

  {{
    "score": <number 0-100>,
    "strengths": ["point1", "point2", "point3"],
    "weaknesses": ["point1", "point2", "point3"],
    "fit": "High" | "Medium" | "Low"
  }}

  Job Description:
  {job_description}

  Resume:
  {resume_text[:6000]}
  """

  resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=500
  )
  parsed = safe_parse_llm_json(resp.choices[0].message.content)

  # Compute final_score:
  final_score = None
  llm_score = None
  if isinstance(parsed, dict) and parsed.get("score") is not None:
    try:
      llm_score = float(parsed.get("score"))
    except Exception:
      llm_score = None

  if llm_score is not None and embedding_score is not None:
    # simple average between LLM numeric score and embedding similarity
    final_score = round((llm_score + embedding_score) / 2.0, 2)
  elif llm_score is not None:
    final_score = round(llm_score, 2)
  elif embedding_score is not None:
    final_score = embedding_score
  else:
    final_score = None

  # Attach embedding info into response
  out = {
    "parsed_llm": parsed,
    "embedding_score": embedding_score,
    "final_score": final_score
  }
  return out
