from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from typing import Dict, Any, List, Optional
from openai import OpenAI
from dotenv import load_dotenv
import re, json, math
import fitz  # PyMuPDF
import docx
from io import BytesIO

load_dotenv()

# Initialize OpenAI client (expects OPENAI_API_KEY in env)
client = OpenAI()

app = FastAPI(title="theAgent (embeddings-only skill matching)", version="0.6")

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

  # PDF detection
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

  # DOCX detection (zip header PK)
  if raw_bytes[:2] == b"PK":
    try:
      doc = docx.Document(BytesIO(raw_bytes))
      paragraphs = [p.text for p in doc.paragraphs if p.text]
      return normalize_text("\n".join(paragraphs))
    except Exception as e:
      raise HTTPException(status_code=415, detail=f"DOCX parse error for {filename}: {e}")

  # Fallback to text
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
    return resp.data[0].embedding
  except Exception:
    return None

def batch_embed_texts(texts: List[str], model: str = "text-embedding-3-small") -> List[Optional[List[float]]]:
  if not texts:
    return []
  try:
    resp = client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in resp.data]
  except Exception:
    # fallback to per-item calls
    return [get_embedding(t, model=model) for t in texts]

def cosine_similarity(a: List[float], b: List[float]) -> float:
  if not a or not b or len(a) != len(b):
    return 0.0
  dot = sum(x * y for x, y in zip(a, b))
  na = math.sqrt(sum(x * x for x in a))
  nb = math.sqrt(sum(x * x for x in b))
  if na == 0 or nb == 0:
    return 0.0
  return dot / (na * nb)

# -----------------------
# Phrase extraction & matching
# -----------------------
def chunk_text_into_sentences(text: str, max_words: int = 40):
  sents = re.split(r'(?<=[.!?;\n])\s+', text.strip())
  out = []
  for s in sents:
    s = s.strip()
    if not s:
      continue
    words = s.split()
    if len(words) <= max_words:
      out.append(s)
    else:
      for i in range(0, len(words), max_words):
        out.append(" ".join(words[i:i+max_words]).strip())
  return [normalize_text(s) for s in out if s]

def extract_jd_skill_phrases(jd_text: str, max_words_per_phrase: int = 30):
  """
  Extract reasonable JD phrases. If a line is very long, split it into smaller
  sentence/comma-separated chunks so each JD phrase is short enough for phrase-level matching.
  """
  parts = []
  for line in jd_text.splitlines():
    line = line.strip()
    if not line:
      continue
    if len(line.split()) > max_words_per_phrase:
      subparts = re.split(r'[.;\n,]', line)
    else:
      subparts = [line]
    for p in subparts:
      p = p.strip()
      if not p:
        continue
      wcount = len(p.split())
      if 1 <= wcount <= max_words_per_phrase:
        parts.append(normalize_text(p))
  if not parts:
    return [normalize_text(jd_text)]
  seen, out = set(), []
  for p in parts:
    if p not in seen:
      seen.add(p)
      out.append(p)
  return out

def lexical_match(jd_phrase: str, resume_chunk: str) -> bool:
  if not jd_phrase or not resume_chunk:
    return False
  jd_tokens = [t for t in re.split(r'\s+', jd_phrase) if t]
  res_tokens = [t for t in re.split(r'\s+', resume_chunk) if t]
  if 1 <= len(jd_tokens) <= 3:
    if all(token in res_tokens for token in jd_tokens):
      return True
    if len(jd_tokens) == 1 and len(res_tokens) >= 2:
      token = jd_tokens[0]
      initials = "".join([w[0] for w in res_tokens if w])
      if token.replace(".", "") == initials:
        return True
  return False

def match_jd_to_resume_phrases(jd_text: str, resume_text: str, threshold: float = 0.70):
  jd_phrases = extract_jd_skill_phrases(jd_text)
  resume_chunks = chunk_text_into_sentences(resume_text, max_words=40)

  to_embed = jd_phrases + resume_chunks
  embeddings = batch_embed_texts(to_embed)
  jd_embs, resume_embs = embeddings[:len(jd_phrases)], embeddings[len(jd_phrases):]

  matches = []
  for i, phrase in enumerate(jd_phrases):
    best_sim, best_chunk = 0.0, None
    if not jd_embs[i]:
      matches.append({"jd_phrase": phrase, "matched": False, "best_sim": 0.0, "best_chunk": None})
      continue
    for j, r_emb in enumerate(resume_embs):
      if not r_emb:
        continue
      sim = cosine_similarity(jd_embs[i], r_emb)
      if sim > best_sim:
        best_sim, best_chunk = sim, resume_chunks[j]
    best_sim_percent = round(best_sim * 100, 2)
    matched = best_sim >= threshold
    if not matched and lexical_match(phrase, best_chunk or ""):
      matched = True
    matches.append({
      "jd_phrase": phrase,
      "matched": matched,
      "best_sim": best_sim_percent,
      "best_chunk": best_chunk
    })
  return {
    "matches": matches,
    "total_jd_phrases": len(jd_phrases),
    "resume_chunks_count": len(resume_chunks)
  }

# -----------------------
# Endpoint (returns requested shape)
# -----------------------
@app.post("/score_resume/")
async def score_resume(file: UploadFile = File(...), job_description: str = Form(...), match_threshold: float = Form(0.70)):
  raw = await file.read()
  resume_text = extract_text_from_bytes(raw, file.filename)
  jd_text = normalize_text(job_description)

  # overall similarity (JD vs full resume)
  jd_emb, resume_emb = get_embedding(jd_text[:20000]), get_embedding(resume_text[:20000])
  embedding_score = round(cosine_similarity(jd_emb, resume_emb) * 100, 2) if jd_emb and resume_emb else None

  # optional overall LLM scoring (kept but not used for phrase canonicalization)
  parsed = {"raw_output": None}
  try:
    prompt = f"""
    You are an AI hiring assistant.
    Compare this resume against the job description.
    extract key skills from resume.
    Respond ONLY in JSON with the following structure:

    {{
      "score": <number 0-100>,
      "technical_skills": ["skill1", "skill2"],
      "strengths": ["point1", "point2"],
      "weaknesses": ["point1", "point2"],
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
      max_tokens=400
    )
    parsed = safe_parse_llm_json(resp.choices[0].message.content)
  except Exception:
    parsed = {"raw_output": None}

  # per-phrase matching
  skill_match_result = match_jd_to_resume_phrases(jd_text, resume_text, threshold=float(match_threshold))
  matches = skill_match_result["matches"]
  matched_count = sum(1 for m in matches if m["matched"])
  total_jd = max(1, skill_match_result["total_jd_phrases"])
  keyword_score = round((matched_count / total_jd) * 100, 2)

  # derive llm_score if available
  llm_score = None
  if isinstance(parsed, dict) and parsed.get("score") is not None:
    try:
      llm_score = float(parsed.get("score"))
    except Exception:
      llm_score = None

  # final score: deterministic embedding+keyword average with optional small LLM influence
  if embedding_score is not None:
    if llm_score is not None:
      final_score = round(0.55 * embedding_score + 0.35 * keyword_score + 0.10 * llm_score, 2)
    else:
      final_score = round(0.6 * embedding_score + 0.4 * keyword_score, 2)
  else:
    # fallback: prefer keyword_score if embeddings not available
    final_score = round(0.5 * keyword_score + 0.5 * llm_score, 2) if llm_score is not None else keyword_score

  # Build response shape requested by user
  # candidate_skills: unique matched resume chunks (human readable)
  candidate_skills = []
  for m in matches:
    if m.get("matched") and m.get("best_chunk"):
      chunk = (m["best_chunk"] or "").strip()
      if chunk and chunk not in candidate_skills:
        candidate_skills.append(chunk)

  # strengths: prefer LLM strengths if present, otherwise use top matched chunks
  strengths: List[str] = []
  if isinstance(parsed, dict):
    s = parsed.get("strengths")
    if isinstance(s, list) and s:
      strengths = s
  if not strengths:
    strengths = candidate_skills[:5]

  # weaknesses: prefer LLM weaknesses if present, otherwise list unmatched JD phrases
  weaknesses: List[str] = []
  if isinstance(parsed, dict):
    w = parsed.get("weaknesses")
    if isinstance(w, list) and w:
      weaknesses = w
  if not weaknesses:
    unmatched = [m["jd_phrase"] for m in matches if not m.get("matched")]
    weaknesses = unmatched[:5]

  response = {
    "fit(%)": final_score, # 0-100
    "strengths": strengths,
    "weaknesses": weaknesses
  }

  return response
