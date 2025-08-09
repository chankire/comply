# app.py — Streamlit Cloud ready
import os, json, re, io, sqlite3, difflib
from datetime import datetime
from typing import Dict, List, Optional

import streamlit as st
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF
from openai import OpenAI, RateLimitError

# Optional KG
try:
    from py2neo import Graph, Node, Relationship  # type: ignore
except Exception:
    Graph = None  # type: ignore

# ===================== Streamlit Cloud config =====================
st.set_page_config(page_title="Regulatory Compliance AI", layout="wide")
st.title("📜 Regulatory Compliance AI Assistant")
st.caption("Streamlit Cloud build • FAISS + SQLite • Optional Neo4j KG")

# ---- Secrets (set these in Streamlit Cloud: Settings → Secrets) ----
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
NEO4J_URI  = st.secrets.get("NEO4J_URI", "")
NEO4J_USER = st.secrets.get("NEO4J_USER", "")
NEO4J_PASS = st.secrets.get("NEO4J_PASS", "")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY missing in Secrets. Add it in Streamlit Cloud → Settings → Secrets.")
else:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY  # used by OpenAI SDK

# ===================== Tunables (safe for Cloud) =====================
MAX_CHARS_PER_CHUNK     = 1600     # small chunks → faster embed + safer prompts
MAX_SNIPPET_CHARS       = 900
MAX_TOTAL_CONTEXT_CHARS = 3600     # total context sent to LLM
TOP_K_RETRIEVE          = 4

# ===================== Data paths =====================
DATA_DIR   = "./data"
SQL_PATH   = os.path.join(DATA_DIR, "vectors.sqlite")
FAISS_PATH = os.path.join(DATA_DIR, "faiss.index")
os.makedirs(DATA_DIR, exist_ok=True)

# ===================== Sidebar Options (speed controls) =====================
with st.sidebar:
    st.header("⚙️ Ingest Options")
    build_kg        = st.checkbox("Build Knowledge Graph on ingest (slower)", value=False)
    kg_max_chunks   = st.number_input("Max chunks to send to LLM for KG", 0, 2000, 40, 10)
    max_chunks_doc  = st.number_input("Max chunks per document", 50, 5000, 400, 50)
    embed_batch_sz  = st.slider("Embedding batch size", 16, 512, 128, 16)
    st.caption("Tip: Keep KG off for demos. Use smaller PDFs for best performance.")

# ===================== OpenAI client =====================
client = OpenAI()

def chat_safe(messages, temperature=0, model="gpt-4o-mini") -> str:
    try:
        r = client.chat.completions.create(model=model, messages=messages, temperature=temperature)
        return r.choices[0].message.content
    except RateLimitError as e:
        return f"⚠️ OpenAI rate/token limit: {e}"
    except Exception as e:
        return f"⚠️ OpenAI error: {e}"

def llm_json(prompt: str):
    raw = chat_safe([{"role": "user", "content": prompt}], 0)
    try:
        return json.loads(raw)
    except Exception:
        try:
            return json.loads(raw.strip("```json").strip("```").strip())
        except Exception:
            return {"entities": [], "relationships": []}

# ===================== Optional Neo4j =====================
def norm_bolt(uri: str) -> str:
    if not uri: return ""
    if uri.startswith("neo4j+s://"): return uri.replace("neo4j+s://", "bolt+s://")
    if uri.startswith("neo4j://"):   return uri.replace("neo4j://", "bolt://")
    return uri

graph = None
if Graph and NEO4J_URI:
    try:
        graph = Graph(norm_bolt(NEO4J_URI), auth=(NEO4J_USER, NEO4J_PASS))
        _ = graph.run("RETURN 1").evaluate()
        st.success("Neo4j connected")
    except Exception as e:
        st.warning(f"Neo4j unavailable: {e}. KG features will be skipped.")

def load_into_neo4j(data: Dict[str, object]):
    if not graph: return
    try:
        nodes = {}
        for ent in data.get("entities", []):
            label = ent.get("type", "Entity")
            name  = ent.get("name")
            n = Node(label, name=name)
            graph.merge(n, label, "name")
            nodes[name] = n
        for rel in data.get("relationships", []):
            s, t, typ = rel.get("source"), rel.get("target"), rel.get("type", "RELATED")
            if s in nodes and t in nodes:
                graph.merge(Relationship(nodes[s], typ, nodes[t]))
    except Exception:
        pass

# ===================== Persistence: SQLite + FAISS =====================
conn = sqlite3.connect(SQL_PATH, check_same_thread=False)
conn.execute("""CREATE TABLE IF NOT EXISTS vectors(
  id INTEGER PRIMARY KEY,
  source TEXT,
  clause_id TEXT,
  text TEXT
)""")
conn.commit()

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
DIM = 384

try:
    index = faiss.read_index(FAISS_PATH)
except Exception:
    index = faiss.IndexFlatL2(DIM)

# Warm in-memory store & rebuild FAISS if mismatched
vector_store: List[Dict[str, object]] = []
rows = conn.execute("SELECT source, clause_id, text FROM vectors ORDER BY id").fetchall()
if rows:
    vector_store = [{"text": r[2], "metadata": {"source": r[0], "clause_id": r[1]}} for r in rows]
    if index.ntotal != len(rows):
        index = faiss.IndexFlatL2(DIM)
        for i in range(0, len(rows), 128):
            texts = [r[2] for r in rows[i:i+128]]
            embs = embed_model.encode(texts)
            index.add(np.array(embs, dtype="float32"))
        faiss.write_index(index, FAISS_PATH)

# ===================== Helpers =====================
def split_long_text(text: str, max_chars=MAX_CHARS_PER_CHUNK):
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    out, cur = [], ""
    for p in paras:
        if len(cur) + len(p) + 2 <= max_chars:
            cur = (cur + "\n\n" + p).strip()
        else:
            if cur: out.append(cur)
            for i in range(0, len(p), max_chars):
                out.append(p[i:i+max_chars])
            cur = ""
    if cur: out.append(cur)
    return out

def pdf_to_chunks(pdf_bytes: bytes, regulation_name: str):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    full = "\n".join(p.get_text("text") for p in doc)
    parts = re.split(r'(?=\bArticle\s+[0-9A-Za-z]+)', full)
    if len(parts) <= 1:
        parts = [full]
    chunks=[]
    for part in parts:
        t = part.strip()
        if len(t) < 120: continue
        for c in split_long_text(t, MAX_CHARS_PER_CHUNK):
            chunks.append((c, regulation_name))
    return chunks

def extract_entities_relationships(text: str, reg_name: str):
    p = f"""Extract KG JSON.
entities: name, type (Regulation, Part, Clause, Term, Metric, Date)
relationships: source, target, type (CONTAINS, DEFINES, REQUIRES, SUPERSEDES, REFERENCES, AFFECTS)
TEXT:
```{text[:1500]}```"""
    data = llm_json(p)
    ents = data.get("entities", [])
    if not any(e.get("name")==reg_name for e in ents):
        ents.append({"name": reg_name, "type": "Regulation"})
        data["entities"] = ents
    return data

def detect_clause_id(text: str) -> Optional[str]:
    m = re.search(r"(Article\s+[0-9A-Za-z]+)", text)
    return m.group(1) if m else None

def clamp(text: str, lim=MAX_SNIPPET_CHARS):
    t = text.strip()
    return t[:lim] + (" …" if len(t) > lim else "")

def build_context(snips: List[Dict[str, object]]) -> str:
    ctx = ""
    for r in snips:
        meta = r["metadata"]
        cid  = meta.get("clause_id") or "best match"
        piece = f"Source ({meta.get('source')}, {cid}):\n{clamp(r['text'])}\n\n"
        if len(ctx) + len(piece) > MAX_TOTAL_CONTEXT_CHARS:
            break
        ctx += piece
    return ctx

def generate_answer(q: str, snips: List[Dict[str, object]]) -> str:
    prompt = f"""You are a regulatory compliance assistant.

Question: {q}

Context:
{build_context(snips)}

Answer clearly and include citations like (Regulation, Clause ID or 'best match')."""
    return chat_safe([{"role":"user","content":prompt}], 0)

def semantic_search(q: str, top_k=TOP_K_RETRIEVE):
    if index.ntotal == 0: return []
    em = embed_model.encode([q])
    D, I = index.search(np.array(em, dtype="float32"), top_k)
    return [vector_store[i] for i in I[0] if i < len(vector_store)]

def semantic_search_filtered(q: str, source: str, top_k=TOP_K_RETRIEVE):
    if index.ntotal == 0: return []
    em = embed_model.encode([q])
    D, I = index.search(np.array(em, dtype="float32"), top_k*3)
    out=[]
    for i in I[0]:
        if i < len(vector_store) and vector_store[i]["metadata"].get("source")==source:
            out.append(vector_store[i])
        if len(out) >= top_k: break
    return out

def html_diff(a_title, a_text, b_title, b_text):
    return difflib.HtmlDiff(wrapcolumn=80).make_table(
        clamp(a_text, 2000).splitlines(),
        clamp(b_text, 2000).splitlines(),
        a_title, b_title
    )

# ===================== UI (Tabs) =====================
tabs = st.tabs(["Ingest","Summaries","Compare","Batch Report","Q&A","Settings"])
exec_cache: Dict[str, str] = st.session_state.setdefault("exec_cache", {})

# ---------- Ingest (FAST, batched) ----------
with tabs[0]:
    st.header("📥 Ingest Regulations — PDF")
    ups = st.file_uploader("Upload one or more PDFs", type="pdf", accept_multiple_files=True)
    reg_name = st.text_input("Regulation Name / Label")
    if st.button("Process & Ingest"):
        if not ups or not reg_name:
            st.warning("Please upload at least one PDF and set a name.")
        else:
            overall = st.progress(0.0, text="Starting…")
            total, done = len(ups), 0
            for up in ups:
                overall.progress(done/max(1,total), text=f"Reading {up.name}")
                chunks = pdf_to_chunks(up.read(), reg_name)
                if max_chunks_doc > 0:
                    chunks = chunks[:max_chunks_doc]
                texts = [t for t,_ in chunks]
                metas = [{"source":reg_name,"type":"Clause","clause_id":detect_clause_id(t)} for t in texts]

                # Optional KG on first N chunks
                if build_kg and kg_max_chunks > 0 and graph:
                    st.write(f"🔗 Building KG from first {min(kg_max_chunks,len(texts))} chunks …")
                    limit = min(kg_max_chunks, len(texts)); prog = st.progress(0.0)
                    for i in range(limit):
                        try:
                            data = extract_entities_relationships(texts[i], reg_name)
                            load_into_neo4j(data)
                        except Exception:
                            pass
                        if i % 5 == 0: prog.progress((i+1)/limit)
                    prog.progress(1.0)

                # Batch insert → SQLite
                conn.executemany(
                    "INSERT INTO vectors(source, clause_id, text) VALUES (?,?,?)",
                    [(m["source"], m["clause_id"], t) for t,m in zip(texts, metas)]
                )
                conn.commit()

                # Batch embeddings → FAISS
                st.write("⚡ Computing embeddings in batches …")
                prog = st.progress(0.0); added = 0
                for i in range(0, len(texts), embed_batch_sz):
                    batch = texts[i:i+embed_batch_sz]
                    embs  = embed_model.encode(batch)
                    index.add(np.array(embs, dtype="float32"))
                    added += len(batch)
                    prog.progress(min(1.0, added/len(texts)))
                try: faiss.write_index(index, FAISS_PATH)
                except Exception: pass

                # Warm in-memory store
                for t,m in zip(texts, metas):
                    vector_store.append({"text":t,"metadata":m})

                done += 1
                overall.progress(done/total, text=f"Ingested {up.name}")

            overall.progress(1.0, text="✅ Ingestion complete.")
            st.success(f"Ingested {done} file(s) as '{reg_name}'.")

# ---------- Summaries ----------
def exec_summary_from_docs(name: str, samples: List[str]) -> str:
    joined = "\n\n".join(s[:800] for s in samples[:8])
    p = f"""You are a senior regulatory analyst. Draft a crisp EXECUTIVE SUMMARY for "{name}" in under 160 words.
Base ONLY on these excerpts:
{joined}"""
    return chat_safe([{"role":"user","content":p}], 0.2)

with tabs[1]:
    st.header("📄 Executive Summary & Technical Overview")
    regs = sorted({v["metadata"].get("source") for v in vector_store if v["metadata"].get("source")})
    if not regs:
        st.info("Ingest at least one regulation first.")
    else:
        c1,c2 = st.columns(2)
        with c1:
            sel = st.selectbox("Regulation", regs)
            if st.button("Generate Executive Summary"):
                samples = [v["text"] for v in vector_store if v["metadata"].get("source")==sel][:30]
                exec_cache[sel] = exec_summary_from_docs(sel, samples)
            if sel in exec_cache:
                st.subheader(f"Executive Summary — {sel}")
                st.write(exec_cache[sel])
        with c2:
            def basic_stats(rn):
                if graph:
                    try:
                        n = graph.run("MATCH (n) RETURN count(n)").evaluate()
                        r = graph.run("MATCH ()-[r]-() RETURN count(r)").evaluate()
                    except Exception:
                        n = r = 0
                else:
                    n = r = 0
                idx = sum(1 for v in vector_store if v["metadata"].get("source")==rn)
                return n, r, idx
            sel2 = st.selectbox("Regulation (stats)", regs, key="stats")
            if st.button("Show Technical Overview"):
                n,r,idx = basic_stats(sel2)
                st.subheader(f"Technical Overview — {sel2}")
                st.write(f"Nodes: {n} | Relationships: {r} | Indexed Clauses: {idx}")

# ---------- Compare ----------
with tabs[2]:
    st.header("⚖️ Compare Two Regulations")
    regs = sorted({v["metadata"].get("source") for v in vector_store if v["metadata"].get("source")})
    if len(regs) < 2:
        st.info("Load at least two regulations to compare.")
    else:
        c1,c2 = st.columns(2)
        with c1: A = st.selectbox("Reg A", regs)
        with c2: B = st.selectbox("Reg B", regs, index=1 if len(regs)>1 else 0)
        topic = st.text_input("Topic / term (e.g., 'Liquidity Coverage Ratio')")
        if st.button("🔎 Compare") and topic:
            a = semantic_search_filtered(topic, A)[:1]
            b = semantic_search_filtered(topic, B)[:1]
            if not a or not b:
                st.warning("No matches. Try broader terms.")
            else:
                a, b = a[0], b[0]
                x1,x2 = st.columns(2)
                with x1: st.caption(f"{A} — {a['metadata'].get('clause_id','best match')}"); st.write(a["text"])
                with x2: st.caption(f"{B} — {b['metadata'].get('clause_id','best match')}"); st.write(b["text"])
                st.subheader("🧩 Diff")
                st.components.v1.html(html_diff(A,a["text"],B,b["text"]), height=420, scrolling=True)
                st.subheader("🧾 Auto Summary with Citations")
                st.write(generate_answer(f"Compare how these two regulations treat: {topic}", [a,b]))

# ---------- Batch Report ----------
with tabs[3]:
    st.header("🗂️ Batch Comparisons — Multi-topic Report")
    regs = sorted({v["metadata"].get("source") for v in vector_store if v["metadata"].get("source")})
    if len(regs) < 2:
        st.info("Need at least two regulations.")
    else:
        l,r = st.columns(2)
        with l: A = st.selectbox("Reg A (batch)", regs, key="bA")
        with r: B = st.selectbox("Reg B (batch)", regs, index=1 if len(regs)>1 else 0, key="bB")
        topics = st.text_area("Topics (one per line)", "Liquidity Coverage Ratio\nOperational Risk\nReporting Frequency")
        include_exec = st.checkbox("Include Executive Summaries", True)

        if st.button("📘 Generate Multi-topic Report") and topics.strip():
            T = [t.strip() for t in topics.splitlines() if t.strip()]
            res = []
            execA, execB = None, None
            if include_exec:
                if A in exec_cache: execA = exec_cache[A]
                else:
                    sa = [v["text"] for v in vector_store if v["metadata"].get("source")==A][:20]
                    execA = exec_summary_from_docs(A, sa); exec_cache[A]=execA
                if B in exec_cache: execB = exec_cache[B]
                else:
                    sb = [v["text"] for v in vector_store if v["metadata"].get("source")==B][:20]
                    execB = exec_summary_from_docs(B, sb); exec_cache[B]=execB
            for t in T:
                a = semantic_search_filtered(t, A)[:1]
                b = semantic_search_filtered(t, B)[:1]
                if a and b:
                    a,b=a[0],b[0]
                    res.append({"topic":t,"a":a,"b":b,
                                "summary": generate_answer(f"Compare how these two regulations treat: {t}", [a,b])})
            if not res:
                st.warning("No matches for your topics.")
            else:
                st.success(f"Built {len(res)} comparisons.")

                # DOCX + PDF downloads
                from docx import Document
                from docx.shared import Pt
                from reportlab.lib.pagesizes import A4
                from reportlab.pdfgen import canvas
                from textwrap import wrap

                def build_docx() -> bytes:
                    doc = Document()
                    doc.add_heading("Multi-topic Regulatory Comparison Report", 0)
                    doc.add_paragraph(f"Generated: {datetime.utcnow().isoformat(timespec='seconds')}Z")
                    doc.add_paragraph(f"Regulations: {A} vs {B}")
                    if include_exec:
                        doc.add_heading("Executive Summaries", 1)
                        doc.add_heading(A, 2); doc.add_paragraph(execA or "(none)")
                        doc.add_heading(B, 2); doc.add_paragraph(execB or "(none)")
                    doc.add_heading("Comparisons", 1)
                    for it in res:
                        doc.add_heading(it["topic"], 2)
                        doc.add_paragraph(it["summary"])
                        a_meta, b_meta = it["a"]["metadata"], it["b"]["metadata"]
                        doc.add_heading(f"{A} — {a_meta.get('clause_id','best match')}", 3)
                        p = doc.add_paragraph(it["a"]["text"]); p.style.font.size = Pt(10)
                        doc.add_heading(f"{B} — {b_meta.get('clause_id','best match')}", 3)
                        p = doc.add_paragraph(it["b"]["text"]); p.style.font.size = Pt(10)
                        doc.add_heading("Citations", 3)
                        doc.add_paragraph(f"- ({A}, {a_meta.get('clause_id','best match')})")
                        doc.add_paragraph(f"- ({B}, {b_meta.get('clause_id','best match')})")
                    bio = io.BytesIO(); doc.save(bio); bio.seek(0); return bio.getvalue()

                def build_pdf() -> bytes:
                    buf = io.BytesIO(); c = canvas.Canvas(buf, pagesize=A4)
                    W,H=A4; L,lh=40,14
                    def draw(title, body, y, bold=False):
                        c.setFont("Helvetica-Bold" if bold else "Helvetica", 12 if bold else 10)
                        for line in wrap(title if bold else body, 110):
                            if y<60: c.showPage(); y=H-60; c.setFont("Helvetica",10)
                            c.drawString(L,y,line); y-=lh
                        return y-(6 if bold else 2)
                    c.setFont("Helvetica-Bold",16); c.drawString(L,H-40,"Multi-topic Regulatory Comparison Report")
                    c.setFont("Helvetica",9); c.drawString(L,H-58,f"Generated: {datetime.utcnow().isoformat(timespec='seconds')}Z")
                    c.drawString(L,H-72,f"Regulations: {A} vs {B}"); y=H-94
                    if include_exec:
                        y=draw("Executive Summaries","",y,True); y=draw(A,execA or "(none)",y); y=draw(B,execB or "(none)",y)
                    y=draw("Comparisons","",y,True)
                    for it in res:
                        y=draw(it["topic"],"",y,True)
                        y=draw("Summary:", it["summary"], y)
                        a_meta, b_meta = it["a"]["metadata"], it["b"]["metadata"]
                        y=draw(f"{A} — {a_meta.get('clause_id','best match')}", it["a"]["text"], y)
                        y=draw(f"{B} — {b_meta.get('clause_id','best match')}", it["b"]["text"], y)
                        y=draw("Citations:", f"({A}, {a_meta.get('clause_id','best match')}) | ({B}, {b_meta.get('clause_id','best match')})", y)
                    c.showPage(); c.save(); buf.seek(0); return buf.getvalue()

                c1,c2 = st.columns(2)
                with c1: st.download_button("⬇️ Download DOCX", build_docx(), file_name=f"batch_{A}_vs_{B}.docx",
                                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
                with c2: st.download_button("⬇️ Download PDF", build_pdf(), file_name=f"batch_{A}_vs_{B}.pdf",
                                            mime="application/pdf")

# ---------- Q&A ----------
with tabs[4]:
    st.header("💬 Q&A — Ask a Question")
    q = st.text_input("Enter your compliance question")
    if st.button("Search & Answer") and q:
        hits = semantic_search(q)
        st.subheader("Answer"); st.write(generate_answer(q, hits))
        st.subheader("Sources")
        if hits:
            for r in hits:
                with st.expander(f"{r['metadata'].get('source')} — {r['metadata'].get('clause_id','best match')}"):
                    st.write(r["text"])
        else:
            st.write("No matching clauses found.")

# ---------- Settings ----------
with tabs[5]:
    st.header("⚙️ Settings")
    regs = {v["metadata"].get("source") for v in vector_store if v["metadata"].get("source")}
    st.write(f"**Indexed chunks:** {index.ntotal} | **Regulations:** {len(regs)} | **KG:** {'On' if graph else 'Off'}")
    if st.button("Clear persisted data"):
        try:
            conn.close()
            if os.path.exists(SQL_PATH): os.remove(SQL_PATH)
            if os.path.exists(FAISS_PATH): os.remove(FAISS_PATH)
            st.success("Cleared. Please restart the app.")
        except Exception as e:
            st.error(f"Failed: {e}")
