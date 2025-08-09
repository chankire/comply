# Comply – Regulatory Compliance AI (Streamlit Cloud)

Upload regulation PDFs, generate executive summaries, compare rules across documents, create batch reports, and ask questions with citations.  
Built with Streamlit, FAISS (vector search), and optional Neo4j (knowledge graph).

## Live Deployment (Streamlit Cloud)
1. In Streamlit Cloud, create a new app and point it to this repo, branch `main`, **main file = `app.py`**.
2. Go to **Settings → Secrets** and add:
   ```toml
   OPENAI_API_KEY = "sk-...your key..."
   # Optional if using Neo4j features
   NEO4J_URI  = "bolt+s://<host>:7687"
   NEO4J_USER = "neo4j"
   NEO4J_PASS = "your-password"
