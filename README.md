# Elofic Catalog Assistant

Simple Streamlit app that loads an Excel catalog, indexes it into a ChromaDB persistent store, and answers user queries using Google Gemini (via `google-genai`).

## Quickstart

1. Create and activate a Python environment (recommended):

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# or Git Bash / cmd
source .venv/Scripts/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Provide credentials:

- Copy `.env.example` to `.env` and set `GEMINI_API_KEY`.
- Or export the variable directly in your environment.

Example `.env` (do NOT commit this file):

```
GEMINI_API_KEY=your_new_gemini_api_key_here
```

4. Start the app:

```bash
streamlit run main.py
```

## Files

- `main.py`: Streamlit app and RAG pipeline
- `requirements.txt`: Python dependencies
- `.env.example`: example environment file (not committed secrets)
- `elofic_vectordb/`: local ChromaDB persistence folder

## Security: leaked API key (important)

If a secret (like `GEMINI_API_KEY`) was accidentally committed, do these steps immediately:

1. Rotate / revoke the compromised key in Google Cloud Console (create a new key).
2. Remove the secret from the repository history (recommended: `git-filter-repo`):

```bash
pip install git-filter-repo
# from repo root
git rm --cached .env || true
git commit -m "Remove .env from index"
git filter-repo --path .env --invert-paths
git reflog expire --expire=now --all
git gc --prune=now --aggressive
```

Alternative: use BFG repo-cleaner (Java) if you prefer.

3. Force-push the cleaned history to the remote (coordinate with collaborators):

```bash
git push origin --force --all
git push origin --force --tags
```

4. Verify the secret is gone from the history before sharing the repo.

If you want help performing the cleanup here, tell me which method you prefer (`git-filter-repo` or `BFG`) and confirm that you have rotated/revoked the compromised key.

## Notes

- Do NOT commit `.env` or secrets. Add additional sensitive files to `.gitignore` as needed.
- The app expects the Excel file `Elofic AI Agent Data.xlsx` in the repo root. If it's large or private, keep it out of version control and store it externally.

---
Created by the repo maintainer to document setup and safe-push guidance.
