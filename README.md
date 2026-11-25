# Email Spam Tester (educational demo)

Scores an email's subject and body with a trained scikit-learn model plus transparent content checks.
**Educational demonstration only: not a production email security or deliverability system.**
Original design and code; it is inspired by the general idea of email spam testers, not a copy of any product.

## Run it (from `backend/`)
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m scripts.generate_sample_data   # synthetic demo data
python -m scripts.train_model            # trains + saves models/spam_pipeline.joblib (re-run to retrain)
python -m scripts.manage_keys create "Alice"   # prints an API key ONCE
uvicorn app.main:app --reload
```
Open http://localhost:8000 (app) and http://localhost:8000/docs (Swagger; use **Authorize**-less header `X-API-Key`).
Paste the key into the app's "API key" box.

## Users and API keys (max 50)
- `python -m scripts.manage_keys create "Name"` / `list` / `revoke <id>` / `bulk 50` (creates user-01..user-50, prints keys once).
- Only SHA-256 hashes of keys are stored. Each user sees only their own history. Limit: 30 requests/min per key (`.env.example`).
- Never commit `.env` or keys. Change CORS/limits and add HTTPS before any real deployment.

## API (header `X-API-Key` required except /health and /api/model-info)
`POST /api/predict` `GET /api/history` `DELETE /api/history[/{id}]` `GET /api/stats` `GET /api/model-info` `GET /health`

Example: `curl -X POST localhost:8000/api/predict -H "X-API-Key: spk_..." -H "Content-Type: application/json" -d '{"subject":"Hi","body":"Meeting at 3"}'`

## How the score works
Risk (0-100) = 70% model spam probability + 30% rule score (caps, "!" count, spam phrases, urgency, links, shorteners, amounts).
The model is TF-IDF + Logistic Regression, 80/20 stratified split, seed 42; metrics are measured on the held-out set.
The bundled dataset is **synthetic**, so its metrics do not reflect real-world accuracy. For real use, train on a public corpus
(e.g. SpamAssassin, Enron-Spam, UCI SMS Spam) after checking its license:
`python -m scripts.train_model --data data/real.csv --dataset-name "..." --dataset-source "URL + license"` (columns: subject, body, label).

## Privacy
Saved email text is stored in SQLite. Untick "Save this message", set `STORE_EMAIL_TEXT=false`, or delete history in the app.

## Tests
`pytest` (from `backend/`). If the model fails to load after a scikit-learn upgrade, retrain it.
