# Support Ticket Triage

A local review queue for synthetic online-store tickets. The model suggests billing, shipping, returns, account access or manual review. A reviewer assigns the final category. SQLite retains both decisions.

## Run the demo

Start Ollama locally and pull the default model with `ollama pull qwen2.5:0.5b`. Then run:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --port 8001
```

Open http://localhost:8001 and submit a synthetic ticket such as “The tracking link says delivered but my porch is empty.” Choose the final category and click **Save decision**. To demonstrate an override, pick a different category and see both decisions retained. An unavailable model, timeout, or invalid response routes the ticket to manual review. The default queue uses a local SQLite file. Set `TICKET_DB` to use a separate demo file.

## Evaluation

```sh
python evaluate.py --baseline-only
python evaluate.py > results.json
pip install pytest
python -m pytest -q
```

The 60 synthetic evaluation tickets have 12 examples per label, including multi-issue and out-of-scope tickets. The prompt examples are separate. Reported macro F1 weights each of the five labels equally. The review rate measures how much work is deferred. Median latency measures each warmed routing call, and direct API fees are USD 0 for both local routers. Electricity and machine time are outside that estimate.

SQLite and a local model keep the demo easy to run. Having a reviewer assign the final route adds a step but preserves a correction trail and gives the model a way to defer ambiguous cases.
