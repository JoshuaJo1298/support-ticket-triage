# JJsauces Support Ticket Triage

A local customer support queue for JJsauces, a fictional Australian condiment shop. The model suggests billing, shipping, returns, account access or manual review. A reviewer assigns the final category. SQLite retains both decisions.

## Run the demo

Start Ollama locally and pull the default model with `ollama pull qwen2.5:0.5b`. Then run:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --port 8001
```

Open http://localhost:8001 and try these fictional enquiries:

| Ticket | Expected category |
| --- | --- |
| The tracking says my hot sauce gift pack was delivered, but it isn’t here. | Shipping |
| I was charged twice for the smoky barbecue sauce bundle. | Billing |
| My chilli sauce bottle arrived cracked. Can I send it back for a replacement? | Returns |
| The password reset link for my JJsauces account has expired. | Account access |
| Do you supply your hot sauces wholesale to cafés? | Manual review |

Check the suggestion, select the final category and click **Save decision**. The queue preserves both the suggestion and the reviewer’s decision. An unavailable model, timeout, or invalid response sends the ticket to manual review. Set `TICKET_DB` to use a separate demo file.

[Earlier prototype screenshot showing the manual-review fallback](queue-demo.png)

## Evaluation

```sh
python evaluate.py --baseline-only
python evaluate.py > results.json
pip install pytest
python -m pytest -q
```

The 60 synthetic evaluation tickets have 12 examples per label, including multi-issue and out-of-scope tickets. The prompt examples are separate. Reported macro F1 weights each of the five labels equally. The review rate measures how much work is deferred. Median latency measures each warmed routing call, and direct API fees are USD 0 for both local routers. Electricity and machine time are outside that estimate.

### Original benchmark on 60 held-out tickets

These recorded scores are from the original generic online-store prompt. The JJsauces prompt needs a fresh evaluation before it can claim new results.

| Router | Macro F1 | Routed to manual review | Median latency | Estimated API cost per ticket |
| --- | ---: | ---: | ---: | ---: |
| Keyword baseline | 0.525 | 40/60 (66.7%) | 0.010 ms | USD 0 |
| Qwen2.5 0.5B via Ollama | 0.570 | 12/60 (20.0%) | 424.7 ms | USD 0 |

The LLM recognized several returns phrased without keywords, such as “The headphones are too tight. Could I send them back?” The lower review rate came with a tradeoff. It correctly deferred just **2 of 12** tickets labelled for manual review. It sent a sponsorship request to billing, sent a delivery reroute to returns, and deferred an expired password reset. The keyword router correctly deferred 11 of the 12 review-labelled cases, while also deferring many tickets that had clear categories.

For the next iteration, I would test an explicit check for unrelated requests and strengthen the examples that distinguish delivery rerouting from product returns. I would measure review recall and per-category F1 on a fresh test set. This run used a small, balanced synthetic dataset on a GitHub-hosted CPU runner, so the scores and latency describe this benchmark.

[Per-ticket predictions and confusion matrices](data/benchmark.json) · [GitHub Actions benchmark run](https://github.com/JoshuaJo1298/support-ticket-triage/actions/runs/45905672464)

### Engineering tradeoff

SQLite and a local model keep the demo easy to run. Having a reviewer assign the final route adds a step but preserves a correction trail and gives the model a way to defer ambiguous cases.
