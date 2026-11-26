import os
import sqlite3
from contextlib import asynccontextmanager, closing
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from routing import LABELS, llm_router

ROOT = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(ROOT / "templates"))


def connect():
    db = sqlite3.connect(os.getenv("TICKET_DB", str(ROOT / "tickets.sqlite3")))
    db.row_factory = sqlite3.Row
    return db


@asynccontextmanager
async def lifespan(app):
    with closing(connect()) as db, db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS tickets (
                id INTEGER PRIMARY KEY, body TEXT NOT NULL,
                suggestion TEXT NOT NULL, model_error TEXT NOT NULL,
                final_category TEXT
            )
        """)
    yield


app = FastAPI(title="Support Ticket Triage", lifespan=lifespan)


@app.get("/")
def queue(request: Request):
    with closing(connect()) as db:
        rows = db.execute("""
            SELECT * FROM tickets ORDER BY final_category IS NOT NULL, id DESC LIMIT 100
        """).fetchall()
    return templates.TemplateResponse(request=request, name="queue.html",
                                      context={"tickets": rows, "labels": LABELS})


@app.post("/tickets")
def submit(body: str = Form(max_length=2000)):
    body = body.strip()
    if not body:
        raise HTTPException(422, "Enter a ticket description")
    prediction = llm_router(body)
    with closing(connect()) as db, db:
        db.execute("INSERT INTO tickets (body, suggestion, model_error) VALUES (?, ?, ?)",
                   (body, prediction.category, prediction.error))
    return RedirectResponse("/", status_code=303)


@app.post("/tickets/{ticket_id}/review")
def review(ticket_id: int, category: str = Form()):
    if category not in LABELS:
        raise HTTPException(422, "Unknown category")
    with closing(connect()) as db, db:
        updated = db.execute("UPDATE tickets SET final_category = ? WHERE id = ? AND final_category IS NULL",
                             (category, ticket_id))
        if not updated.rowcount:
            exists = db.execute("SELECT 1 FROM tickets WHERE id = ?", (ticket_id,)).fetchone()
            raise HTTPException(409 if exists else 404, "Ticket already reviewed" if exists else "Ticket not found")
    return RedirectResponse("/", status_code=303)
