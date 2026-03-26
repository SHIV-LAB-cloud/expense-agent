from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
import os
import json
import re
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.orm import sessionmaker, declarative_base

# ── Configure Gemini ──────────────────────────────────────────────────────────
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

app = FastAPI(title="Expense Tracker Agent", version="0.1.0")

# ── Database ──────────────────────────────────────────────────────────────────
engine = create_engine("sqlite:///expenses.db", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class Expense(Base):
    __tablename__ = "expenses"
    id       = Column(Integer, primary_key=True, index=True)
    text     = Column(String, nullable=False)
    category = Column(String, nullable=False)
    amount   = Column(Float, nullable=False)   # FIX: was Integer (truncated decimals)


Base.metadata.create_all(bind=engine)

# ── Keyword fallback map (all 8 categories) ───────────────────────────────────
CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "Transport":     ["uber", "ola", "rapido", "metro", "bus", "auto", "cab",
                      "petrol", "fuel", "train", "flight", "ticket", "toll"],
    "Food":          ["zomato", "swiggy", "restaurant", "lunch", "dinner",
                      "breakfast", "coffee", "chai", "cafe", "hotel", "eat",
                      "food", "bhojan", "snack", "juice", "pizza", "biryani"],
    "Shopping":      ["amazon", "flipkart", "myntra", "meesho", "ajio",
                      "cloth", "shirt", "shoe", "bag", "watch", "buy",
                      "order", "purchase", "mall", "market"],
    "Bills":         ["electricity", "rent", "wifi", "broadband", "internet",
                      "recharge", "mobile", "phone", "water bill", "gas bill",
                      "emi", "insurance", "subscription"],
    "Entertainment": ["netflix", "prime", "hotstar", "movie", "cinema",
                      "concert", "spotify", "game", "play", "event",
                      "ticket", "show", "ott"],
    "Health":        ["pharmacy", "medicine", "doctor", "hospital", "clinic",
                      "gym", "fitness", "yoga", "lab", "test", "health",
                      "medical", "tablet", "injection"],
    "Education":     ["course", "book", "udemy", "coursera", "tuition",
                      "school", "college", "fee", "exam", "coaching",
                      "class", "study", "notes"],
}


# ── Pydantic schemas ──────────────────────────────────────────────────────────
class ExpenseInput(BaseModel):
    text: str


class ExpenseResponse(BaseModel):
    id:       int
    text:     str
    category: str
    amount:   float


# ── Helper: keyword-based category detection ──────────────────────────────────
def keyword_category(text: str) -> str:
    text_lower = text.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(k in text_lower for k in keywords):
            return category
    return "Other"


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def read_root():
    return {"message": "Expense Tracker API running 🚀", "docs": "/docs"}


@app.post("/categorize")
def categorize(input: ExpenseInput):
    """Classify an expense using Gemini and store it in the database."""

    if not input.text.strip():
        raise HTTPException(status_code=400, detail="Expense text cannot be empty.")

    prompt = f"""
Classify this expense into exactly one of these categories:
Food, Transport, Shopping, Bills, Entertainment, Health, Education, Other

Return ONLY a valid JSON object with no extra text, no markdown, no backticks:
{{"category": "Transport", "amount": 250}}

Rules:
- "amount" must be a number (integer or float).
- If amount is not clear from the text, return 0.
- Use "Other" only if no category fits.

Expense: {input.text}
"""

    result = {"category": "Other", "amount": 0.0}

    try:
        response = model.generate_content(prompt)
        text_response = (response.text or "").strip()
        print("RAW Gemini response:", text_response)

        # Strip markdown fences if present
        text_response = re.sub(r"```(?:json)?|```", "", text_response).strip()

        try:
            parsed = json.loads(text_response)
            result["category"] = str(parsed.get("category", "Other")).strip()
            result["amount"]   = float(parsed.get("amount", 0))

        except json.JSONDecodeError:
            # FIX: fallback ONLY runs when Gemini JSON fails, not every call
            print("JSON parse failed — using keyword fallback")
            result["category"] = keyword_category(input.text)
            amount_match = re.findall(r"\d+\.?\d*", text_response)
            result["amount"] = float(amount_match[0]) if amount_match else 0.0

    except Exception as e:
        print("Gemini error:", e)
        # Still attempt keyword fallback so we don't return empty data
        result["category"] = keyword_category(input.text)

    # Warn (but still save) when amount could not be determined
    if result["amount"] <= 0:
        print(f"Warning: amount is {result['amount']} for input: '{input.text}'")

    # FIX: session always closed via try/finally
    db = SessionLocal()
    try:
        expense = Expense(
            text=input.text,
            category=result["category"],
            amount=result["amount"],
        )
        db.add(expense)
        db.commit()
        db.refresh(expense)
        result["id"] = expense.id
    finally:
        db.close()

    return result


@app.get("/total")
def get_total():
    """Return the total amount spent across all expenses."""
    db = SessionLocal()
    try:
        total = sum(e.amount for e in db.query(Expense).all())
    finally:
        db.close()
    return {"total": round(total, 2)}


@app.get("/logs", response_model=list[ExpenseResponse])
def get_logs():
    """Return all stored expense records."""
    db = SessionLocal()
    try:
        expenses = db.query(Expense).order_by(Expense.id.desc()).all()
        return [
            ExpenseResponse(id=e.id, text=e.text, category=e.category, amount=e.amount)
            for e in expenses
        ]
    finally:
        db.close()


@app.get("/category-summary")
def category_summary():
    """Return total amount spent per category."""
    db = SessionLocal()
    try:
        expenses = db.query(Expense).all()
    finally:
        db.close()

    summary: dict[str, float] = {}
    for e in expenses:
        summary[e.category] = round(summary.get(e.category, 0.0) + e.amount, 2)

    # Sort by amount descending for readability
    return dict(sorted(summary.items(), key=lambda x: x[1], reverse=True))


@app.delete("/logs/{expense_id}")
def delete_expense(expense_id: int):
    """Delete a specific expense by ID."""
    db = SessionLocal()
    try:
        expense = db.query(Expense).filter(Expense.id == expense_id).first()
        if not expense:
            raise HTTPException(status_code=404, detail=f"Expense {expense_id} not found.")
        db.delete(expense)
        db.commit()
    finally:
        db.close()
    return {"message": f"Expense {expense_id} deleted successfully."}