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

# response_mime_type="application/json" forces Gemini to ALWAYS return valid JSON.
# temperature=0 makes output deterministic — no hallucinated or echoed numbers.
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=genai.GenerationConfig(
        response_mime_type="application/json",
        temperature=0,
    ),
)

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
    amount   = Column(Float, nullable=False)


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

VALID_CATEGORIES = set(CATEGORY_KEYWORDS.keys()) | {"Other"}


# ── Pydantic schemas ──────────────────────────────────────────────────────────
class ExpenseInput(BaseModel):
    text: str


class ExpenseResponse(BaseModel):
    id:       int
    text:     str
    category: str
    amount:   float


# ── Helpers ───────────────────────────────────────────────────────────────────
def keyword_category(text: str) -> str:
    """Determine category purely from keywords in the input text."""
    text_lower = text.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(k in text_lower for k in keywords):
            return category
    return "Other"


def extract_amount_from_text(text: str) -> float:
    """
    Extract the most likely monetary amount from a free-text expense description.

    Priority:
      1. Number after a currency symbol  →  "₹250"  "Rs 49.99"
      2. First number >= 10 in the text  →  "Paid 250 for Uber ride"
      3. Any number found                →  "5 coffees"
      4. 0.0 if nothing found
    """
    # 1. Currency-prefixed number
    prefixed = re.search(r"(?:₹|rs\.?\s*)(\d[\d,]*(?:\.\d+)?)", text, re.IGNORECASE)
    if prefixed:
        return float(prefixed.group(1).replace(",", ""))

    # 2. All numbers — pick first that is >= 10 to skip counts like "1 ride"
    all_numbers = re.findall(r"\d[\d,]*(?:\.\d+)?", text)
    for n in all_numbers:
        val = float(n.replace(",", ""))
        if val >= 10:
            return val

    # 3. Fallback: whatever number exists
    if all_numbers:
        return float(all_numbers[0].replace(",", ""))

    return 0.0


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def read_root():
    return {"message": "Expense Tracker API running 🚀", "docs": "/docs"}


@app.post("/categorize")
def categorize(input: ExpenseInput):
    """Classify an expense using Gemini and store it in the database."""

    if not input.text.strip():
        raise HTTPException(status_code=400, detail="Expense text cannot be empty.")

    # ── Step 1: Extract amount directly from input text (always-reliable baseline)
    input_amount = extract_amount_from_text(input.text)

    # ── Step 2: Ask Gemini with JSON mode enforced ────────────────────────────
    # No example numbers in the prompt — avoids Gemini echoing them.
    prompt = f"""You are an expense classifier.

Given the expense text below, return a JSON object with exactly two keys:
  "category": one of Food, Transport, Shopping, Bills, Entertainment, Health, Education, Other
  "amount": the numeric rupee amount from the text as a number (not a string). Use 0 if no amount is mentioned.

Expense: {input.text}"""

    result = {"category": "Other", "amount": input_amount}

    try:
        response = model.generate_content(prompt)
        text_response = (response.text or "").strip()
        print(f"[Gemini RAW] '{input.text}' → '{text_response}'")

        parsed = json.loads(text_response)

        # Validate category
        gemini_category = str(parsed.get("category", "Other")).strip()
        result["category"] = gemini_category if gemini_category in VALID_CATEGORIES \
                             else keyword_category(input.text)

        # Sanitise amount (Gemini may return "250" as string or 250 as number)
        raw_amount = parsed.get("amount", 0)
        try:
            gemini_amount = float(str(raw_amount).replace(",", "").strip())
        except (ValueError, TypeError):
            gemini_amount = 0.0

        # If Gemini returned 0 but we know the amount from the text, use our value
        result["amount"] = gemini_amount if gemini_amount > 0 else input_amount
        print(f"[Result] category={result['category']}  amount={result['amount']}")

    except json.JSONDecodeError as e:
        print(f"[JSON parse failed] {e} — keyword fallback")
        result["category"] = keyword_category(input.text)
        result["amount"]   = input_amount

    except Exception as e:
        print(f"[Gemini error] {e} — keyword fallback")
        result["category"] = keyword_category(input.text)
        result["amount"]   = input_amount

    if result["amount"] <= 0:
        print(f"[Warning] amount=0 for: '{input.text}'")

    # ── Step 3: Persist ───────────────────────────────────────────────────────
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
    """Return the sum of all expense amounts."""
    db = SessionLocal()
    try:
        total = sum(e.amount for e in db.query(Expense).all())
    finally:
        db.close()
    return {"total": round(total, 2)}


@app.get("/logs", response_model=list[ExpenseResponse])
def get_logs():
    """Return all expense records, newest first."""
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
    """Return total spend per category, sorted highest first."""
    db = SessionLocal()
    try:
        expenses = db.query(Expense).all()
    finally:
        db.close()

    summary: dict[str, float] = {}
    for e in expenses:
        summary[e.category] = round(summary.get(e.category, 0.0) + e.amount, 2)

    return dict(sorted(summary.items(), key=lambda x: x[1], reverse=True))


@app.delete("/logs/{expense_id}")
def delete_expense(expense_id: int):
    """Delete a specific expense record by ID."""
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