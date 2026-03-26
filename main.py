from fastapi import FastAPI
from pydantic import BaseModel
import google.generativeai as genai
import os
import json
import re
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base

# Load env
load_dotenv()

# ✅ Use ENV variable (IMPORTANT)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-2.5-flash")

app = FastAPI(title="Expense Tracker Agent")

# Database setup
engine = create_engine("sqlite:///expenses.db")
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class Expense(Base):
    __tablename__ = "expenses"
    id = Column(Integer, primary_key=True, index=True)
    text = Column(String)
    category = Column(String)
    amount = Column(Integer)

Base.metadata.create_all(bind=engine)

class ExpenseInput(BaseModel):
    text: str

# ✅ Keep only ONE root endpoint
@app.get("/")
def read_root():
    return {
        "message": "Expense Tracker Agent Running 🚀",
        "docs": "/docs"
    }

@app.post("/categorize")
def categorize_expense(input: ExpenseInput):

    prompt = f"""
You are a strict expense classification system.

Your job:
1. Extract amount (number only)
2. Classify into EXACTLY one category:

Categories:
Food, Transport, Shopping, Bills, Entertainment, Health, Education, Other

Rules (VERY IMPORTANT):
- Uber, Ola, auto, taxi, metro → Transport
- Pizza, restaurant, food, Swiggy, Zomato → Food
- Amazon, Flipkart, clothes, shoes → Shopping
- Electricity, bill, recharge → Bills
- Netflix, movie → Entertainment
- Doctor, hospital → Health
- Course, fees → Education

Return ONLY JSON (no explanation):
{{"category": "Transport", "amount": 250}}

If unsure → use "Other"

Expense: "{input.text}"
"""

    try:
        response = model.generate_content(prompt)
        cleaned = re.search(r'\{.*\}', response.text, re.DOTALL).group()
        result = json.loads(cleaned)
    except Exception as e:
        print(f"Error: {e}")
        result = {"category": "Other", "amount": 0}

    db = SessionLocal()
    expense = Expense(
        text=input.text,
        category=result["category"],
        amount=result["amount"]
    )
    db.add(expense)
    db.commit()
    db.close()

    return result

@app.get("/total")
def get_total():
    db = SessionLocal()
    total = sum(e.amount for e in db.query(Expense).all())
    db.close()
    return {"total_expense": total}

@app.get("/logs")
def get_logs():
    db = SessionLocal()
    expenses = db.query(Expense).all()
    db.close()

    return {
        "expenses": [
            {"text": e.text, "category": e.category, "amount": e.amount}
            for e in expenses
        ]
    }

@app.get("/category-summary")
def category_summary():
    db = SessionLocal()
    expenses = db.query(Expense).all()
    db.close()

    summary = {}
    for e in expenses:
        summary[e.category] = summary.get(e.category, 0) + e.amount

    return summary


