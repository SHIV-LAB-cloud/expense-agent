from fastapi import FastAPI
from pydantic import BaseModel
import google.generativeai as genai
import os
import json
import re
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base

# Load environment variables
load_dotenv()

# ✅ Configure Gemini using ENV variable
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ✅ Use stable model
model = genai.GenerativeModel("gemini-1.5-flash")

# FastAPI app
app = FastAPI(title="Expense Tracker Agent")

# Database setup
engine = create_engine("sqlite:///expenses.db")
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# Database table
class Expense(Base):
    __tablename__ = "expenses"
    id = Column(Integer, primary_key=True, index=True)
    text = Column(String)
    category = Column(String)
    amount = Column(Integer)

Base.metadata.create_all(bind=engine)

# Input schema
class ExpenseInput(BaseModel):
    text: str

# Root endpoint
@app.get("/")
def read_root():
    return {
        "message": "Expense Tracker Agent Running 🚀",
        "docs": "/docs"
    }

# Main AI endpoint
@app.post("/categorize")
def categorize_expense(input: ExpenseInput):

    prompt = f"""
Classify this expense.

Categories:
Food, Transport, Shopping, Bills, Entertainment, Health, Education, Other

Rules:
Uber/Ola/Taxi/Auto → Transport
Pizza/Food/Restaurant → Food
Amazon/Flipkart → Shopping
Electricity/Bill → Bills
Netflix/Movie → Entertainment
Doctor/Hospital → Health
Course/Fees → Education

Return ONLY JSON:
{{"category":"Transport","amount":250}}

Expense: {input.text}
"""

    try:
        response = model.generate_content(prompt)

        # 🔍 Debug (optional)
        print("RAW RESPONSE:", response.text)

        text_response = response.text.strip()

        # ✅ Safe JSON parsing
        if text_response.startswith("{"):
            result = json.loads(text_response)
        else:
            match = re.search(r'\{.*\}', text_response, re.DOTALL)
            if match:
                result = json.loads(match.group())
            else:
                result = {"category": "Other", "amount": 0}

    except Exception as e:
        print("Error:", e)
        result = {"category": "Other", "amount": 0}

    # 🔥 Keyword fallback (IMPORTANT)
    text = input.text.lower()

    if result["category"] == "Other":
        if any(word in text for word in ["uber", "ola", "taxi", "auto", "metro"]):
            result["category"] = "Transport"
        elif any(word in text for word in ["pizza", "food", "restaurant", "swiggy", "zomato"]):
            result["category"] = "Food"
        elif any(word in text for word in ["amazon", "flipkart", "clothes", "shoes"]):
            result["category"] = "Shopping"
        elif any(word in text for word in ["electricity", "bill", "recharge"]):
            result["category"] = "Bills"
        elif any(word in text for word in ["netflix", "movie"]):
            result["category"] = "Entertainment"
        elif any(word in text for word in ["doctor", "hospital"]):
            result["category"] = "Health"
        elif any(word in text for word in ["course", "fees"]):
            result["category"] = "Education"

    # Save to DB
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


# Total expense
@app.get("/total")
def get_total():
    db = SessionLocal()
    total = sum(e.amount for e in db.query(Expense).all())
    db.close()
    return {"total_expense": total}


# Logs
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


# Category summary
@app.get("/category-summary")
def category_summary():
    db = SessionLocal()
    expenses = db.query(Expense).all()
    db.close()

    summary = {}
    for e in expenses:
        summary[e.category] = summary.get(e.category, 0) + e.amount

    return summary