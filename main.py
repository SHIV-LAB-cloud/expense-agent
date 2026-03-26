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

# Gemini setup
genai.configure(api_key="AIzaSyCwApDydpnGn5KTbXPzaKmzKU7UvePTEUw")
model = genai.GenerativeModel("gemini-2.5-flash")

# FastAPI app
app = FastAPI(title="Expense Tracker Agent")

# Database setup
engine = create_engine("sqlite:///expenses.db")
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# Table
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

# Output schema
class ExpenseOutput(BaseModel):
    category: str
    amount: int

# 🏠 Root endpoint to test if it's running
@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Expense Tracker Agent API!",
        "documentation": "Go to http://127.0.0.1:8000/docs to test the endpoints."
    }

# 🏠 Root endpoint to test if it's running
@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Expense Tracker Agent API!",
        "documentation": "Go to http://127.0.0.1:8000/docs to test the endpoints."
    }

# 🧠 AI Categorization + Amount Extraction
@app.post("/categorize")
def categorize_expense(input: ExpenseInput):

    prompt = f"""
    Extract:
    1. Category (Food, Transport, Shopping, Bills, Entertainment, Health, Education, Other)
    2. Amount (number only)

    Return JSON:
    {{
      "category": "...",
      "amount": 0
    }}

    Expense: "{input.text}"
    """

    try:
        response = model.generate_content(prompt)
        cleaned = re.search(r'\{.*\}', response.text, re.DOTALL).group()
        result = json.loads(cleaned)
    except Exception as e:
        print(f"Error: {e}")
        result = {"category": "Other", "amount": 0}

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

# 📊 Get total expenses
@app.get("/total")
def get_total():
    db = SessionLocal()
    expenses = db.query(Expense).all()
    total = sum(e.amount for e in expenses)
    db.close()
    return {"total_expense": total}

# 📅 Monthly logs (simple version)
@app.get("/logs")
def get_logs():
    db = SessionLocal()
    expenses = db.query(Expense).all()

    data = []
    for e in expenses:
        data.append({
            "text": e.text,
            "category": e.category,
            "amount": e.amount
        })

    db.close()
    return {"expenses": data}

# 📂 Category-wise total
@app.get("/category-summary")
def category_summary():
    db = SessionLocal()
    expenses = db.query(Expense).all()

    summary = {}
    for e in expenses:
        summary[e.category] = summary.get(e.category, 0) + e.amount

    db.close()
    return summary
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port)    