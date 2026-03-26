from fastapi import FastAPI
from pydantic import BaseModel
import google.generativeai as genai
import os
import json
import re
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

app = FastAPI(title="Expense Tracker Agent")

# Database
engine = create_engine("sqlite:///expenses.db")
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class Expense(Base):
    __tablename__ = "expenses"
    id = Column(Integer, primary_key=True)
    text = Column(String)
    category = Column(String)
    amount = Column(Integer)

Base.metadata.create_all(bind=engine)

class ExpenseInput(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "API Running 🚀", "docs": "/docs"}

@app.post("/categorize")
def categorize(input: ExpenseInput):

    prompt = f"""
Classify this expense.

Categories:
Food, Transport, Shopping, Bills, Entertainment, Health, Education, Other

Return JSON:
{{"category":"Transport","amount":250}}

Expense: {input.text}
"""

    try:
        response = model.generate_content(prompt)
        text_response = (response.text or "").strip()

        print("RAW:", text_response)

        # Try JSON
        try:
            result = json.loads(text_response)
        except:
            text_lower = text_response.lower()

            category = "Other"
            if "transport" in text_lower:
                category = "Transport"
            elif "food" in text_lower:
                category = "Food"

            amount_match = re.findall(r'\d+', text_response)
            amount = int(amount_match[0]) if amount_match else 0

            result = {"category": category, "amount": amount}

    except Exception as e:
        print("Error:", e)
        result = {"category": "Other", "amount": 0}

    # Fallback
    text = input.text.lower()
    if result["category"] == "Other":
        if "uber" in text or "ola" in text:
            result["category"] = "Transport"

    # Save
    db = SessionLocal()
    db.add(Expense(text=input.text, category=result["category"], amount=result["amount"]))
    db.commit()
    db.close()

    return result

@app.get("/total")
def total():
    db = SessionLocal()
    total = sum(e.amount for e in db.query(Expense).all())
    db.close()
    return {"total": total}