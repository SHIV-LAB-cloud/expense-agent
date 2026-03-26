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

Return ONLY JSON:
{{"category":"Transport","amount":250}}

Expense: {input.text}
"""

    try:
        response = model.generate_content(prompt)
        text_response = (response.text or "").strip()

        print("RAW RESPONSE:", text_response)

        # Try JSON first
        try:
            result = json.loads(text_response)
        except:
            text_lower = text_response.lower()

            category = "Other"
            if "transport" in text_lower:
                category = "Transport"
            elif "food" in text_lower:
                category = "Food"

            # Extract amount
            amount_match = re.findall(r'\d+', text_response)
            amount = int(amount_match[0]) if amount_match else 0

            result = {
                "category": category,
                "amount": amount
            }

    except Exception as e:
        print("Error:", e)
        result = {"category": "Other", "amount": 0}

    # 🔥 Final fallback (VERY IMPORTANT)
    text = input.text.lower()

    if result["category"] == "Other":
        if any(word in text for word in ["uber", "ola", "taxi", "auto", "metro"]):
            result["category"] = "Transport"
        elif any(word in text for word in ["pizza", "food", "restaurant", "swiggy", "zomato"]):
            result["category"] = "Food"

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