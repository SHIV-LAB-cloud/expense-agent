FROM python:3.10

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

FROM python:3.10

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# ✅ IMPORTANT: Use PORT from environment
CMD exec uvicorn main:app --host 0.0.0.0 --port $PORT