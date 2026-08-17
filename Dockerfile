FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

ENV HF_HUB_DISABLE_PROGRESS_BARS=1

COPY . .

CMD ["python", "app.py"]