FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    poppler-utils \
    tesseract-ocr \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

# Copy app code
COPY . .

# Expose port
EXPOSE 8501

CMD ["streamlit", "run", "RAG_streamlit.py", "--server.address=0.0.0.0", "--server.port=8501"]