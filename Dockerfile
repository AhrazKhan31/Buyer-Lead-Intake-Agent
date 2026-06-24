# Step 1: Secure lightweight runtime build environment
FROM python:3.11-slim

WORKDIR /app

# System clean-up and dependencies deployment
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Cache Python dependency installation layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy internal modular package directories
COPY src/ ./src/
COPY app.py .
COPY miami_mls_listings.csv .
COPY sample_buyer_inquiries.json .

# Cloud Run relies on default binding variables 
EXPOSE 8080

# Configure production flags for headless Streamlit execution
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]