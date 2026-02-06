# LLMSQL2 - Text-to-SQL with TinyLlama and GPT
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies in stages to avoid disk space issues
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    transformers>=4.36.0 \
    accelerate>=0.25.0 \
    pandas>=2.0.0 \
    numpy>=1.24.0

RUN pip install --no-cache-dir \
    sqlparse>=0.4.4 \
    pg8000>=1.31.0 \
    openai>=1.0.0 \
    python-dotenv>=1.0.0 \
    tqdm>=4.66.0 \
    huggingface-hub>=0.20.0

RUN pip install --no-cache-dir \
    matplotlib>=3.7.0 \
    seaborn>=0.12.0 \
    jupyter>=1.0.0 \
    jupyterlab>=4.0.0

# Flask and PEFT for web app
RUN pip install --no-cache-dir \
    flask>=3.0.0 \
    peft>=0.10.0

# Copy application code
COPY src/ ./src/
COPY notebooks/ ./notebooks/
COPY templates/ ./templates/

# Create directories
RUN mkdir -p results data

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Expose Jupyter port and Web App port
EXPOSE 8888
EXPOSE 5000

# Default command - start Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]
