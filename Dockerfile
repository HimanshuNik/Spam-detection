# Stage 1: Build & Train
FROM python:3.10-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY backend/requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt

# Copy backend files
COPY backend /app/backend
COPY frontend /app/frontend

WORKDIR /app/backend

# Install the wheels
RUN pip install --no-cache /app/wheels/*

# Train the model during build (so it's baked into the image)
RUN python train_model.py

# Stage 2: Runtime
FROM python:3.10-slim

WORKDIR /app

# Copy the wheels and install them
COPY --from=builder /app/wheels /wheels
RUN pip install --no-cache /wheels/*

# Copy the trained models and application code
COPY --from=builder /app/backend /app/backend
COPY --from=builder /app/frontend /app/frontend

# Download NLTK data to the container
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"

# Setup environment
ENV FLASK_HOST=0.0.0.0
ENV FLASK_PORT=5000
ENV FLASK_DEBUG=0
ENV OPEN_BROWSER=0

EXPOSE 5000

WORKDIR /app/backend

# Use Waitress for production serving
CMD ["python", "app.py"]
