FROM python:3.11-slim

# The image ships only backend/, so it serves the ViT frame classifier. Serving the fine-tuned
# video model needs deepshield/, scripts/, assets/face_detector/ and its checkpoint as well.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=7860 \
    SERVE_MODEL=vit

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY backend ./backend

EXPOSE 7860

CMD ["gunicorn", "--bind", "0.0.0.0:7860", "backend.app:app", "--workers", "1", "--timeout", "300"]
