FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY sceneforge ./sceneforge
COPY frontend ./frontend
COPY data/motion_library.json ./data/motion_library.json
COPY data/motions ./data/motions
COPY app.py .
RUN useradd --create-home sf && chown -R sf /app
USER sf
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/api/health')"
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
