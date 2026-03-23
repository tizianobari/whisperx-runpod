FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

WORKDIR /app

# System Dependencies
RUN apt-get update && apt-get install -y ffmpeg git

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Handler kopieren
COPY handler.py /app/

CMD ["python", "-u", "handler.py"]