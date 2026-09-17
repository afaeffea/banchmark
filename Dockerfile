FROM python:3.12-slim

# libsndfile нужен soundfile для чтения и записи WAV
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY common/ ./common/
COPY stt_tts_bench/ ./stt_tts_bench/
COPY toolcalling_bench/ ./toolcalling_bench/
COPY webui/ ./webui/

# бенчмарки пишут сюда до переноса в Postgres; том не нужен — данные уезжают в базу
RUN mkdir -p results/logs data

ENV PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8

# Воркер строго один: состояние текущего прогона (подпроцесс, накопленный лог) живёт
# в памяти процесса, при нескольких воркерах запросы попадали бы в разные копии.
# Параллелизм даётся потоками — запросов мало, работа в них ожидающая.
# Долгий таймаут: прогон бенчмарка идёт минутами, а поток-читатель лога не должен убиваться.
CMD ["sh", "-c", "gunicorn --workers 1 --threads 8 --timeout 600 --bind 0.0.0.0:${PORT:-8080} webui.server:app"]
