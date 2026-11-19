# Официальная проверка: только stdlib Python 3.11, без сети, GPU и платных API.
#   docker build -t audit-env .
#   docker run --rm --network none --cpus 2 --memory 4g audit-env            # = python check.py
#   docker run --rm --network none audit-env python -m env.demo              # демо
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY requirements.txt ./
# requirements.txt не содержит сторонних пакетов; шаг оставлен, чтобы фиксация зависимостей была явной.
RUN pip install -r requirements.txt
COPY . .
RUN useradd --create-home --uid 1000 auditor && chown -R auditor /app
USER auditor

CMD ["python", "check.py"]
