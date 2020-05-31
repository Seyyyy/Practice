FROM python:3.8.2-slim

RUN apt-get -y update && apt-get -y install \
build-essential

RUN pip install \
flask \
uWSGI

WORKDIR /app
COPY app /app

CMD ["uwsgi", "--http", "0.0.0.0:9090", "--wsgi-file", "/app/app.py",\
    "--callable", "app", "--stats", "0.0.0.0:9191"]