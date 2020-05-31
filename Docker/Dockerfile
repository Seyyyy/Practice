FROM python:3.8.2-slim AS prod

RUN groupadd -r uwsgi && useradd -r -g uwsgi uwsgi

RUN apt-get -y update && apt-get -y install \
build-essential

RUN pip install \
flask \
uWSGI

WORKDIR /app
COPY app /app

EXPOSE 9090 9191

USER uwsgi

CMD ["uwsgi", "--http", "0.0.0.0:9090", "--wsgi-file", "/app/app.py",\
    "--callable", "app", "--stats", "0.0.0.0:9191"]



FROM python:3.8.2-slim AS dev

RUN groupadd -r uwsgi && useradd -r -g uwsgi uwsgi

RUN apt-get -y update && apt-get -y install \
build-essential

RUN pip install \
flask \
uWSGI

WORKDIR /app
COPY app /app

EXPOSE 9090 9191

USER uwsgi

CMD ["python", "app.py"]