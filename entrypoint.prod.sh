#!/usr/bin/env bash


MYSQL_HOST="${MYSQL_HOST:?MYSQL_HOST must point to the qcyber_api MySQL instance}"
MYSQL_PORT="${MYSQL_PORT:?MYSQL_PORT must point to the qcyber_api MySQL instance}"

# Espera o MySQL externo do qcyber_api ficar pronto. QML nao cria nem migra schema.
/wait-for-it.sh "${MYSQL_HOST}:${MYSQL_PORT}" --timeout=35 --strict -- echo "MySQL do qcyber_api está pronto em ${MYSQL_HOST}:${MYSQL_PORT}"

python -m gunicorn --timeout 1500 --bind 0.0.0.0:$FLASK_API_PORT --workers 4 qml.api.main_api:app
