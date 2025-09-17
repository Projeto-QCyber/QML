#!/usr/bin/env bash


# Espera o MySQL ficar pronto
/wait-for-it.sh $MYSQL_HOST:3306 --timeout=35 --strict -- echo "MySQL está pronto! Ouvindo na porta interna 3306"

python -m gunicorn --bind 0.0.0.0:$FLASK_API_PORT --workers 4 qml.api.main_api:app