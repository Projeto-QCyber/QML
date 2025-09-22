#!/usr/bin/env bash


# Espera o MySQL ficar pronto
/wait-for-it.sh mysql:3306 --timeout=35 --strict -- echo "MySQL está pronto! Ouvindo na porta interna 3306"

# Executa bootstrap do banco antes de iniciar a API (única vez no boot)
python - <<'PY'
import sys
print('[ENTRYPOINT] Executando bootstrap inicial do banco...')
try:
    from qml.api.create_qcyber_db import ensure_bootstrap
    ok = ensure_bootstrap()
    print(f'[ENTRYPOINT] Bootstrap concluído: {ok}')
except Exception as e:
    print(f'[ENTRYPOINT] Falha no bootstrap: {e}')
PY

python -m gunicorn --timeout 900 --bind 0.0.0.0:$FLASK_API_PORT --workers 4 qml.api.main_api:app