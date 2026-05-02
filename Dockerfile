# Dockerfile
FROM python:3.12.11-slim

# Set the working directory in the container
WORKDIR /app

RUN apt update
RUN apt install python3-dev default-libmysqlclient-dev build-essential pkg-config -y

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN useradd -m -r appuser && \
   chown -R appuser /app

# Set the PYTHONPATH to include src/
ENV PYTHONPATH=/app/src

# Copy the application code
COPY --chown=appuser:appuser ./IA ./IA
COPY --chown=appuser:appuser ./src ./src
COPY --chown=appuser:appuser ./data ./data
COPY --chown=appuser:appuser ./meta-learning .
COPY --chown=appuser:appuser pyproject.toml .
COPY --chown=appuser:appuser uv.lock .

COPY --chown=appuser:appuser entrypoint.prod.sh .
COPY --chown=appuser:appuser wait-for-it.sh /wait-for-it.sh

USER appuser

# Expose the app port
EXPOSE 5000

# Specify the command to run on container start
# CMD ["crewai", "run"]

RUN chmod +x /wait-for-it.sh
RUN chmod +x /app/entrypoint.prod.sh


# Comando para rodar a api FLASK
CMD ["/app/entrypoint.prod.sh"]
