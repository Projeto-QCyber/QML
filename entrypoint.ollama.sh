#!/bin/bash

# Start Ollama in the background.
/bin/ollama serve &
# Record Process ID.
pid=$!

# Pause briefly for Ollama to start.
sleep 5

echo "🔴 Retrieve Qwen3 8B q4_K_M Model"
ollama pull qwen3:8b-q4_K_M || echo "Model pull failed or already present. Continuing..."
echo "🟢 Done!"

# Wait for Ollama process to finish.
wait $pid
