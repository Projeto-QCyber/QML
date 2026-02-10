#!/bin/bash

# Start Ollama in the background.
/bin/ollama serve &
# Record Process ID.
pid=$!

# Pause briefly for Ollama to start.
sleep 5

echo "🔴 Retrieve Qwen2.5 7B Instruct q4_K_M Model"
ollama pull qwen2.5:7b-instruct-q4_K_M || echo "Model pull failed or already present. Continuing..."
echo "🟢 Done!"

# Wait for Ollama process to finish.
wait $pid
