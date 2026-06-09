FROM vllm/vllm-openai:v0.4.3

# Build argument for Hugging Face authentication
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

ENV MODEL_NAME="google/gemma-2-7b-it"
ENV PORT=8000

# Expose the internal vLLM OpenAI-compatible API port
EXPOSE 8000

ENTRYPOINT python3 -m vllm.entrypoints.openai.api_server \
    --model ${MODEL_NAME} \
    --port ${PORT} \
    --max-model-len 2048 \
    ${VLLM_EXTRA_ARGS}