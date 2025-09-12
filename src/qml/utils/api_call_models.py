from crewai import LLM
# -----------------------------------------------------------------------------
# Helper factories to avoid duplication
# -----------------------------------------------------------------------------

def _llm_default() -> LLM:
    return LLM(
        model="ollama/qwen3:4b", 
        base_url="http://localhost:11434",
        api_key="ollama",
        temperature=0.2,
        max_tokens=400,
        extra_body={"keep_alive": -1, "stop": ["\n\n", "\n###", "```"]},
    )

def _llm_leader() -> LLM:
    return LLM(
        model="ollama/qwen3:4b",
        base_url="http://localhost:11434",
        api_key="ollama",
        temperature=0.2,
        max_tokens=700,
        extra_body={"keep_alive": -1, "stop": ["\n\n", "\n###", "```"]},
    )