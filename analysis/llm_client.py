"""
analysis.llm_client
~~~~~~~~~~~~~~~~~~~
Handles interaction with the local LLM service (Ollama).
"""
import toml
import pathlib
from openai import OpenAI

# --- Load Configuration ---
config_path = pathlib.Path(__file__).parent.parent / "config/env.toml"
if not config_path.exists():
    raise FileNotFoundError(f"Configuration file not found at {config_path}")

config = toml.load(config_path)
llm_config = config.get("llm", {})

# --- Initialize Client ---
if llm_config.get("provider") == "ollama":
    client = OpenAI(
        base_url=llm_config.get("base_url"),
        api_key=llm_config.get("api_key", "ollama"), # api_key is required but not used by Ollama
    )
    MODEL = llm_config.get("model")
else:
    # Placeholder for other providers like Hugging Face Transformers
    client = None
    MODEL = None
    print("Warning: LLM provider not configured or supported.")

def summarize(text: str) -> str:
    """
    Summarizes the given text using the configured local LLM.
    """
    if not client or not MODEL:
        return "Error: LLM client not initialized."

    system_prompt = "You are a concise news summarizer. Respond in Japanese in 3 bullet points."
    
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            temperature=0.2,
            max_tokens=400,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error during summarization with Ollama: {e}")
        return "Summarization failed."

# ---------- CLI self-test ----------
if __name__ == "__main__":
    sample_text = """
    In a move that shocked markets, the central bank of a major economic powerhouse announced a surprise interest rate hike of 50 basis points. 
    The decision was made to combat rising inflation, which has been a persistent issue for the past several quarters. 
    Analysts are now scrambling to predict the short-term and long-term effects on the global economy.
    """
    print("--- Testing Ollama Summarization ---")
    summary = summarize(sample_text)
    print(f"Original Text:\n{sample_text}")
    print(f"\nSummary:\n{summary}")
