# Manual Trader Assistant

This sub-project adds a GPT-assisted workflow for discretionary USD/JPY trading inside QuantRabbit. It shares the
same market-data and indicator stack as the autonomous bot but stops short of sending orders — the human trader stays
in control.

## Components

- `context.py` – fetches recent M1/H4 candles from OANDA, computes indicators, and packages news + event state.
- `prompt_builder.py` – converts the context into a compact prompt and JSON schema for GPT.
- `gpt_manual.py` – wraps the OpenAI chat call with cost guards and a deterministic fallback when GPT is unavailable.
- `cli.py` – interactive session runner that prints the context, GPT guidance, and optional manual order notes.

## Usage

```bash
python -m manual_trader.cli --instrument USD_JPY
```

Options:
- `--non-interactive` – skip the post-guidance questions (useful in scripts).
- `--no-log` – avoid writing `logs/manual_sessions.jsonl`.
- `--m1-count / --h4-count` – control how many candles are pulled for each timeframe.

## GPT Configuration

The assistant reuses `openai_api_key`, cost guard, and model selection logic. Override the following keys in
`config/env.local.toml` if you want to isolate budgets from the autonomous agent:

```toml
openai_manual_model = "gpt-4o-mini"
openai_manual_max_month_tokens = 150000
openai_manual_max_month_usd = 25
openai_manual_cost_per_million_input = 0.30
openai_manual_cost_per_million_output = 2.50
```

## Session Log

Each run appends a JSON object to `logs/manual_sessions.jsonl` containing:

- the gathered context (technical summary, regimes, news),
- the GPT response,
- optional manual notes (direction, entry plan, stops, etc.).

This makes it easy to review decisions post-trade without touching the automated trade database.

## Roadmap Ideas

- add a lightweight web panel (Streamlit/FastAPI) for richer visualization,
- sync manual notes with Notion or Google Sheets for team-wide journaling,
- plug in a checklist of risk-guard validations before allowing submission.

