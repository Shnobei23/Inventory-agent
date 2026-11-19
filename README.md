# Inventory Decision Agent (BAN 517)

## Run locally (PowerShell, from this folder)
1. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and paste your Anthropic API key.
2. `python -m pip install -r requirements.txt; streamlit run app.py`

## Deploy (Streamlit Community Cloud)
1. Push this folder to a GitHub repo (secrets.toml is git-ignored, so it is never uploaded).
2. share.streamlit.io -> Create app -> pick the repo, main file `app.py`.
3. Advanced settings -> Secrets: `ANTHROPIC_API_KEY = "sk-ant-..."` -> Deploy.

## Files
- `inventory_tools.py`: all calculations (EOQ, quantity discount, SS/ROP, fixed-period, newsvendor, classification). Unit conversions happen here, not in the LLM.
- `app.py`: Streamlit UI and the Claude tool-calling agent.
