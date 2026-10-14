"""
classify.py - Week 1, step C2: AI labelling that fails loudly.

Sends each of the 30 'main' rows to Groq, validates the reply WITHOUT
repairing it, records every failure, and reports the parse failure rate.

Usage (run from the repo folder):
    python classify.py --prompt p1        # Prompt 1 exactly as given (one word)
    python classify.py --prompt p2        # Prompt 2 exactly as given (one word)
    python classify.py --prompt p1_json   # Prompt 1 rules + JSON {label, confidence}

Output: ai_<prompt>.csv  (id, ai_label, ai_confidence, status, finish_reason, raw_reply)
Never reads the official_label column.
"""

import argparse
import json
import os
import sys
import time

import pandas as pd
from dotenv import load_dotenv
import groq
from groq import Groq

# ---- Config -----------------------------------------------------------------
DATA_FILE = "week1_data.csv"
ID_COL, TEXT_COL, SET_COL = "id", "text", "set"   # change if your columns differ
LABELS = {"World", "Sports", "Business", "Sci/Tech"}
PAUSE_SECONDS = 4          # free tier: 30 requests / 8,000 tokens per minute
MAX_RETRIES = 5
MAX_TOKENS = 1024          # room for reasoning models to think AND answer
DEFAULT_MODEL = "openai/gpt-oss-120b"  # check console.groq.com/docs/models

# ---- Prompts (copied from the Appendix) -------------------------------------
RULES = """You are classifying news items into exactly one of four categories.
World  international news, politics, conflict, disasters, national affairs
Sports  matches, results, athletes, teams, transfers, tournaments
Business  companies, markets, earnings, the economy, trade, jobs, deals
Sci/Tech  science, research, technology, software, space, health research
Boundary rules, apply them in this order:
1. A foreign company's earnings or a currency move is Business, not World.
2. A broadcasting rights deal or a club's finances is Business, not Sports.
3. A product's technical capability is Sci/Tech. Its commercial impact is Business.
4. If two categories still fit, choose the one the item is mostly about."""

PROMPTS = {
    "p1": RULES + """
Reply with exactly one of: World, Sports, Business, Sci/Tech
No explanation. No punctuation. One word only.
News item:
{text}""",
    "p2": """What category is this news item? Pick one.
World, Sports, Business, Sci/Tech
News item:
{text}""",
    "p1_json": RULES + """
Reply with JSON only, no other text, in exactly this shape:
{{"label": "<World|Sports|Business|Sci/Tech>", "confidence": <number between 0 and 1>}}
News item:
{text}""",
}


# ---- Validation: record failures, never coerce ------------------------------
def parse_word(reply: str):
    if not reply.strip():
        return None, None, "empty_reply"
    word = reply.strip()          # whitespace only; no case fixing, no guessing
    if word in LABELS:
        return word, None, "ok"
    return None, None, "invalid_label"


def parse_json(reply: str):
    if not reply.strip():
        return None, None, "empty_reply"
    try:
        data = json.loads(reply.strip())
    except json.JSONDecodeError:
        return None, None, "json_parse_error"
    if not isinstance(data, dict):
        return None, None, "json_parse_error"
    label, conf = data.get("label"), data.get("confidence")
    if label not in LABELS:
        return None, None, "invalid_label"
    if not isinstance(conf, (int, float)) or not 0 <= conf <= 1:
        return None, None, "invalid_confidence"
    return label, float(conf), "ok"


# ---- Groq call with retry-after handling ------------------------------------
def call_model(client, model, prompt):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            extra = {}
            if model.startswith("openai/gpt-oss"):     # reasoning model
                extra["reasoning_effort"] = "low"
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,      # repeatable results
                max_completion_tokens=MAX_TOKENS,
                **extra,
            )
            choice = resp.choices[0]
            return choice.message.content or "", choice.finish_reason
        except groq.RateLimitError as e:
            wait = float(e.response.headers.get("retry-after", 10))
            print(f"  429 rate limited, waiting {wait:.0f}s (attempt {attempt})")
            time.sleep(wait)
        except groq.AuthenticationError:
            sys.exit("401: API key is wrong or not loaded. Check your .env file.")
        except groq.NotFoundError as e:
            sys.exit(f"404: model '{model}' not available to your key. "
                     f"Run: python classify.py --list-models\n{e}")
        except (groq.BadRequestError, groq.PermissionDeniedError) as e:
            sys.exit(f"Request rejected, stopping (fix this before re-running):\n{e}")
    raise RuntimeError("Still rate limited after retries")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", choices=PROMPTS)
    ap.add_argument("--list-models", action="store_true",
                    help="print the model IDs your key can use, then exit")
    ap.add_argument("--force", action="store_true", help="overwrite existing output")
    args = ap.parse_args()

    load_dotenv()
    key = os.getenv("GROQ_API_KEY")
    if not key:
        sys.exit("GROQ_API_KEY missing. Add it to your .env file.")

    if args.list_models:
        for m in sorted(Groq(api_key=key).models.list().data, key=lambda m: m.id):
            print(m.id)
        return
    if not args.prompt:
        sys.exit("Choose a prompt: --prompt p1 | p2 | p1_json")

    out_file = f"ai_{args.prompt}.csv"
    if os.path.exists(out_file) and not args.force:
        sys.exit(f"{out_file} already exists. Use --force to re-run (spends quota).")

    model = os.getenv("GROQ_MODEL", DEFAULT_MODEL)

    df = pd.read_csv(DATA_FILE)
    missing = {ID_COL, TEXT_COL, SET_COL} - set(df.columns)
    if missing:
        sys.exit(f"Columns not found: {missing}. Edit ID_COL/TEXT_COL/SET_COL at the top.")
    main_rows = df[df[SET_COL] == "main"][[ID_COL, TEXT_COL]]
    print(f"{len(main_rows)} rows | prompt={args.prompt} | model={model}")
    with open("run_log.txt", "a") as log:     # record which model made each file
        log.write(f"{time.strftime('%Y-%m-%d %H:%M')} {out_file} model={model}\n")

    client = Groq(api_key=key)
    parser = parse_json if args.prompt.endswith("_json") else parse_word
    results = []

    for i, row in enumerate(main_rows.itertuples(index=False), 1):
        item_id, text = getattr(row, ID_COL), getattr(row, TEXT_COL)
        prompt = PROMPTS[args.prompt].format(text=text)
        try:
            reply, finish = call_model(client, model, prompt)
            label, conf, status = parser(reply)
            if status != "ok" and finish == "length":
                status = "cut_off_by_token_limit"   # our setting, not the model
        except Exception as e:                 # network or other API failure
            reply, finish, label, conf, status = str(e), None, None, None, "api_error"
        results.append({"id": item_id, "ai_label": label, "ai_confidence": conf,
                        "status": status, "finish_reason": finish, "raw_reply": reply})
        print(f"[{i}/{len(main_rows)}] {status:<18} {label}")
        time.sleep(PAUSE_SECONDS)

    out = pd.DataFrame(results)
    out.to_csv(out_file, index=False)

    failures = (out["status"] != "ok").sum()
    print(f"\nSaved {out_file}")
    print(f"Parse failure rate: {failures}/{len(out)} = {failures / len(out):.1%}")
    print(out["status"].value_counts().to_string())


if __name__ == "__main__":
    main()