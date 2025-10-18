#!/usr/bin/env python3
"""
Discord -> AutoTrader link bot

Flow:
- In DMs: paste Title -> bot asks Mileage -> bot replies with the link.
- In servers: start with `!car <TITLE>` -> bot asks Mileage in that channel -> reply with link.
- Mileage accepts 50000 / 50,000 / 50k / 50.5k / "50k miles".

ENV (.env or real env):
  DISCORD_BOT_TOKEN=...
  OPENAI_API_KEY=...

Run:
  pip install -r requirements.txt
  python discord_bot_autotrader.py
"""

import os, re, json, logging, unicodedata
from urllib.parse import urlencode

import requests
import discord
from discord import Intents
from dotenv import load_dotenv

from keep_alive import keep_alive
keep_alive()


# ---------------- Constants ----------------
POSTCODE = "N14 4EG"
MILEAGE_BAND = 10_000
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL = "gpt-4o-mini"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("autotrader-discord")

# ---------------- Prompt tuned for AutoTrader ----------------
SYSTEM_PROMPT = """
You convert UK car auction titles into AutoTrader search parameters.

Return ONLY compact JSON, no commentary. Use these keys when applicable:
- "year-from" (int), "year-to" (int)  -> MUST both exist, identical to the year in the title.
- "make" (string)
- "model" (string)
- "body-type" (Hatchback|Saloon|Estate|Coupe|Convertible|SUV|MPV|Pickup)
- "fuel-type" (Petrol|Diesel|Hybrid|Electric)
- "transmission" (Manual|Automatic)
- "quantity-of-doors" (int)
- "trim" (equipment line like "M Sport", "AMG Line", etc.)
- Exactly one when detectable:
    * "aggregatedTrim"  -> BMW/Mercedes/MINI badges like "118d", "320i", "E220d", "Cooper S"
    * "model-variant"   -> Audi/Vauxhall/VW/Škoda/SEAT/Cupra variants/lines like
                           "35 TFSI", "40 TDI", "S line", "Sport", "Technik",
                           "Edition 1", "R-Line", "Life", "Style", "FR", "SE L",
                           "vRS", "GTI", "GTD", "Design", "Elite Edition", "GS Line",
                           "SE Edition", "SRi Edition", "Ultimate", "Turbo", "Active",
                           "Exclusiv", "SE", "SRi", "SXi", "ecoFLEX"
Rules:
- Audi/Vauxhall/VAG lines -> "model-variant" (NEVER in "aggregatedTrim").
- BMW/Mercedes/MINI badges -> "aggregatedTrim" (normalize to lowercase suffix: 118d, 320i).
- If unclear, omit that specific key. Never invent a different make/model/year.
"""

USER_PROMPT_TEMPLATE = """Title: {title}

Return JSON like:
{{
  "year-from": 2022, "year-to": 2022,
  "make": "Audi", "model": "A3",
  "model-variant": "35 TFSI",
  "trim": "S line",
  "fuel-type": "Petrol",
  "transmission": "Manual",
  "body-type": "Hatchback",
  "quantity-of-doors": 5
}}"""

# ---------------- Helpers ----------------
def to_ascii(s: str) -> str:
    if not isinstance(s, str):
        return s
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")

CANON_MAKE = {
    "AUDI": "Audi", "BMW": "BMW", "MERCEDES-BENZ": "Mercedes-Benz", "MERCEDES": "Mercedes-Benz",
    "VAUXHALL": "Vauxhall", "VOLKSWAGEN": "Volkswagen", "SKODA": "Skoda", "SEAT": "SEAT",
    "CUPRA": "Cupra", "MINI": "Mini", "FIAT": "Fiat", "FORD": "Ford", "RENAULT": "Renault",
    "PEUGEOT": "Peugeot", "CITROEN": "Citroen", "KIA": "Kia", "HYUNDAI": "Hyundai",
    "NISSAN": "Nissan", "HONDA": "Honda", "JAGUAR": "Jaguar", "LAND ROVER": "Land Rover",
    "TOYOTA": "Toyota", "VOLVO": "Volvo",
}

def canonicalize_make_model(data: dict) -> dict:
    mk = str(data.get("make", "")).strip()
    if mk:
        mk_ascii_up = to_ascii(mk).upper()
        data["make"] = CANON_MAKE.get(mk_ascii_up, to_ascii(mk).title())
    mdl = str(data.get("model", "")).strip()
    if mdl:
        mdl_ascii = to_ascii(mdl).title()
        for pat, repl in [
            (r"\bGti\b", "GTI"), (r"\bGtd\b", "GTD"), (r"\bVrs\b", "vRS"),
            (r"\bSe L\b", "SE L"), (r"\bS Line\b", "S line"), (r"\bR-Line\b", "R-Line"),
        ]:
            mdl_ascii = re.sub(pat, repl, mdl_ascii)
        data["model"] = mdl_ascii
    return data

def normalize_mileage(text: str) -> int | None:
    t = text.strip().lower()
    m = re.search(r"(\d+(?:\.\d+)?)\s*k\b", t)
    if m: return int(round(float(m.group(1)) * 1000))
    digits = re.findall(r"\d{1,7}", re.sub(r"[,\s]", "", t))
    return int(digits[0]) if digits else None

# ----- Fuel inference & correction -----
def infer_fuel_from_title_and_params(title: str, params: dict) -> str | None:
    t = to_ascii(title).lower()
    if re.search(r"\bplug[- ]?in\b|\bphev\b|\bhybrid\b", t): return "Hybrid"
    if re.search(r"\belectric\b|\bev\b|\be-tron\b|\bi3\b", t): return "Electric"
    if re.search(r"\btdi\b|\bcdti\b|\bdci\b|\bbluehdi\b|\bdiesel\b", t): return "Diesel"
    if re.search(r"\btfsi\b|\btsi\b|\bmpi\b|\bpetrol\b", t): return "Petrol"
    agg = str(params.get("aggregatedTrim", "")).lower()
    m = re.match(r"^\s*\d{3}([di])\s*$", agg)
    if m: return "Diesel" if m.group(1) == "d" else "Petrol"
    mv = str(params.get("model-variant", "")).lower()
    if "tdi" in mv: return "Diesel"
    if "tfsi" in mv or re.search(r"\b\d{2,3}\s+tsi\b", mv): return "Petrol"
    return None

def apply_fuel_fix(title: str, params: dict) -> dict:
    inferred = infer_fuel_from_title_and_params(title, params)
    existing = params.get("fuel-type")
    if inferred and (not existing or existing != inferred):
        params["fuel-type"] = inferred
    return params

# ----- Brand-specific “variant vs trim” enforcement -----
# Vauxhall older Corsas/Astras etc. show Active/Exclusiv/SE/SRi/SXi/ecoFLEX as **Model variant**
VAUXHALL_VARIANTS = re.compile(
    r"\b(Active|Exclusiv|SE|SRi|SXi|ecoFLEX|Design|Elite Edition|GS Line|SE Edition|Ultimate|Turbo)\b",
    re.I
)
# Audi/VAG families often expose S line/Technik/Black Edition/R-Line/Life/Style/FR/SE L/vRS/GTI/GTD as **Model variant**
AUDI_VAG_VARIANTS = re.compile(
    r"\b(S\s*line|Sport|Technik|Black Edition|Edition\s*1|R-?Line|Life|Style|Match|FR|SE\s*L|vRS|GTI|GTD)\b",
    re.I
)

def move_trim_to_variant_if_needed(make: str, data: dict) -> dict:
    """If model put a brand-specific line into 'trim', move it to 'model-variant'."""
    if not data.get("trim"):
        return data
    trim_val = str(data["trim"]).strip()

    if make.lower() == "vauxhall" and VAUXHALL_VARIANTS.search(trim_val):
        data["model-variant"] = trim_val
        data.pop("trim", None)
        return data

    if make.lower() in {"audi", "volkswagen", "skoda", "škoda", "seat", "cupra"} and AUDI_VAG_VARIANTS.search(trim_val):
        data["model-variant"] = trim_val
        data.pop("trim", None)
        return data

    # BMW/Mercedes keep equipment lines (M Sport / AMG Line) in 'trim' – leave as is.
    return data

# ---------------- OpenAI mapping ----------------
def ai_map_title_to_params(openai_key: str, title: str) -> dict:
    headers = {"Authorization": f"Bearer {openai_key}", "Content-Type": "application/json"}
    payload = {
        "model": OPENAI_MODEL,
        "temperature": 0.1,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(title=title)},
        ],
        "response_format": {"type": "json_object"},
    }
    resp = requests.post(OPENAI_URL, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    data = json.loads(content)

    if "year-from" not in data:
        raise ValueError(f"OpenAI mapping missing 'year-from'. Got: {data}")
    if "year-to" not in data or data["year-to"] in (None, "", 0):
        data["year-to"] = data["year-from"]
    for k in ("make", "model"):
        if k not in data:
            raise ValueError(f"OpenAI mapping missing '{k}'. Got: {data}")

    make = str(data["make"]).strip()

    def looks_like_audi_vag_variant(s: str) -> bool:
        s2 = str(s).strip()
        if re.match(r"^\d{2,3}\s+(TFSI|TDI)\b", s2, re.I): return True
        return bool(AUDI_VAG_VARIANTS.search(s2) or VAUXHALL_VARIANTS.search(s2))

    def looks_like_bmw_badge(s: str) -> bool:
        return bool(re.match(r"^\s*(\d{3})([di])\s*$", str(s), re.I)) or \
               bool(re.match(r"^\s*Cooper\s*S\b", str(s), re.I)) or \
               bool(re.match(r"^\s*(C|E|A|S)\d{3}d?\s*$", str(s), re.I))

    # Move misplaced fields (aggTrim <-> model-variant)
    if data.get("aggregatedTrim") and looks_like_audi_vag_variant(data["aggregatedTrim"]):
        data["model-variant"] = data.pop("aggregatedTrim")
    if data.get("model-variant") and make in {"BMW", "Mercedes-Benz", "Mini"} and looks_like_bmw_badge(data["model-variant"]):
        data["aggregatedTrim"] = data.pop("model-variant")

    # Normalize BMW badge casing
    if data.get("aggregatedTrim"):
        m = re.match(r"^\s*(\d{3})([di])\s*$", str(data["aggregatedTrim"]), re.I)
        if m: data["aggregatedTrim"] = f"{m.group(1)}{m.group(2).lower()}"

    # Tidy casing
    if data.get("trim"):
        data["trim"] = str(data["trim"]).strip().title()
    if data.get("model-variant"):
        mv = str(data["model-variant"]).strip()
        mv = re.sub(r"\bS\s*Line\b", "S line", mv, flags=re.I)
        mv = re.sub(r"\bR-?Line\b", "R-Line", mv, flags=re.I)
        mv = re.sub(r"\bSe\s*L\b", "SE L", mv, flags=re.I)
        mv = re.sub(r"\bSri\b", "SRi", mv, flags=re.I)
        mv = re.sub(r"\bGti\b", "GTI", mv, flags=re.I)
        mv = re.sub(r"\bGtd\b", "GTD", mv, flags=re.I)
        mv = re.sub(r"\bTfsi\b", "TFSI", mv, flags=re.I)
        mv = re.sub(r"\bTdi\b", "TDI", mv, flags=re.I)
        data["model-variant"] = mv

    # Canonicalize make/model
    data = canonicalize_make_model(data)

    # NEW: If brand uses "Model variant" for lines we got in trim, move them
    data = move_trim_to_variant_if_needed(data["make"], data)

    return data

def build_autotrader_url(params_ai: dict, mileage: int) -> str:
    q = {
        "make": params_ai["make"],
        "model": params_ai["model"],
        "year-from": params_ai["year-from"],
        "year-to": params_ai["year-to"],
        "postcode": POSTCODE,
        "sort": "relevance",
        "minimum-mileage": max(0, int(mileage) - MILEAGE_BAND),
        "maximum-mileage": int(mileage) + MILEAGE_BAND,
    }
    if params_ai.get("aggregatedTrim"):
        q["aggregatedTrim"] = params_ai["aggregatedTrim"]
    if params_ai.get("model-variant"):
        q["model-variant"] = params_ai["model-variant"]
    if params_ai.get("trim"):
        q["trim"] = params_ai["trim"]
    for k in ("body-type", "fuel-type", "transmission", "quantity-of-doors"):
        if k in params_ai:
            q[k] = params_ai[k]
    q = {k: to_ascii(v) if isinstance(v, str) else v for k, v in q.items()}
    return "https://www.autotrader.co.uk/car-search?" + urlencode(q, doseq=True)

# ---------------- Discord bot ----------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DISCORD_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")
if not DISCORD_TOKEN:
    raise RuntimeError("DISCORD_BOT_TOKEN not set")

intents = Intents.default()
intents.message_content = True  # enable in Dev Portal too
bot = discord.Client(intents=intents)

# per-conversation state: {(channel_id, user_id): {"stage": "await_title"/"await_mileage", "title": "..."}}
STATE: dict[tuple[int, int], dict] = {}

async def prompt_mileage(channel, user):
    await channel.send(f"{user.mention} Got the title. Now send the **Mileage** (e.g., 50000 / 50,000 / 50k).")

@bot.event
async def on_ready():
    log.info("Logged in as %s", bot.user)

@bot.event
async def on_message(msg: discord.Message):
    if msg.author.bot:
        return

    content = (msg.content or "").strip()
    key = (msg.channel.id, msg.author.id)

    # Commands
    if content.lower() in {"!reset", "/reset"}:
        STATE.pop(key, None)
        await msg.channel.send(f"{msg.author.mention} Reset. Send a title (or use `!car <TITLE>` in servers).")
        return
    if content.lower() in {"!start", "/start", "!car"}:
        STATE[key] = {"stage": "await_title"}
        await msg.channel.send(f"{msg.author.mention} Send the auction **Title** for the car.")
        return
    if content.lower().startswith("!car "):
        title = content[5:].strip()
        if not title:
            await msg.channel.send(f"{msg.author.mention} Send the title after `!car`.")
            return
        STATE[key] = {"stage": "await_mileage", "title": title}
        await prompt_mileage(msg.channel, msg.author)
        return

    # Flow:
    st = STATE.get(key)

    # In DMs, any first message becomes title
    if isinstance(msg.channel, discord.DMChannel) and not st:
        STATE[key] = {"stage": "await_mileage", "title": content}
        await prompt_mileage(msg.channel, msg.author)
        return

    if not st:
        # In servers require !car or explicit start
        return

    if st.get("stage") == "await_title":
        STATE[key] = {"stage": "await_mileage", "title": content}
        await prompt_mileage(msg.channel, msg.author)
        return

    if st.get("stage") == "await_mileage":
        miles = normalize_mileage(content)
        if miles is None:
            await msg.channel.send(f"{msg.author.mention} Couldn't read that. Send mileage like 50000, 50,000, or 50k.")
            return

        title = st.get("title", "").strip()
        if not title:
            STATE.pop(key, None)
            await msg.channel.send(f"{msg.author.mention} I lost the title — try again with `!car <TITLE>`.")
            return

        await msg.channel.send(f"{msg.author.mention} Mapping title…")
        try:
            params = ai_map_title_to_params(OPENAI_API_KEY, title)
            params = apply_fuel_fix(title, params)
            url = build_autotrader_url(params, miles)
        except Exception as e:
            logging.exception("Mapping failed")
            STATE.pop(key, None)
            await msg.channel.send(f"{msg.author.mention} OpenAI mapping failed: {e}")
            return

        pretty = json.dumps(params, ensure_ascii=False)
        await msg.channel.send(
            f"{msg.author.mention} ✅ Parameters:\n```json\n{pretty}\n```\n"
            f"🔗 AutoTrader (±{MILEAGE_BAND:,} miles):\n{url}"
        )
        STATE.pop(key, None)
        return

if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)
