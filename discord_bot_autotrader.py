#!/usr/bin/env python3
"""
Discord -> AutoTrader + Car Flip Cost Calculator Bot (Text command version)
"""

import os, re, json, logging, unicodedata
from urllib.parse import urlencode
import requests
import discord
from discord import Intents
from dotenv import load_dotenv

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

Return ONLY compact JSON, no commentary.
"""
USER_PROMPT_TEMPLATE = """Title: {title}

Return JSON like:
{
  "year-from": 2020, "year-to": 2020,
  "make": "BMW", "model": "3 Series",
  "aggregatedTrim": "320i", "trim": "M Sport",
  "fuel-type": "Petrol", "transmission": "Automatic",
  "body-type": "Saloon", "quantity-of-doors": 4
}"""

# ---------------- Helpers ----------------
def to_ascii(s: str) -> str:
    if not isinstance(s, str):
        return s
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")

def normalize_mileage(text: str) -> int | None:
    t = text.strip().lower()
    m = re.search(r"(\d+(?:\.\d+)?)\s*k\b", t)
    if m:
        return int(round(float(m.group(1)) * 1000))
    digits = re.findall(r"\d{1,7}", re.sub(r"[,\s]", "", t))
    return int(digits[0]) if digits else None

# ---------------- Cost Calculator ----------------
def get_variable_fee(price: float) -> float:
    ranges = [
        (0, 249.99, 30), (250, 499.99, 50), (500, 749.99, 100),
        (750, 999.99, 120), (1000, 1499.99, 160), (1500, 1999.99, 210),
        (2000, 2499.99, 260), (2500, 2999.99, 320), (3000, 3999.99, 390),
        (4000, 4999.99, 460), (5000, 7499.99, 530), (7500, 9999.99, 550),
        (10000, 12499.99, 680), (12500, 14999.99, 830),
    ]
    for low, high, fee in ranges:
        if low <= price <= high:
            return fee
    return price * 0.06

def calculate_total(price: float, distance: float) -> dict:
    processing = 90
    admin = 40
    variable = get_variable_fee(price)
    fees = processing + admin + variable
    vat = fees * 0.2
    fees_with_vat = fees + vat
    fuel = (distance * 3 / 40) * 1.39 * 4.546
    extras = 50 + 50 + 30
    total = price + fees_with_vat + fuel + extras
    return {
        "price": price,
        "processing": processing,
        "admin": admin,
        "variable": variable,
        "vat": vat,
        "fuel": fuel,
        "extras": extras,
        "total": total,
    }

def format_cost(data: dict) -> str:
    return (
        f"💸 **Car Flip Cost Breakdown**\n\n"
        f"🔹 Auction Buy Price: £{data['price']:.2f}\n"
        f"🔹 Item Processing Fee: £{data['processing']:.2f}\n"
        f"🔹 Admin Fee: £{data['admin']:.2f}\n"
        f"🔹 Auction Fee: £{data['variable']:.2f}\n"
        f"🔹 VAT (20%): £{data['vat']:.2f}\n"
        f"🔹 Fuel (x3 trip): £{data['fuel']:.2f}\n"
        f"🔹 Extras: £{data['extras']:.2f}\n"
        f"\n---------------------------------\n"
        f"🧾 **Total Cost: £{data['total']:.2f}**"
    )

# ---------------- AutoTrader AI Mapping ----------------
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
    data = json.loads(resp.json()["choices"][0]["message"]["content"])
    return data

def build_autotrader_url(params_ai: dict, mileage: int) -> str:
    q = {
        "make": params_ai.get("make"),
        "model": params_ai.get("model"),
        "year-from": params_ai.get("year-from"),
        "year-to": params_ai.get("year-to"),
        "postcode": POSTCODE,
        "sort": "relevance",
        "minimum-mileage": max(0, int(mileage) - MILEAGE_BAND),
        "maximum-mileage": int(mileage) + MILEAGE_BAND,
    }
    return "https://www.autotrader.co.uk/car-search?" + urlencode(q, doseq=True)

# ---------------- Discord Setup ----------------
load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not DISCORD_TOKEN or not OPENAI_API_KEY:
    raise RuntimeError("Missing keys in .env")

intents = Intents.default()
intents.message_content = True
bot = discord.Client(intents=intents)
STATE = {}

async def prompt_mileage(channel, user):
    await channel.send(f"{user.mention} Got the title. Now send the **Mileage** (e.g., 50k or 50000).")

@bot.event
async def on_ready():
    log.info(f"✅ Logged in as {bot.user}")

@bot.event
async def on_message(msg: discord.Message):
    if msg.author.bot:
        return

    content = (msg.content or "").strip()
    parts = content.split()
    key = (msg.channel.id, msg.author.id)

    # --- !cost ---
    if content.lower().startswith("!cost"):
        try:
            _, price, distance = parts
            price = float(price)
            distance = float(distance)
            data = calculate_total(price, distance)
            await msg.channel.send(format_cost(data))
        except Exception:
            await msg.channel.send("Usage: `!cost <price> <distance>`")
        return

    # --- !compare ---
    if content.lower().startswith("!compare"):
        try:
            _, distance, *prices = parts
            distance = float(distance)
            prices = [float(p) for p in prices]
            if not prices:
                await msg.channel.send("Usage: `!compare <distance> <price1> <price2> ...`")
                return
            lines = [f"📊 **Cost Comparison (Distance {distance} miles)**\n"]
            for p in prices:
                d = calculate_total(p, distance)
                lines.append(f"• £{p:.0f} → **£{d['total']:.2f}** total")
            await msg.channel.send("\n".join(lines))
        except Exception:
            await msg.channel.send("Usage: `!compare <distance> <price1> <price2> ...`")
        return

    # --- !car ---
    if content.lower().startswith("!car "):
        title = content[5:].strip()
        if not title:
            await msg.channel.send(f"{msg.author.mention} Send the title after `!car`.")
            return
        STATE[key] = {"stage": "await_mileage", "title": title}
        await prompt_mileage(msg.channel, msg.author)
        return

    if content.lower() in {"!reset", "/reset"}:
        STATE.pop(key, None)
        await msg.channel.send(f"{msg.author.mention} Reset. Send a title with `!car <TITLE>`.")
        return

    st = STATE.get(key)
    if not st:
        return

    if st.get("stage") == "await_mileage":
        miles = normalize_mileage(content)
        if miles is None:
            await msg.channel.send(f"{msg.author.mention} Couldn't read that. Send mileage like 50k or 50000.")
            return

        title = st.get("title", "").strip()
        await msg.channel.send(f"{msg.author.mention} Mapping title…")
        try:
            params = ai_map_title_to_params(OPENAI_API_KEY, title)
            url = build_autotrader_url(params, miles)
        except Exception as e:
            log.exception("Mapping failed")
            STATE.pop(key, None)
            await msg.channel.send(f"{msg.author.mention} OpenAI mapping failed: {e}")
            return

        make = params.get("make", "")
        model = params.get("model", "")
        year = params.get("year-from", "")
        display_title = f"{year} {make} {model}".strip()

        await msg.channel.send(
            f"{msg.author.mention} 🔎 **{display_title}**\n"
            f"🔗 {url}"
        )
        STATE.pop(key, None)

# ---------------- Keep Alive & Run ----------------
if __name__ == "__main__":
    from keep_alive import keep_alive
    keep_alive()
    bot.run(DISCORD_TOKEN)
