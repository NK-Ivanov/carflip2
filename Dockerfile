# Use lightweight Python image
FROM python:3.11-slim

WORKDIR /app
COPY . .

# Install dependencies
RUN pip install -r requirements.txt

# 💡 Resource hints for Railway (not mandatory but helps optimize usage)
LABEL railway.cpu=0.2
LABEL railway.memory=256Mi

# Expose the port (Railway expects 8080)
EXPOSE 8080

# Start the bot
CMD ["python", "discord_bot_autotrader.py"]
