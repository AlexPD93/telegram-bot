# Telegram Echo Bot

A simple Telegram bot that echoes user messages and responds to basic commands.  
This project uses [python-telegram-bot](https://python-telegram-bot.org/) and loads secrets from a `.env` file.

## Features

- Replies with a greeting on `/start`
- Replies with "Help!" on `/help`
- Replies with the GitHub repo link on `/code`
- Echoes any non-command text message

## Setup

### 1. Clone the repository

```sh
git clone https://github.com/AlexPD93/telegram-bot.git
cd telegram-bot
```

### 2. Create a virtual environment (optional but recommended)

```sh
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```sh
pip install -r requirements.txt
```

### 4. Create a `.env` file

Add your Telegram bot token to a file named `.env` in the project root:

```
TELEGRAM_BOT_TOKEN=your-telegram-bot-token-here
```

### 5. Run the bot

```sh
python echobot.py
```

## Usage

- `/start` — Greets the user.
- `/help` — Shows a help message.
- `/code` — Sends the GitHub repository link.
- Any other text — The bot will echo your message.

## License

This project is dedicated to the public domain under the CC0 license.
