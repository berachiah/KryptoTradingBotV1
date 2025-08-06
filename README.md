# KryptoBotV1 - Binance Crypto Trading Bot

A Python-based cryptocurrency trading bot that uses technical analysis to automate trading on the Binance exchange.

## Features

- Technical Analysis based trading
- Real-time market data processing
- Risk management system
- Stop-loss and take-profit orders
- Logging and alerting system
- Testnet support for safe testing

## Requirements

- Python 3.7+
- ccxt
- pandas
- numpy

## Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

1. Create a Binance account and generate API keys
2. Configure your API credentials in the `crypto_trader.py` file:
   ```python
   API_KEY = "your_binance_api_key_here"
   API_SECRET = "your_binance_api_secret_here"
   ```

## Usage

1. Start with testnet mode (default):
   ```bash
   python crypto_trader.py
   ```

2. For live trading, set `USE_TESTNET = False` in the code

## Risk Warning

This bot is for educational purposes only. Use at your own risk. Cryptocurrency trading is highly volatile and can result in significant financial losses.

## License

MIT License
