import ccxt
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime, timedelta
import json
import os
from typing import Dict, Optional, List
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BinanceCryptoBot:
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        """
        Initialize Binance Trading Bot
        
        Args:
            api_key: Binance API key
            api_secret: Binance API secret
            testnet: Use testnet for testing (default: True)
        """
        self.api_key = api_key
        self.api_secret = api_secret
        
        # Initialize Binance exchange
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',  # spot trading
                'adjustForTimeDifference': True,
                'createMarketBuyOrderRequiresPrice': False
            }
        })
        
        # Set up testnet if required
        if testnet:
            self.exchange.set_sandbox_mode(True)
            # Configure all testnet URLs
            self.exchange.urls = {
                'api': {
                    'public': 'https://testnet.binance.vision/api/v3',
                    'private': 'https://testnet.binance.vision/api/v3',
                    'web': 'https://testnet.binance.vision'
                },
                'test': 'https://testnet.binance.vision/api/v3',
                'www': 'https://testnet.binance.vision'
            }
        
        # Trading parameters (BTC/USDT is one of the most liquid pairs)
        self.symbol = 'BTC/USDT'
        self.base_currency = 'BTC'
        self.quote_currency = 'USDT'
        self.min_profit_threshold = 0.015  # 1.5% minimum profit
        self.max_position_size = 0.1  # 10% of portfolio
        self.stop_loss_pct = 0.02  # 2% stop loss
        self.take_profit_pct = 0.03  # 3% take profit
        
        # Bot state
        self.running = False
        self.positions = {}
        self.trade_history = []
        self.last_prices = []
        
        # Risk management
        self.max_daily_loss = 0.05  # 5% max daily loss
        self.daily_pnl = 0.0
        self.trade_count = 0
        
        logger.info(f"Initialized Binance Bot - Testnet: {testnet}")
    
    # ==================== DATA COLLECTION MODULE ====================
    
    def fetch_balance(self) -> Optional[Dict]:
        """Fetch account balance"""
        try:
            # Load markets first to ensure we have the latest trading pairs
            self.exchange.load_markets()
            balance = self.exchange.fetch_balance()
            return balance
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            return None
    
    def fetch_ticker(self, symbol: str) -> Optional[Dict]:
        """Fetch ticker data for symbol"""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker
        except Exception as e:
            logger.error(f"Error fetching ticker for {symbol}: {e}")
            return None
    
    def fetch_order_book(self, symbol: str, limit: int = 20) -> Optional[Dict]:
        """Fetch order book data"""
        try:
            order_book = self.exchange.fetch_order_book(symbol, limit)
            return order_book
        except Exception as e:
            logger.error(f"Error fetching order book for {symbol}: {e}")
            return None
    
    def fetch_ohlcv(self, symbol: str, timeframe: str = '1m', limit: int = 100) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data and return as DataFrame"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            logger.error(f"Error fetching OHLCV data for {symbol}: {e}")
            return None
    
    # ==================== STRATEGY ENGINE / AI LAYER ====================
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        try:
            # Simple Moving Averages
            df['sma_5'] = df['close'].rolling(window=5).mean()
            df['sma_10'] = df['close'].rolling(window=10).mean()
            df['sma_20'] = df['close'].rolling(window=20).mean()
            
            # Exponential Moving Averages
            df['ema_5'] = df['close'].ewm(span=5).mean()
            df['ema_10'] = df['close'].ewm(span=10).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = df['close'].ewm(span=12).mean()
            exp2 = df['close'].ewm(span=26).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            
            # Volume indicators
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            return df
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return df
    
    def generate_trading_signals(self, df: pd.DataFrame) -> Dict:
        """Generate trading signals based on technical analysis"""
        try:
            if len(df) < 26:  # Need enough data for indicators
                return {'signal': 'HOLD', 'strength': 0, 'reason': 'Insufficient data'}
            
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
            signals = []
            reasons = []
            
            # Moving Average Crossover
            if latest['ema_5'] > latest['ema_10'] and prev['ema_5'] <= prev['ema_10']:
                signals.append(1)  # Bullish
                reasons.append("EMA5 crossed above EMA10")
            elif latest['ema_5'] < latest['ema_10'] and prev['ema_5'] >= prev['ema_10']:
                signals.append(-1)  # Bearish  
                reasons.append("EMA5 crossed below EMA10")
            
            # RSI signals
            if latest['rsi'] < 30:
                signals.append(1)  # Oversold - Buy signal
                reasons.append(f"RSI oversold: {latest['rsi']:.2f}")
            elif latest['rsi'] > 70:
                signals.append(-1)  # Overbought - Sell signal
                reasons.append(f"RSI overbought: {latest['rsi']:.2f}")
            
            # MACD signals
            if latest['macd'] > latest['macd_signal'] and prev['macd'] <= prev['macd_signal']:
                signals.append(1)  # Bullish crossover
                reasons.append("MACD bullish crossover")
            elif latest['macd'] < latest['macd_signal'] and prev['macd'] >= prev['macd_signal']:
                signals.append(-1)  # Bearish crossover
                reasons.append("MACD bearish crossover")
            
            # Bollinger Bands
            if latest['close'] < latest['bb_lower']:
                signals.append(1)  # Price below lower band - Buy
                reasons.append("Price below Bollinger lower band")
            elif latest['close'] > latest['bb_upper']:
                signals.append(-1)  # Price above upper band - Sell
                reasons.append("Price above Bollinger upper band")
            
            # Volume confirmation
            if latest['volume_ratio'] > 1.5:  # High volume
                reasons.append("High volume confirmation")
            
            # Calculate overall signal
            if not signals:
                return {'signal': 'HOLD', 'strength': 0, 'reason': 'No clear signals'}
            
            signal_sum = sum(signals)
            signal_strength = abs(signal_sum) / len(signals)
            
            if signal_sum > 0:
                signal = 'BUY'
            elif signal_sum < 0:
                signal = 'SELL'
            else:
                signal = 'HOLD'
            
            return {
                'signal': signal,
                'strength': signal_strength,
                'reason': '; '.join(reasons),
                'price': latest['close'],
                'rsi': latest['rsi'],
                'macd': latest['macd']
            }
            
        except Exception as e:
            logger.error(f"Error generating trading signals: {e}")
            return {'signal': 'HOLD', 'strength': 0, 'reason': f'Error: {e}'}
    
    # ==================== TRADE EXECUTION MODULE ====================
    
    def place_market_order(self, symbol: str, side: str, amount: float) -> Optional[Dict]:
        """Place a market order"""
        try:
            order = self.exchange.create_market_order(symbol, side, amount)
            logger.info(f"Market order placed: {side} {amount} {symbol}")
            return order
        except Exception as e:
            logger.error(f"Error placing market order: {e}")
            return None
    
    def place_limit_order(self, symbol: str, side: str, amount: float, price: float) -> Optional[Dict]:
        """Place a limit order"""
        try:
            order = self.exchange.create_limit_order(symbol, side, amount, price)
            logger.info(f"Limit order placed: {side} {amount} {symbol} at {price}")
            return order
        except Exception as e:
            logger.error(f"Error placing limit order: {e}")
            return None
    
    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an order"""
        try:
            self.exchange.cancel_order(order_id, symbol)
            logger.info(f"Order {order_id} cancelled")
            return True
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    def get_open_orders(self, symbol: str = None) -> List[Dict]:
        """Get open orders"""
        try:
            orders = self.exchange.fetch_open_orders(symbol)
            return orders
        except Exception as e:
            logger.error(f"Error fetching open orders: {e}")
            return []
    
    # ==================== RISK MANAGEMENT MODULE ====================
    
    def calculate_position_size(self, balance: Dict, current_price: float, risk_pct: float = 0.02) -> float:
        """Calculate position size based on risk management"""
        try:
            # Get available BTC balance
            btc_balance = balance.get('BTC', {}).get('free', 0)
            
            # Calculate position size (max 10% of portfolio)
            max_position_value = btc_balance * self.max_position_size
            risk_position_value = btc_balance * risk_pct
            
            # Use the smaller of the two
            position_value = min(max_position_value, risk_position_value)
            position_size = position_value  # Already in BTC for BTC/ETH pair
            
            # Ensure minimum trade size
            min_notional = 0.001  # Minimum 0.001 BTC
            if position_size < min_notional:
                logger.warning(f"Position size {position_size} below minimum {min_notional}")
                return 0
            
            return round(position_size, 6)
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0
    
    def check_risk_limits(self) -> bool:
        """Check if risk limits are exceeded"""
        try:
            # Check daily loss limit
            if self.daily_pnl < -self.max_daily_loss:
                logger.warning(f"Daily loss limit exceeded: {self.daily_pnl:.4f}")
                return False
            
            # Check maximum number of trades per day
            if self.trade_count > 50:  # Max 50 trades per day
                logger.warning("Maximum daily trades exceeded")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking risk limits: {e}")
            return False
    
    def set_stop_loss_take_profit(self, order: Dict, side: str) -> None:
        """Set stop loss and take profit orders"""
        try:
            if not order or 'id' not in order:
                return
            
            fill_price = float(order.get('average') or order.get('price', 0))
            if fill_price == 0:
                return
            
            amount = float(order['amount'])
            
            if side == 'buy':
                # Set stop loss and take profit for long position
                stop_price = fill_price * (1 - self.stop_loss_pct)
                take_profit_price = fill_price * (1 + self.take_profit_pct)
                
                # Place stop loss order
                self.place_limit_order(self.symbol, 'sell', amount, stop_price)
                # Place take profit order  
                self.place_limit_order(self.symbol, 'sell', amount, take_profit_price)
                
            elif side == 'sell':
                # Set stop loss and take profit for short position
                stop_price = fill_price * (1 + self.stop_loss_pct)
                take_profit_price = fill_price * (1 - self.take_profit_pct)
                
                # Place stop loss order
                self.place_limit_order(self.symbol, 'buy', amount, stop_price)
                # Place take profit order
                self.place_limit_order(self.symbol, 'buy', amount, take_profit_price)
                
        except Exception as e:
            logger.error(f"Error setting stop loss/take profit: {e}")
    
    # ==================== LOGGING + ALERTS MODULE ====================
    
    def log_trade(self, trade_data: Dict) -> None:
        """Log trade to history"""
        try:
            trade_record = {
                'timestamp': datetime.now().isoformat(),
                'symbol': trade_data.get('symbol'),
                'side': trade_data.get('side'),
                'amount': trade_data.get('amount'),
                'price': trade_data.get('price'),
                'order_id': trade_data.get('id'),
                'signal': trade_data.get('signal', {}),
                'pnl': trade_data.get('pnl', 0)
            }
            
            self.trade_history.append(trade_record)
            
            # Save to file
            with open('trade_history.json', 'w') as f:
                json.dump(self.trade_history, f, indent=2)
                
            logger.info(f"Trade logged: {trade_record}")
            
        except Exception as e:
            logger.error(f"Error logging trade: {e}")
    
    def send_alert(self, message: str, alert_type: str = "INFO") -> None:
        """Send alert via Telegram or email"""
        try:
            alert_message = f"🤖 Binance Bot Alert [{alert_type}]\n\n{message}\n\nTime: {datetime.now()}"
            
            # Log the alert
            logger.info(f"ALERT [{alert_type}]: {message}")
            
            # Here you can add Telegram bot integration
            # self.send_telegram_alert(alert_message)
            
            # Or email integration
            # self.send_email_alert(alert_message)
            
        except Exception as e:
            logger.error(f"Error sending alert: {e}")
    
    # ==================== MAIN TRADING STRATEGY ====================
    
    def execute_trading_cycle(self) -> None:
        """Execute one complete trading cycle"""
        try:
            logger.info("=== Starting Trading Cycle ===")
            
            # Check risk limits
            if not self.check_risk_limits():
                logger.warning("Risk limits exceeded, skipping cycle")
                return
            
            # Fetch current data
            balance = self.fetch_balance()
            ticker = self.fetch_ticker(self.symbol)
            ohlcv_df = self.fetch_ohlcv(self.symbol, '1m', 100)
            
            if not all([balance, ticker, ohlcv_df is not None]):
                logger.error("Failed to fetch required data")
                return
            
            # Calculate indicators and generate signals
            df_with_indicators = self.calculate_technical_indicators(ohlcv_df)
            signal_data = self.generate_trading_signals(df_with_indicators)
            
            current_price = ticker['last']
            
            logger.info(f"Current Price: {current_price}")
            logger.info(f"Signal: {signal_data['signal']} (Strength: {signal_data['strength']:.2f})")
            logger.info(f"Reason: {signal_data['reason']}")
            
            # Execute trades based on signals
            if signal_data['signal'] == 'BUY' and signal_data['strength'] >= 0.6:
                self.execute_buy_signal(balance, current_price, signal_data)
                
            elif signal_data['signal'] == 'SELL' and signal_data['strength'] >= 0.6:
                self.execute_sell_signal(balance, current_price, signal_data)
            
            # Reinvest profits into Bitcoin
            self.reinvest_profits(balance, current_price)
            
            # Update daily PnL
            self.update_daily_pnl(balance)
            
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
            self.send_alert(f"Trading cycle error: {e}", "ERROR")
    
    def execute_buy_signal(self, balance: Dict, current_price: float, signal_data: Dict) -> None:
        """Execute buy signal (USDT -> BTC)"""
        try:
            # Get available USDT balance
            usdt_balance = balance.get('USDT', {}).get('free', 0)
            position_size = (usdt_balance * self.calculate_position_size(balance, current_price)) / current_price
            
            if position_size == 0:
                logger.info("Position size too small, skipping buy")
                return
            
            logger.info(f"Executing BUY signal: {position_size} BTC at {current_price} USDT")
            
            # Place market buy order
            order = self.place_market_order(self.symbol, 'buy', position_size)
            
            if order:
                # Set stop loss and take profit
                self.set_stop_loss_take_profit(order, 'buy')
                
                # Log trade
                trade_data = {**order, 'signal': signal_data}
                self.log_trade(trade_data)
                
                # Send alert
                self.send_alert(f"BUY order executed: {position_size} {self.symbol} at {current_price}", "TRADE")
                
                self.trade_count += 1
                
        except Exception as e:
            logger.error(f"Error executing buy signal: {e}")
    
    def execute_sell_signal(self, balance: Dict, current_price: float, signal_data: Dict) -> None:
        """Execute sell signal (ETH -> BTC)"""
        try:
            # Get available ETH balance
            eth_balance = balance.get('ETH', {}).get('free', 0)
            
            if eth_balance < 0.001:  # Minimum ETH to sell
                logger.info("Insufficient ETH balance for sell signal")
                return
            
            # Sell portion of ETH holdings
            sell_amount = min(eth_balance * 0.5, eth_balance)  # Sell up to 50%
            
            logger.info(f"Executing SELL signal: {sell_amount} ETH")
            
            # Place market sell order
            order = self.place_market_order(self.symbol, 'sell', sell_amount)
            
            if order:
                # Set stop loss and take profit
                self.set_stop_loss_take_profit(order, 'sell')
                
                # Log trade
                trade_data = {**order, 'signal': signal_data}
                self.log_trade(trade_data)
                
                # Send alert
                self.send_alert(f"SELL order executed: {sell_amount} ETH at {current_price}", "TRADE")
                
                self.trade_count += 1
                
        except Exception as e:
            logger.error(f"Error executing sell signal: {e}")
    
    def reinvest_profits(self, balance: Dict, current_price: float) -> None:
        """Reinvest ETH profits back into Bitcoin"""
        try:
            eth_balance = balance.get('ETH', {}).get('free', 0)
            
            # If we have significant ETH profits, convert some back to BTC
            if eth_balance > 0.01:  # More than 0.01 ETH
                # Convert 30% of ETH back to BTC to maintain BTC accumulation
                convert_amount = eth_balance * 0.3
                
                logger.info(f"Reinvesting {convert_amount} ETH back to BTC")
                
                order = self.place_market_order(self.symbol, 'sell', convert_amount)
                
                if order:
                    self.send_alert(f"Profit reinvestment: {convert_amount} ETH -> BTC", "REINVEST")
                    
        except Exception as e:
            logger.error(f"Error in profit reinvestment: {e}")
    
    def update_daily_pnl(self, balance: Dict) -> None:
        """Update daily profit/loss tracking"""
        try:
            # This is a simplified PnL calculation
            # In practice, you'd track entry prices and calculate unrealized PnL
            btc_balance = balance.get('BTC', {}).get('free', 0)
            eth_balance = balance.get('ETH', {}).get('free', 0)
            
            # Store current portfolio value for PnL tracking
            # Implementation depends on your specific needs
            
        except Exception as e:
            logger.error(f"Error updating daily PnL: {e}")
    
    # ==================== BOT CONTROL ====================
    
    def start_trading(self, interval: int = 60) -> None:
        """Start the automated trading bot"""
        logger.info("🚀 Starting Binance Crypto Trading Bot...")
        
        # Send startup alert
        self.send_alert("Trading bot started successfully!", "STARTUP")
        
        self.running = True
        
        try:
            while self.running:
                self.execute_trading_cycle()
                
                logger.info(f"Cycle complete. Waiting {interval} seconds...")
                time.sleep(interval)
                
        except KeyboardInterrupt:
            logger.info("Received interrupt signal...")
            self.stop_trading()
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}")
            self.send_alert(f"Bot crashed: {e}", "ERROR")
            self.stop_trading()
    
    def stop_trading(self) -> None:
        """Stop the trading bot safely"""
        logger.info("🛑 Stopping trading bot...")
        self.running = False
        
        # Cancel all open orders
        try:
            open_orders = self.get_open_orders(self.symbol)
            for order in open_orders:
                self.cancel_order(order['id'], self.symbol)
                
            logger.info("All open orders cancelled")
            
        except Exception as e:
            logger.error(f"Error cancelling orders: {e}")
        
        # Final balance report
        try:
            final_balance = self.fetch_balance()
            if final_balance:
                btc_balance = final_balance.get('BTC', {}).get('free', 0)
                eth_balance = final_balance.get('ETH', {}).get('free', 0)
                
                logger.info(f"Final Balance - BTC: {btc_balance}, ETH: {eth_balance}")
                self.send_alert(f"Bot stopped. Final Balance - BTC: {btc_balance}, ETH: {eth_balance}", "SHUTDOWN")
                
        except Exception as e:
            logger.error(f"Error getting final balance: {e}")

# ==================== CONFIGURATION AND MAIN ====================

def main():
    """Main function to run the trading bot"""
    
    print("""
    🚀 BINANCE CRYPTO TRADING BOT
    ============================
    
    ⚠️  IMPORTANT SETUP STEPS:
    
    1. Create Binance API Keys:
       - Go to Binance.com -> Account -> API Management
       - Create new API key with spot trading permissions
       - Enable "Enable Reading" and "Enable Spot & Margin Trading"
       - Add IP whitelist for security
    
    2. Test Environment:
       - Start with testnet: testnet=True
       - Test with small amounts first
    
    3. Configure Parameters:
       - Adjust risk parameters below
       - Set up alert notifications
    
    ⚠️  SECURITY WARNING:
    - Never share your API keys
    - Use environment variables for production
    - Enable IP whitelist and 2FA
    """)
    
    # Configuration - Binance API credentials
    API_KEY = "MUgkepXLXTMUGrsCNpYiYW6WilOkA1HwRmlwv23laM8p9dAFap4uVpYcPpKhifn6"
    API_SECRET = "B7XRKedZ5BWl120FFb9SI2EQYbFbVlYCrw5pG5xLXuTIOV9XlG0OYmjKjJlCmMpR"
    
    # Use testnet for testing (set to False for live trading)
    USE_TESTNET = True
    
    try:
        # Initialize the trading bot
        bot = BinanceCryptoBot(API_KEY, API_SECRET, testnet=USE_TESTNET)
        
        # Configure trading parameters
        bot.min_profit_threshold = 0.015  # 1.5% minimum profit
        bot.max_position_size = 0.05      # 5% max position size
        bot.stop_loss_pct = 0.02          # 2% stop loss
        bot.take_profit_pct = 0.03        # 3% take profit
        bot.max_daily_loss = 0.05         # 5% max daily loss
        
        # Start trading (runs every 60 seconds)
        bot.start_trading(interval=60)
        
    except Exception as e:
        logger.error(f"Failed to start bot: {e}")
        print(f"❌ Bot failed to start: {e}")

if __name__ == "__main__":
    main()