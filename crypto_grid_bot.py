#!/usr/bin/env python3
"""
KryptoBotV1 — BTC/USDT Arithmetic Grid Trading Bot
Exchange: Binance | Pair: BTC/USDT
Running since: March 11, 2026
"""

import ccxt
import json
import time
import logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

STATE_FILE = 'grid_state.json'


class GridBot:
    """
    Arithmetic grid trading bot.

    How it works:
    - Divides a price range into N equally-spaced grid levels
    - Places BUY limit orders at every level below current price
    - Places SELL limit orders at every level above current price
    - When a BUY fills  → immediately places SELL at the next level up
    - When a SELL fills → records profit, places BUY at the next level down
    - Profit per cycle  = grid_gap / buy_price × per_grid_investment

    Best conditions: sideways/ranging markets with moderate volatility.
    """

    def __init__(self, config: dict):
        self.symbol            = config['symbol']
        self.lower             = config['lower_price']
        self.upper             = config['upper_price']
        self.num_grids         = config['num_grids']
        self.total_investment  = config['investment_usdt']

        self.exchange = ccxt.binance({
            'apiKey':          config['api_key'],
            'secret':          config['api_secret'],
            'enableRateLimit': True,
        })

        self.grid_levels   = self._calc_levels()
        self.grid_gap      = (self.upper - self.lower) / self.num_grids
        self.per_grid_usdt = self.total_investment / self.num_grids

        # State
        self.active_orders     : dict  = {}
        self.completed_cycles  : int   = 0
        self.total_profit      : float = 0.0
        self.trade_history     : list  = []

        self._load_state()
        log.info(
            f"GridBot ready | {self.lower:,.0f}–{self.upper:,.0f} "
            f"| {self.num_grids} grids @ ${self.grid_gap:,.0f} each"
        )

    # ──────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────

    def setup_grid(self) -> bool:
        """Place the initial grid of orders around the current market price."""
        price = self._ticker_price()
        log.info(f"Setting up grid. BTC/USDT = ${price:,.2f}")

        if not (self.lower < price < self.upper):
            log.error(
                f"Price ${price:,.0f} is outside the grid range "
                f"(${self.lower:,.0f}–${self.upper:,.0f}). "
                f"Adjust config.json and restart."
            )
            return False

        count = 0
        for i, level in enumerate(self.grid_levels[:-1]):
            next_level = self.grid_levels[i + 1]
            amount = self.per_grid_usdt / level

            if level < price:
                order = self.exchange.create_limit_buy_order(
                    self.symbol, round(amount, 6), level
                )
                self.active_orders[level] = {
                    'id': order['id'], 'side': 'buy', 'idx': i
                }
            elif level >= price:
                order = self.exchange.create_limit_sell_order(
                    self.symbol, round(amount, 6), next_level
                )
                self.active_orders[next_level] = {
                    'id': order['id'], 'side': 'sell', 'idx': i + 1
                }
            count += 1

        self._save_state()
        log.info(f"Grid ready — {count} orders placed")
        return True

    def poll(self):
        """
        Check every active order for fills.
        Called on each tick of the main loop.
        """
        for level, meta in list(self.active_orders.items()):
            try:
                order = self.exchange.fetch_order(meta['id'], self.symbol)
            except Exception as e:
                log.warning(f"Could not fetch order {meta['id']}: {e}")
                continue

            if order['status'] != 'closed':
                continue

            side       = meta['side']
            fill_price = order['price']
            fill_qty   = order['filled']
            idx        = meta['idx']

            if side == 'buy':
                self._on_buy_filled(level, fill_price, fill_qty, idx)
            else:
                self._on_sell_filled(level, fill_price, fill_qty, idx, meta)

        self._save_state()

    def run(self, interval: int = 30):
        """Main event loop. Press Ctrl-C to stop gracefully."""
        log.info("=== KryptoBotV1 running ===")
        try:
            while True:
                self.poll()
                s = self.status()
                log.info(
                    f"BTC ${s['price']:,.0f} | "
                    f"Orders: {s['active_orders']} | "
                    f"Cycles: {s['cycles']} | "
                    f"Profit: ${s['profit']:.2f}"
                )
                time.sleep(interval)
        except KeyboardInterrupt:
            log.info("Stopping — cancelling open orders…")
            self.cancel_all()
            log.info("Shutdown complete.")

    def cancel_all(self):
        """Cancel every active limit order."""
        cancelled = 0
        for meta in self.active_orders.values():
            try:
                self.exchange.cancel_order(meta['id'], self.symbol)
                cancelled += 1
            except Exception as e:
                log.warning(f"Cancel failed: {e}")
        self.active_orders.clear()
        self._save_state()
        log.info(f"Cancelled {cancelled} orders.")

    def status(self) -> dict:
        """Snapshot of current bot state."""
        bal   = self.exchange.fetch_balance()
        price = self._ticker_price()
        return {
            'timestamp':    datetime.now().isoformat(),
            'price':        price,
            'active_orders': len(self.active_orders),
            'cycles':       self.completed_cycles,
            'profit':       round(self.total_profit, 2),
            'btc':          bal['BTC']['free'],
            'usdt':         bal['USDT']['free'],
            'in_range':     self.lower <= price <= self.upper,
        }

    # ──────────────────────────────────────────────
    # Internal handlers
    # ──────────────────────────────────────────────

    def _on_buy_filled(self, level, fill_price, fill_qty, idx):
        """Buy filled — place sell at next grid level up."""
        del self.active_orders[level]
        if idx + 1 < len(self.grid_levels):
            next_level = self.grid_levels[idx + 1]
            order = self.exchange.create_limit_sell_order(
                self.symbol, fill_qty, next_level
            )
            self.active_orders[next_level] = {
                'id': order['id'], 'side': 'sell',
                'idx': idx + 1, 'buy_price': fill_price
            }
            log.info(f"  BUY  filled @ ${fill_price:,.0f} → SELL placed @ ${next_level:,.0f}")

    def _on_sell_filled(self, level, fill_price, fill_qty, idx, meta):
        """Sell filled — record profit, place buy at next grid level down."""
        buy_price = meta.get('buy_price', self.grid_levels[max(idx - 1, 0)])
        profit    = (fill_price - buy_price) * fill_qty
        self.total_profit      += profit
        self.completed_cycles  += 1

        self.trade_history.append({
            'ts':         datetime.now().isoformat(),
            'buy_price':  round(buy_price, 2),
            'sell_price': round(fill_price, 2),
            'qty':        round(fill_qty, 6),
            'profit':     round(profit, 4),
        })

        del self.active_orders[level]
        if idx - 1 >= 0:
            next_level = self.grid_levels[idx - 1]
            amount     = self.per_grid_usdt / next_level
            order = self.exchange.create_limit_buy_order(
                self.symbol, round(amount, 6), next_level
            )
            self.active_orders[next_level] = {
                'id': order['id'], 'side': 'buy', 'idx': idx - 1
            }

        log.info(
            f"  SELL filled @ ${fill_price:,.0f} | "
            f"Cycle profit: ${profit:.4f} | "
            f"Running total: ${self.total_profit:.2f}"
        )

    # ──────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────

    def _calc_levels(self) -> list:
        step = (self.upper - self.lower) / self.num_grids
        return [round(self.lower + i * step, 2) for i in range(self.num_grids + 1)]

    def _ticker_price(self) -> float:
        return self.exchange.fetch_ticker(self.symbol)['last']

    def _save_state(self):
        state = {
            'active_orders':    {str(k): v for k, v in self.active_orders.items()},
            'completed_cycles': self.completed_cycles,
            'total_profit':     self.total_profit,
            'trade_history':    self.trade_history[-1000:],
        }
        Path(STATE_FILE).write_text(json.dumps(state, indent=2))

    def _load_state(self):
        if not Path(STATE_FILE).exists():
            return
        try:
            state = json.loads(Path(STATE_FILE).read_text())
            self.active_orders    = {float(k): v for k, v in state.get('active_orders', {}).items()}
            self.completed_cycles = state.get('completed_cycles', 0)
            self.total_profit     = state.get('total_profit', 0.0)
            self.trade_history    = state.get('trade_history', [])
            log.info(
                f"State restored | Cycles: {self.completed_cycles} | "
                f"Profit: ${self.total_profit:.2f}"
            )
        except Exception as e:
            log.warning(f"Could not load state: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    config_path = Path('config.json')
    if not config_path.exists():
        log.error("config.json not found. Copy config.example.json and fill in your credentials.")
        raise SystemExit(1)

    config = json.loads(config_path.read_text())
    bot = GridBot(config)

    if not Path(STATE_FILE).exists():
        log.info("No saved state — initialising grid for the first time")
        if not bot.setup_grid():
            raise SystemExit(1)

    bot.run(interval=config.get('poll_interval', 30))


if __name__ == '__main__':
    main()
