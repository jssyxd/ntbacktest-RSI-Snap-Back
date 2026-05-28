from decimal import Decimal

from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.enums import OrderType
from nautilus_trader.model.enums import TimeInForce
from nautilus_trader.model.enums import TriggerType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Price
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.indicators import RelativeStrengthIndex
from nautilus_trader.indicators import Stochastics
from nautilus_trader.indicators import ExponentialMovingAverage


class RSISnapBackLongConfig(StrategyConfig):
    instrument_id: InstrumentId
    bar_type: BarType
    rsi_period: int = 14
    stoch_period_k: int = 14
    stoch_period_d: int = 3
    ema_filter_period: int = 200
    rsi_buy_threshold: float = 0.2
    stoch_buy_threshold: float = 25.0
    stop_loss_pct: float = 0.025
    take_profit_pct: float = 0.075
    leverage: int = 5
    trade_percent: Decimal = Decimal("10")


class RSISnapBackLong(Strategy):
    def __init__(self, config: RSISnapBackLongConfig):
        super().__init__(config)
        self.instrument = None
        self.entry_price: Price | None = None

    def on_start(self):
        self.instrument = self.cache.instrument(self.config.instrument_id)
        if self.instrument is None:
            self.log.error(f"Instrument {self.config.instrument_id} not found")
            self.stop()
            return

        self.rsi = RelativeStrengthIndex(self.config.rsi_period)
        self.stoch = Stochastics(
            self.config.stoch_period_k,
            self.config.stoch_period_d,
        )
        self.ema = ExponentialMovingAverage(self.config.ema_filter_period)

        self.register_indicator_for_bars(self.config.bar_type, self.rsi)
        self.register_indicator_for_bars(self.config.bar_type, self.stoch)
        self.register_indicator_for_bars(self.config.bar_type, self.ema)

        self.subscribe_bars(self.config.bar_type)

    def on_bar(self, bar: Bar):
        if not self.indicators_initialized():
            return

        rsi_value = self.rsi.value
        stoch_k = self.stoch.value_k
        ema_value = self.ema.value
        close = bar.close.as_double()

        in_position = self.portfolio.is_net_long(self.config.instrument_id)

        if not in_position:
            long_signal = (
                rsi_value < self.config.rsi_buy_threshold
                and stoch_k < self.config.stoch_buy_threshold
                and close > ema_value * 0.9
            )

            if long_signal:
                self.entry_price = close
                stop_price = close * (1 - self.config.stop_loss_pct)
                tp_price = close * (1 + self.config.take_profit_pct)

                qty = self.instrument.make_qty(
                    self.config.trade_percent * self.config.leverage / 100
                )

                reasoning = (
                    f"Long: RSI={rsi_value*100:.1f}, "
                    f"StochK={stoch_k:.1f}, "
                    f"EMA={ema_value:.2f}"
                )

                order_list = self.order_factory.bracket(
                    instrument_id=self.config.instrument_id,
                    order_side=OrderSide.BUY,
                    quantity=qty,
                    time_in_force=TimeInForce.GTC,
                    entry_order_type=OrderType.MARKET,
                    sl_trigger_price=self.instrument.make_price(stop_price),
                    tp_price=self.instrument.make_price(tp_price),
                    emulation_trigger=TriggerType.NO_TRIGGER,
                )
                self.submit_order_list(order_list)
                self.log.info(reasoning)

    def on_stop(self):
        self.cancel_all_orders(self.config.instrument_id)
        self.close_all_positions(self.config.instrument_id)
        self.unsubscribe_bars(self.config.bar_type)
