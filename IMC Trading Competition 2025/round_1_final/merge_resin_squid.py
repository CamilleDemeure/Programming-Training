from typing import Dict, List
from datamodel import Order, OrderDepth, TradingState, Trade, Symbol, Listing, Product

POSITION_LIMITS = {
    Product.RAINFOREST_RESIN: 50,
    Product.SQUID_INK: 50,
}

class Trader:
    def __init__(self):
        self.squid_fair_values: List[float] = []

    def run(self, state: TradingState):
        orders: Dict[Symbol, List[Order]] = {}

        # Run both strategies independently
        if Product.RAINFOREST_RESIN in state.order_depths:
            orders.update(self.run_resin_strategy(state))

        if Product.SQUID_INK in state.order_depths:
            orders.update(self.run_squid_strategy(state))

        return orders, {}, {}

    def run_resin_strategy(self, state: TradingState) -> Dict[Symbol, List[Order]]:
        orders: List[Order] = []
        symbol = Product.RAINFOREST_RESIN
        order_depth: OrderDepth = state.order_depths[symbol]
        position = state.position.get(symbol, 0)

        fair_value = 10000
        take_width = 100
        clear_width = 400
        disregard_edge = 600
        join_edge = 200

        best_ask = min(order_depth.sell_orders.keys(), default=None)
        best_bid = max(order_depth.buy_orders.keys(), default=None)

        # Opportunity taking (aggressive)
        if best_ask is not None and best_ask < fair_value - take_width and position < POSITION_LIMITS[symbol]:
            buy_volume = min(order_depth.sell_orders[best_ask], POSITION_LIMITS[symbol] - position)
            orders.append(Order(symbol, best_ask, buy_volume))

        if best_bid is not None and best_bid > fair_value + take_width and position > -POSITION_LIMITS[symbol]:
            sell_volume = min(-order_depth.buy_orders[best_bid], position + POSITION_LIMITS[symbol])
            orders.append(Order(symbol, best_bid, -sell_volume))

        # Position clearing (passive)
        if position > 0:
            ask_price = fair_value + clear_width
            orders.append(Order(symbol, ask_price, -position))

        elif position < 0:
            bid_price = fair_value - clear_width
            orders.append(Order(symbol, bid_price, -position))

        # Market making
        if position < POSITION_LIMITS[symbol]:
            bid_price = fair_value - join_edge
            bid_volume = POSITION_LIMITS[symbol] - position
            orders.append(Order(symbol, bid_price, bid_volume))

        if position > -POSITION_LIMITS[symbol]:
            ask_price = fair_value + join_edge
            ask_volume = position + POSITION_LIMITS[symbol]
            orders.append(Order(symbol, ask_price, -ask_volume))

        return {symbol: orders}

    def run_squid_strategy(self, state: TradingState) -> Dict[Symbol, List[Order]]:
        orders: List[Order] = []
        symbol = Product.SQUID_INK
        order_depth: OrderDepth = state.order_depths[symbol]
        position = state.position.get(symbol, 0)

        # Calculate mid-price and fair value
        best_ask = min(order_depth.sell_orders.keys(), default=None)
        best_bid = max(order_depth.buy_orders.keys(), default=None)
        fair_value = (best_ask + best_bid) / 2 if best_ask is not None and best_bid is not None else 10000

        self.squid_fair_values.append(fair_value)
        if len(self.squid_fair_values) > 2:
            self.squid_fair_values.pop(0)

        expected_return = 0
        if len(self.squid_fair_values) >= 2:
            expected_return = self.squid_fair_values[-1] - self.squid_fair_values[-2]

        buy_threshold = 1
        sell_threshold = -1

        # Prediction-based strategy
        if expected_return > buy_threshold and position < POSITION_LIMITS[symbol]:
            ask_price = min(order_depth.sell_orders.keys(), default=None)
            if ask_price is not None:
                buy_volume = min(order_depth.sell_orders[ask_price], POSITION_LIMITS[symbol] - position)
                orders.append(Order(symbol, ask_price, buy_volume))

        elif expected_return < sell_threshold and position > -POSITION_LIMITS[symbol]:
            bid_price = max(order_depth.buy_orders.keys(), default=None)
            if bid_price is not None:
                sell_volume = min(-order_depth.buy_orders[bid_price], position + POSITION_LIMITS[symbol])
                orders.append(Order(symbol, bid_price, -sell_volume))

        return {symbol: orders}
