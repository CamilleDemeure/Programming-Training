import json
import numpy as np
from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List

class Trader:
    SUBMISSION_ID = "59f81e67-f6c6-4254-b61e-39661eac6141"

    def __init__(self):
        self.lookback_period = 50
        self.position_limit = 50
        self.products = ["KELP", "SQUID_INK"]
        self.beta = -0.09494  # From Engle-Granger regression
        self.entry_threshold = 1.0
        self.exit_threshold = 0.3

    def run(self, state: TradingState):
        historical_data = self.load_historical_data(state.traderData)
        self.update_historical_data(historical_data, state)
        orders = self.generate_orders(state, historical_data)
        traderData = self.save_historical_data(historical_data)
        return orders, 0, traderData

    def load_historical_data(self, traderData: str) -> Dict[str, List[float]]:
        if traderData:
            try:
                historical_data = json.loads(traderData)
                for product in self.products:
                    if product not in historical_data:
                        historical_data[product] = []
            except json.JSONDecodeError:
                historical_data = {p: [] for p in self.products}
        else:
            historical_data = {p: [] for p in self.products}
        return historical_data

    def save_historical_data(self, historical_data: Dict[str, List[float]]) -> str:
        return json.dumps(historical_data)

    def update_historical_data(self, historical_data: Dict[str, List[float]], state: TradingState):
        for product in self.products:
            if product in state.order_depths:
                depth = state.order_depths[product]
                best_bid = max(depth.buy_orders.keys(), default=None)
                best_ask = min(depth.sell_orders.keys(), default=None)
                if best_bid is not None and best_ask is not None:
                    mid_price = (best_bid + best_ask) / 2.0
                    historical_data[product].append(mid_price)
                    if len(historical_data[product]) > self.lookback_period:
                        historical_data[product] = historical_data[product][-self.lookback_period:]

    def generate_orders(self, state: TradingState, historical_data: Dict[str, List[float]]) -> Dict[str, List[Order]]:
        orders = {product: [] for product in self.products}

        if all(len(historical_data[p]) >= self.lookback_period for p in self.products):
            kelp = np.array(historical_data["KELP"])
            squid = np.array(historical_data["SQUID_INK"])

            spread = kelp - self.beta * squid
            mean = np.mean(spread)
            std = np.std(spread)
            z_score = (spread[-1] - mean) / std if std > 0 else 0.0

            kelp_position = state.position.get("KELP", 0)
            squid_position = state.position.get("SQUID_INK", 0)
            kelp_depth = state.order_depths["KELP"]
            squid_depth = state.order_depths["SQUID_INK"]

            # Entry logic
            if z_score > self.entry_threshold:
                # Short KELP, Long SQUID
                kelp_bid = max(kelp_depth.buy_orders.keys(), default=None)
                squid_ask = min(squid_depth.sell_orders.keys(), default=None)

                if kelp_bid is not None:
                    kelp_volume = min(self.position_limit + kelp_position, kelp_depth.buy_orders[kelp_bid])
                    if kelp_volume > 0:
                        orders["KELP"].append(Order("KELP", kelp_bid, -kelp_volume))

                if squid_ask is not None:
                    squid_volume = min(self.position_limit - squid_position, squid_depth.sell_orders[squid_ask])
                    if squid_volume > 0:
                        orders["SQUID_INK"].append(Order("SQUID_INK", squid_ask, squid_volume))

            elif z_score < -self.entry_threshold:
                # Long KELP, Short SQUID
                kelp_ask = min(kelp_depth.sell_orders.keys(), default=None)
                squid_bid = max(squid_depth.buy_orders.keys(), default=None)

                if kelp_ask is not None:
                    kelp_volume = min(self.position_limit - kelp_position, kelp_depth.sell_orders[kelp_ask])
                    if kelp_volume > 0:
                        orders["KELP"].append(Order("KELP", kelp_ask, kelp_volume))

                if squid_bid is not None:
                    squid_volume = min(self.position_limit + squid_position, squid_depth.buy_orders[squid_bid])
                    if squid_volume > 0:
                        orders["SQUID_INK"].append(Order("SQUID_INK", squid_bid, -squid_volume))

            # Exit logic
            elif abs(z_score) < self.exit_threshold:
                # Close KELP position
                if kelp_position != 0:
                    if kelp_position > 0:
                        kelp_bid = max(kelp_depth.buy_orders.keys(), default=None)
                        if kelp_bid is not None:
                            volume = min(kelp_position, kelp_depth.buy_orders[kelp_bid])
                            orders["KELP"].append(Order("KELP", kelp_bid, -volume))
                    else:
                        kelp_ask = min(kelp_depth.sell_orders.keys(), default=None)
                        if kelp_ask is not None:
                            volume = min(-kelp_position, kelp_depth.sell_orders[kelp_ask])
                            orders["KELP"].append(Order("KELP", kelp_ask, volume))

                # Close SQUID position
                if squid_position != 0:
                    if squid_position > 0:
                        squid_bid = max(squid_depth.buy_orders.keys(), default=None)
                        if squid_bid is not None:
                            volume = min(squid_position, squid_depth.buy_orders[squid_bid])
                            orders["SQUID_INK"].append(Order("SQUID_INK", squid_bid, -volume))
                    else:
                        squid_ask = min(squid_depth.sell_orders.keys(), default=None)
                        if squid_ask is not None:
                            volume = min(-squid_position, squid_depth.sell_orders[squid_ask])
                            orders["SQUID_INK"].append(Order("SQUID_INK", squid_ask, volume))

        return orders
