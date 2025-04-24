import json
import math
from datamodel import (
    Listing,
    Observation,
    Order,
    OrderDepth,
    ProsperityEncoder,
    Symbol,
    Trade,
    TradingState,
)
from typing import List


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(
        self,
        state: TradingState,
        orders: dict[Symbol, list[Order]],
        conversions: int,
        trader_data: str,
    ) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(
                        state, self.truncate(state.traderData, max_item_length)
                    ),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(
        self, order_depths: dict[Symbol, OrderDepth]
    ) -> dict[Symbol, list]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."


logger = Logger()


class Trader:
    # Submission identifier (include this in any related communications)
    SUBMISSION_ID = "59f81e67-f6c6-4254-b61e-39661eac6141"

    def rain_forest_resin_strat(
        self, state: TradingState, historical: dict
    ) -> (List[Order], dict):
        """
        Mean-reversion strategy for RAINFOREST_RESIN.

        - Updates an EMA (mean) and variance for the midprice.
        - Computes:
              buy_threshold = mean - (z_threshold * std)
              sell_threshold = mean + (z_threshold * std)
        - Iterates over ask orders (lowest first) and buys if the ask price is below buy_threshold,
          until the allowed buy volume (position limit) is filled.
        - Iterates over bid orders (highest first) and sells if the bid price is above sell_threshold,
          until the allowed sell volume is filled.
        - Records each executed trade in a trade_history which is maintained only for the last 100 samples.
        - Position limit for RAINFOREST_RESIN is 50.

        Returns:
            A list of Order objects for RAINFOREST_RESIN and the updated historical state.
        """
        product = "RAINFOREST_RESIN"
        if product not in state.order_depths:
            return [], historical

        order_depth: OrderDepth = state.order_depths[product]
        orders: List[Order] = []

        # Compute midprice using best bid and ask.
        best_bid = (
            max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        )
        best_ask = (
            min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        )

        if best_bid is not None and best_ask is not None:
            midprice = (best_bid + best_ask) / 2.0
        elif best_bid is not None:
            midprice = best_bid
        elif best_ask is not None:
            midprice = best_ask
        else:
            return orders, historical

        # Strategy parameters.
        alpha = 0.2  # Smoothing factor for EMA and variance.
        z_threshold = 0.1  # Z-score threshold.
        position_limit = 50

        # Update or initialize EMA and variance.
        if product in historical:
            old_mean = historical[product]["mean"]
            old_var = historical[product]["var"]
            new_mean = (1 - alpha) * old_mean + alpha * midprice
            new_var = (1 - alpha) * old_var + alpha * ((midprice - old_mean) ** 2)
        else:
            new_mean = midprice
            new_var = 0.0

        historical[product] = {"mean": new_mean, "var": new_var}
        std = math.sqrt(new_var) if new_var > 0 else 0.0

        # Define thresholds.
        buy_threshold = new_mean - z_threshold * std
        sell_threshold = new_mean + z_threshold * std

        # Determine allowed volumes.
        current_position = state.position.get(product, 0)
        allowed_buy = (
            position_limit - current_position
        )  # Additional units allowed to buy.
        allowed_sell = (
            position_limit + current_position
        )  # Additional units allowed to sell.

        # Process ask orders for buying.
        for price in sorted(order_depth.sell_orders.keys()):
            if allowed_buy <= 0:
                break
            if price < buy_threshold:
                available_qty = -order_depth.sell_orders[
                    price
                ]  # Convert negative volume to positive.
                qty_to_buy = min(available_qty, allowed_buy)
                orders.append(Order(product, price, qty_to_buy))
                allowed_buy -= qty_to_buy
                # Record trade (profit estimation: (midprice - price) * qty).
                trade_record = {
                    "midprice": midprice,
                    "price": price,
                    "side": "buy",
                    "qty": qty_to_buy,
                    "profit": (midprice - price) * qty_to_buy,
                }
                historical.setdefault(product, {}).setdefault(
                    "trade_history", []
                ).append(trade_record)
            else:
                break

        # Process bid orders for selling.
        for price in sorted(order_depth.buy_orders.keys(), reverse=True):
            if allowed_sell <= 0:
                break
            if price > sell_threshold:
                available_qty = order_depth.buy_orders[price]
                qty_to_sell = min(available_qty, allowed_sell)
                orders.append(Order(product, price, -qty_to_sell))
                allowed_sell -= qty_to_sell
                # Record trade (profit estimation: (price - midprice) * qty).
                trade_record = {
                    "midprice": midprice,
                    "price": price,
                    "side": "sell",
                    "qty": qty_to_sell,
                    "profit": (price - midprice) * qty_to_sell,
                }
                historical.setdefault(product, {}).setdefault(
                    "trade_history", []
                ).append(trade_record)
            else:
                break

        # Limit the trade history to the last 100 samples.
        if "trade_history" in historical[product]:
            if len(historical[product]["trade_history"]) > 100:
                historical[product]["trade_history"] = historical[product][
                    "trade_history"
                ][-100:]

        return orders, historical

    def kelp_trend_following_regime(
        self, state: TradingState, historical: dict
    ) -> (List[Order], dict):
        """
        Trend-following strategy for KELP based on the slope of an EMA over the last 100 steps.

        - Computes midprice using best bid and ask.
        - Maintains an EMA (trend indicator) for KELP using a fixed alpha (0.3).
        - Stores the EMA values in a trend_history list (only the last 100 samples are kept).
        - Computes the slope as (current_EMA - oldest_EMA) / (N - 1), where N is the number of samples.
        - If the slope >= 0.25 (strong uptrend), generates a buy signal.
        - If the slope <= -0.25 (strong downtrend), generates a sell signal.
        - Orders are filled by iterating over the corresponding side of the order book until the allowed volume (position limit) is filled.
        - The position limit for KELP is assumed to be 50.

        Returns the list of orders for KELP and the updated historical state.
        """
        product = "KELP"
        if product not in state.order_depths:
            return [], historical

        order_depth: OrderDepth = state.order_depths[product]
        orders: List[Order] = []

        # Compute midprice.
        best_bid = (
            max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        )
        best_ask = (
            min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        )

        if best_bid is not None and best_ask is not None:
            midprice = (best_bid + best_ask) / 2.0
        elif best_bid is not None:
            midprice = best_bid
        elif best_ask is not None:
            midprice = best_ask
        else:
            return orders, historical

        # Parameters for trend following.
        alpha = 0.3  # Fixed smoothing factor for the EMA (trend indicator).
        slope_threshold = 0.25  # Minimum absolute slope to trigger a trade.
        position_limit = 50
        sample_KELP = 2000

        # Initialize or update the EMA (trend indicator) for KELP.
        if product in historical and "trend" in historical[product]:
            old_trend = historical[product]["trend"]
            new_trend = (1 - alpha) * old_trend + alpha * midprice
        else:
            new_trend = midprice

        historical.setdefault(product, {})["trend"] = new_trend

        # Maintain trend_history (last sample_KELP samples).
        trend_history = historical[product].get("trend_history", [])
        trend_history.append(new_trend)
        if len(trend_history) > sample_KELP:
            trend_history = trend_history[-sample_KELP:]
        historical[product]["trend_history"] = trend_history

        # Compute slope if we have at least 2 samples.
        slope = 0.0
        if len(trend_history) >= 2:
            slope = (trend_history[-1] - trend_history[0]) / (len(trend_history) - 1)

        # Determine signal based on slope.
        signal = None
        if slope >= slope_threshold:
            signal = "buy"
        elif slope <= -slope_threshold:
            signal = "sell"

        # Determine allowed volumes.
        current_position = state.position.get(product, 0)
        allowed_buy = position_limit - current_position  # Units allowed to buy.
        allowed_sell = position_limit + current_position  # Units allowed to sell.

        # Execute orders based on the signal.
        if signal == "buy":
            for price in sorted(order_depth.sell_orders.keys()):
                if allowed_buy <= 0:
                    break
                available_qty = -order_depth.sell_orders[price]
                qty_to_buy = min(available_qty, allowed_buy)
                orders.append(Order(product, price, qty_to_buy))
                allowed_buy -= qty_to_buy
        elif signal == "sell":
            for price in sorted(order_depth.buy_orders.keys(), reverse=True):
                if allowed_sell <= 0:
                    break
                available_qty = order_depth.buy_orders[price]
                qty_to_sell = min(available_qty, allowed_sell)
                orders.append(Order(product, price, -qty_to_sell))
                allowed_sell -= qty_to_sell

        return orders, historical

    def run(self, state: TradingState):
        """
        Main entry point for the trading algorithm.

        Loads persistent state, runs the RAINFOREST_RESIN mean-reversion strategy and
        the KELP trend-following regime (which uses the slope of the EMA over the last 100 samples),
        and returns the combined orders along with updated traderData.
        """
        # Load persistent state.
        if state.traderData:
            try:
                historical = json.loads(state.traderData)
            except Exception:
                historical = {}
        else:
            historical = {}

        orders_rainforest, historical = self.rain_forest_resin_strat(state, historical)
        orders_kelp, historical = self.kelp_trend_following_regime(state, historical)

        result = {"RAINFOREST_RESIN": orders_rainforest, "KELP": orders_kelp}
        traderData = json.dumps(historical)
        conversions = 0

        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData
