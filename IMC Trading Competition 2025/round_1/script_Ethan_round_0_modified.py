import json
import math
from datamodel import OrderDepth, TradingState, Order
from typing import List
import numpy as np

class Trader:
    # Submission identifier (include this in any related communications)
    SUBMISSION_ID = "59f81e67-f6c6-4254-b61e-39661eac6141"

    def calculate_target_price(self, midprice, mean, variance, z_threshold):
        """
        Calculate the target price based on a normal distribution.
        """
        std = math.sqrt(variance) if variance > 0 else 0.0
        target_price = mean + z_threshold * std
        return target_price

    def calculate_probability(self, price, mean, variance):
        """
        Calculate the probability of an order being filled based on a normal distribution.
        """
        std = math.sqrt(variance) if variance > 0 else 0.0
        if std == 0:
            return 0.0
        z_score = (price - mean) / std
        probability = stats.norm.cdf(z_score)
        return probability

    def rain_forest_resin_strat(self, state: TradingState, historical: dict) -> (List[Order], dict):
        """
        Mean-reversion strategy for RAINFOREST_RESIN with dynamic target price and limit orders.
        """
        product = "RAINFOREST_RESIN"
        if product not in state.order_depths:
            return [], historical

        order_depth: OrderDepth = state.order_depths[product]
        orders: List[Order] = []

        # Compute midprice using best bid and ask.
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None

        if best_bid is not None and best_ask is not None:
            midprice = (best_bid + best_ask) / 2.0
        elif best_bid is not None:
            midprice = best_bid
        elif best_ask is not None:
            midprice = best_ask
        else:
            return orders, historical

        # Strategy parameters.
        alpha = 0.2         # Smoothing factor for EMA and variance.
        z_threshold = 0.1   # Z-score threshold.
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

        # Calculate target prices.
        buy_target_price = self.calculate_target_price(midprice, new_mean, new_var, -z_threshold)
        sell_target_price = self.calculate_target_price(midprice, new_mean, new_var, z_threshold)

        # Determine allowed volumes.
        current_position = state.position.get(product, 0)
        allowed_buy = position_limit - current_position      # Additional units allowed to buy.
        allowed_sell = position_limit + current_position     # Additional units allowed to sell.

        # Process ask orders for buying.
        for price in sorted(order_depth.sell_orders.keys()):
            if allowed_buy <= 0:
                break
            if price <= buy_target_price:
                available_qty = -order_depth.sell_orders[price]  # Convert negative volume to positive.
                qty_to_buy = min(available_qty, allowed_buy)
                orders.append(Order(product, price, qty_to_buy))
                allowed_buy -= qty_to_buy
                # Record trade (profit estimation: (midprice - price) * qty).
                trade_record = {
                    "midprice": midprice,
                    "price": price,
                    "side": "buy",
                    "qty": qty_to_buy,
                    "profit": (midprice - price) * qty_to_buy
                }
                historical.setdefault(product, {}).setdefault("trade_history", []).append(trade_record)
            else:
                break

        # Process bid orders for selling.
        for price in sorted(order_depth.buy_orders.keys(), reverse=True):
            if allowed_sell <= 0:
                break
            if price >= sell_target_price:
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
                    "profit": (price - midprice) * qty_to_sell
                }
                historical.setdefault(product, {}).setdefault("trade_history", []).append(trade_record)
            else:
                break

        # Limit the trade history to the last 100 samples.
        if "trade_history" in historical[product]:
            if len(historical[product]["trade_history"]) > 100:
                historical[product]["trade_history"] = historical[product]["trade_history"][-100:]

        return orders, historical

    def kelp_trend_following_regime(self, state: TradingState, historical: dict) -> (List[Order], dict):
        """
        Trend-following strategy for KELP with dynamic target price and limit orders.
        """
        product = "KELP"
        if product not in state.order_depths:
            return [], historical

        order_depth: OrderDepth = state.order_depths[product]
        orders: List[Order] = []

        # Compute midprice.
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None

        if best_bid is not None and best_ask is not None:
            midprice = (best_bid + best_ask) / 2.0
        elif best_bid is not None:
            midprice = best_bid
        elif best_ask is not None:
            midprice = best_ask
        else:
            return orders, historical

        # Parameters for trend following.
        alpha = 0.3           # Fixed smoothing factor for the EMA (trend indicator).
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
        allowed_buy = position_limit - current_position   # Units allowed to buy.
        allowed_sell = position_limit + current_position   # Units allowed to sell.

        # Calculate target prices.
        buy_target_price = self.calculate_target_price(midprice, new_trend, np.var(trend_history), -slope_threshold)
        sell_target_price = self.calculate_target_price(midprice, new_trend, np.var(trend_history), slope_threshold)

        # Execute orders based on the signal.
        if signal == "buy":
            for price in sorted(order_depth.sell_orders.keys()):
                if allowed_buy <= 0:
                    break
                if price <= buy_target_price:
                    available_qty = -order_depth.sell_orders[price]
                    qty_to_buy = min(available_qty, allowed_buy)
                    orders.append(Order(product, price, qty_to_buy))
                    allowed_buy -= qty_to_buy
        elif signal == "sell":
            for price in sorted(order_depth.buy_orders.keys(), reverse=True):
                if allowed_sell <= 0:
                    break
                if price >= sell_target_price:
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

        result = {
            "RAINFOREST_RESIN": orders_rainforest,
            "KELP": orders_kelp
        }
        traderData = json.dumps(historical)
        conversions = 0

        print("Submission ID:", Trader.SUBMISSION_ID)
        return result, conversions, traderData
