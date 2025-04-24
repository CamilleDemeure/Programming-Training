from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List
import statistics
import jsonpickle
import math

class Trader:
    def __init__(self):
        # Initialize lists to store historical mid prices for each product.
        self.resin_prices = []
        self.kelp_prices = []
        self.squid_ink_prices = []
        
        # List to track timestamps across iterations (useful for time-based analyses).
        self.timestamps = []
        
        # Dictionary to store predicted adjustments from Squid Ink to Kelp (lead-lag relationship).
        self.squid_to_kelp_predictions = {}
        
        # Define window sizes for calculating moving averages.
        self.short_window = 10
        self.medium_window = 50
        self.long_window = 200
        
        # Maximum history length to store to avoid memory bloat.
        self.max_history = 5000
        
        # Initial fair value estimates for each product.
        self.fair_values = {
            "RAINFOREST_RESIN": 10000,
            "KELP": 2000,
            "SQUID_INK": 2000
        }
        
        # Initial volatility estimates for each product.
        self.volatility = {
            "RAINFOREST_RESIN": 1,
            "KELP": 10,
            "SQUID_INK": 50
        }
        
        # Hardcoded z_threshold values (derived from a normal distribution)
        # For example, if you want a 60% CI for RAINFOREST_RESIN and 75% CI for KELP and SQUID_INK.
        self.z_threshold = {
            "RAINFOREST_RESIN": 0.26,  # CI 60%
            "KELP": 0.67,            # CI 75%
            "SQUID_INK": 0.67        # CI 75%
        }
        
        # Counter to track the number of iterations.
        self.iteration = 0

    def get_risk_factor(self, product: str) -> float:
        """
        Computes a scaling factor based on product volatility.
        The factor decreases as volatility increases to reduce order sizes in turbulent conditions.
        The returned factor is clamped between 0.5 and 1.0.
        """
        baseline = 100  # Baseline used to scale volatility impact.
        factor = 1 - (self.volatility[product] / baseline)
        return max(0.5, min(1.0, factor))

    def apply_stop_loss(self, product: str, mid_price: float, position: int) -> List[Order]:
        """
        Implements a stop-loss mechanism:
          - For long positions: if the current mid_price falls below fair value minus a multiple of volatility,
            then a sell order is generated to exit the position.
          - For short positions: if the current mid_price rises above fair value plus a multiple of volatility,
            then a buy order is generated to cover the short.
        """
        stop_loss_multiplier = 2.0  # Multiplier to determine threshold from fair value.
        orders = []
        if position > 0 and mid_price < self.fair_values[product] - stop_loss_multiplier * self.volatility[product]:
            orders.append(Order(product, int(round(mid_price)), -position))
        elif position < 0 and mid_price > self.fair_values[product] + stop_loss_multiplier * self.volatility[product]:
            orders.append(Order(product, int(round(mid_price)), abs(position)))
        return orders

    def run(self, state: TradingState):
        """
        Main method called on each trading iteration.
        Processes each product, applies risk management and trading strategies, and updates state.
        """
        result = {}

        # Load previous state data if available.
        if state.traderData:
            try:
                saved_data = jsonpickle.decode(state.traderData)
                self.__dict__.update(saved_data)
            except Exception:
                pass

        # Record current timestamp and update iteration counter.
        self.timestamps.append(state.timestamp)
        self.iteration += 1

        # Process orders for each product.
        for product in state.order_depths:
            if product == "RAINFOREST_RESIN":
                result[product] = self.trade_rainforest_resin(state, product)
            elif product == "KELP":
                continue
                #result[product] = self.trade_kelp(state, product)
            elif product == "SQUID_INK":
                continue
                #result[product] = self.trade_squid_ink(state, product)
            else:
                result[product] = []
        
        # Update predictions based on the lead-lag relationship between Squid Ink and Kelp.
        self.update_lead_lag_predictions()
        
        # Trim historical data to prevent excessive memory use.
        self.trim_history()
        
        # Serialize state for the next iteration.
        traderData = jsonpickle.encode(self.__dict__)
        conversions = 0
        return result, conversions, traderData

    def trade_rainforest_resin(self, state: TradingState, product: str) -> List[Order]:
        """
        Mean reversion strategy for Rainforest Resin.
        Applies dynamic sizing, a stop-loss mechanism, and uses a hardcoded z_threshold 
        (derived from a normal distribution) to determine acceptable price deviations.
        """
        order_depth = state.order_depths[product]
        mid_price = self.get_mid_price(order_depth)
        if mid_price:
            self.resin_prices.append(mid_price)
        if len(self.resin_prices) < 5:
            return []
        
        # Calculate the mean and standard deviation of recent mid prices.
        mean_price = statistics.mean(self.resin_prices[-100:]) if len(self.resin_prices) >= 100 else statistics.mean(self.resin_prices)
        std_dev = statistics.stdev(self.resin_prices[-100:]) if len(self.resin_prices) >= 100 else statistics.stdev(self.resin_prices)
        
        # Update fair value and volatility estimates.
        self.fair_values[product] = mean_price
        self.volatility[product] = std_dev
        
        # Use hardcoded z_threshold for this product.
        z_threshold = self.z_threshold[product]
        acceptable_deviation = std_dev * z_threshold
        
        # Set acceptable buy and sell price thresholds.
        buy_price = mean_price - acceptable_deviation
        sell_price = mean_price + acceptable_deviation
        
        # Determine current position and available capacity.
        position = state.position.get(product, 0)
        position_limit = 50
        available_to_buy = position_limit - position
        available_to_sell = position_limit + position
        
        orders = []
        # Apply stop-loss mechanism.
        stop_loss_orders = self.apply_stop_loss(product, mid_price, position)
        if stop_loss_orders:
            return stop_loss_orders

        # Calculate risk factor for dynamic order sizing.
        risk_factor = self.get_risk_factor(product)

        # Process sell orders (from bots) for buying.
        if order_depth.sell_orders:
            sorted_sells = sorted(order_depth.sell_orders.items())
            for price, quantity in sorted_sells:
                if price <= buy_price and available_to_buy > 0:
                    base_qty = min(-quantity, available_to_buy)
                    quantity_to_buy = int(base_qty * risk_factor)
                    if quantity_to_buy > 0:
                        orders.append(Order(product, int(round(price)), quantity_to_buy))
                        available_to_buy -= quantity_to_buy

        # Process buy orders (from bots) for selling.
        if order_depth.buy_orders:
            sorted_buys = sorted(order_depth.buy_orders.items(), reverse=True)
            for price, quantity in sorted_buys:
                if price >= sell_price and available_to_sell > 0:
                    base_qty = min(quantity, available_to_sell)
                    quantity_to_sell = int(base_qty * risk_factor)
                    if quantity_to_sell > 0:
                        orders.append(Order(product, int(round(price)), -quantity_to_sell))
                        available_to_sell -= quantity_to_sell

        return orders

    def trade_squid_ink(self, state: TradingState, product: str) -> List[Order]:
        """
        Composite strategy for Squid Ink based on momentum, mean reversion,
        and intraday pattern recognition. Incorporates dynamic sizing, stop-loss,
        and uses a hardcoded z_threshold (from a normal distribution) for setting acceptable price ranges.
        """
        order_depth = state.order_depths[product]
        mid_price = self.get_mid_price(order_depth)
        if mid_price:
            self.squid_ink_prices.append(mid_price)
        if len(self.squid_ink_prices) < 10:
            return []
        
        # Calculate moving averages.
        short_ma = self.calculate_sma(self.squid_ink_prices, self.short_window)
        medium_ma = self.calculate_sma(self.squid_ink_prices, self.medium_window)
        
        # Retrieve current position and available capacity.
        position = state.position.get(product, 0)
        position_limit = 50
        available_to_buy = position_limit - position
        available_to_sell = position_limit + position
        
        orders = []
        # Apply stop-loss mechanism.
        stop_loss_orders = self.apply_stop_loss(product, mid_price, position)
        if stop_loss_orders:
            return stop_loss_orders

        # Calculate short-term momentum.
        price_momentum = 0
        if len(self.squid_ink_prices) >= 5:
            recent_change = (self.squid_ink_prices[-1] / self.squid_ink_prices[-5] - 1) * 100
            price_momentum = recent_change * 3

        # Calculate mean reversion component.
        reversion_component = 0
        if medium_ma > 0:
            reversion_gap = (mid_price / medium_ma - 1) * 100
            reversion_component = -reversion_gap * 2
        
        # Determine intraday pattern based on the quarter of a 30000-tick day.
        day_pattern = 0
        if self.timestamps:
            day_ticks = 30000
            current_tick = self.timestamps[-1] % day_ticks
            day_quarter = current_tick / (day_ticks / 4)
            if day_quarter < 1:
                day_pattern = -8
            elif day_quarter < 2:
                day_pattern = 0
            elif day_quarter < 3:
                day_pattern = 10
            else:
                day_pattern = 5
        
        # Combine signals.
        signal_strength = price_momentum * 0.4 + reversion_component * 0.3 + day_pattern * 0.3
        
        # Update fair value based on combined signal.
        if mid_price:
            self.fair_values[product] = mid_price * (1 + signal_strength / 100)
        fair_value = self.fair_values[product]
        
        # Use hardcoded z_threshold for this product.
        z_threshold = self.z_threshold[product]
        acceptable_deviation = self.volatility[product] * z_threshold
        
        buy_price = fair_value - acceptable_deviation
        sell_price = fair_value + acceptable_deviation

        # Compute risk factor for dynamic order sizing.
        risk_factor = self.get_risk_factor(product)
        
        # Process sell orders for buying.
        if order_depth.sell_orders and available_to_buy > 0:
            sorted_sells = sorted(order_depth.sell_orders.items())
            for price, quantity in sorted_sells:
                if price <= buy_price:
                    base_qty = min(-quantity, available_to_buy)
                    quantity_to_buy = int(base_qty * risk_factor)
                    if quantity_to_buy > 0:
                        orders.append(Order(product, int(round(price)), quantity_to_buy))
                        available_to_buy -= quantity_to_buy

        # Process buy orders for selling.
        if order_depth.buy_orders and available_to_sell > 0:
            sorted_buys = sorted(order_depth.buy_orders.items(), reverse=True)
            for price, quantity in sorted_buys:
                if price >= sell_price:
                    base_qty = min(quantity, available_to_sell)
                    quantity_to_sell = int(base_qty * risk_factor)
                    if quantity_to_sell > 0:
                        orders.append(Order(product, int(round(price)), -quantity_to_sell))
                        available_to_sell -= quantity_to_sell

        return orders

    def trade_kelp(self, state: TradingState, product: str) -> List[Order]:
        """
        Hybrid strategy for Kelp combining medium-term mean reversion, trend analysis,
        and predictions from the Squid Ink correlation.
        Uses dynamic sizing, stop-loss, and a hardcoded z_threshold (from a normal distribution)
        to set acceptable price ranges.
        """
        order_depth = state.order_depths[product]
        mid_price = self.get_mid_price(order_depth)
        if mid_price:
            self.kelp_prices.append(mid_price)
        if len(self.kelp_prices) < 10:
            return []
        
        # Retrieve current position and available capacity.
        position = state.position.get(product, 0)
        position_limit = 50
        available_to_buy = position_limit - position
        available_to_sell = position_limit + position
        
        orders = []
        # Apply stop-loss mechanism.
        stop_loss_orders = self.apply_stop_loss(product, mid_price, position)
        if stop_loss_orders:
            return stop_loss_orders

        # Calculate moving averages.
        short_ma = self.calculate_sma(self.kelp_prices, self.short_window)
        medium_ma = self.calculate_sma(self.kelp_prices, self.medium_window)
        base_fair_value = medium_ma
        
        # Compute trend component.
        trend_component = 0
        if len(self.kelp_prices) >= self.medium_window * 2:
            old_medium_ma = self.calculate_sma(self.kelp_prices[:-self.medium_window], self.medium_window)
            if old_medium_ma > 0:
                trend_pct = (medium_ma / old_medium_ma - 1) * 100
                trend_component = trend_pct * 5
        
        # Incorporate Squid Ink prediction component.
        squid_ink_component = 0
        current_timestamp = self.timestamps[-1] if self.timestamps else 0
        for prediction_time in sorted(self.squid_to_kelp_predictions.keys(), key=lambda x: int(x)):
            if int(prediction_time) > current_timestamp:
                squid_ink_component = self.squid_to_kelp_predictions[prediction_time]
                break
        
        # Update fair value for Kelp.
        if base_fair_value > 0:
            self.fair_values[product] = base_fair_value * (1 + (trend_component + squid_ink_component) / 100)
        fair_value = self.fair_values[product]
        
        # Use hardcoded z_threshold for Kelp.
        z_threshold = self.z_threshold[product]
        acceptable_deviation = self.volatility[product] * z_threshold
        
        buy_price = fair_value - acceptable_deviation
        sell_price = fair_value + acceptable_deviation

        # Apply risk factor for dynamic order sizing.
        risk_factor = self.get_risk_factor(product)
        
        # Process sell orders for buying Kelp.
        if order_depth.sell_orders and available_to_buy > 0:
            sorted_sells = sorted(order_depth.sell_orders.items())
            for price, quantity in sorted_sells:
                if price <= buy_price:
                    base_qty = min(-quantity, available_to_buy)
                    quantity_to_buy = int(base_qty * risk_factor)
                    if quantity_to_buy > 0:
                        orders.append(Order(product, int(round(price)), quantity_to_buy))
                        available_to_buy -= quantity_to_buy

        # Process buy orders for selling Kelp.
        if order_depth.buy_orders and available_to_sell > 0:
            sorted_buys = sorted(order_depth.buy_orders.items(), reverse=True)
            for price, quantity in sorted_buys:
                if price >= sell_price:
                    base_qty = min(quantity, available_to_sell)
                    quantity_to_sell = int(base_qty * risk_factor)
                    if quantity_to_sell > 0:
                        orders.append(Order(product, int(round(price)), -quantity_to_sell))
                        available_to_sell -= quantity_to_sell

        return orders

    def update_lead_lag_predictions(self):
        """
        Updates predictions for Kelp based on observed Squid Ink prices.
        Uses key lag values and predetermined prediction factors.
        """
        if len(self.squid_ink_prices) < 100 or len(self.timestamps) < 100:
            return

        # Key lag values where significant correlations were observed.
        key_lags = [3600, 5000, 6300, 8400, 10000]
        current_timestamp = self.timestamps[-1] if self.timestamps else 0
        latest_squid_price = self.squid_ink_prices[-1]
        
        for lag in key_lags:
            future_timestamp = current_timestamp + lag
            if lag == 3600:
                prediction_factor = 0.08  # Maximum positive correlation.
            elif lag == 6300:
                prediction_factor = -0.07  # Maximum negative correlation.
            elif lag == 10000:
                prediction_factor = -0.04  # Reversal point.
            else:
                prediction_factor = 0.02  # Default modest movement.
            squid_avg = statistics.mean(self.squid_ink_prices[-50:])
            deviation = (latest_squid_price / squid_avg - 1) * 100
            self.squid_to_kelp_predictions[future_timestamp] = deviation * prediction_factor

    def get_mid_price(self, order_depth: OrderDepth) -> float:
        """
        Calculates the mid price from the order book:
          - If both buy and sell orders are present, returns the average of the best bid and ask.
          - Otherwise, returns the best available price.
        """
        if order_depth.buy_orders and order_depth.sell_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            return (best_bid + best_ask) / 2
        elif order_depth.buy_orders:
            return max(order_depth.buy_orders.keys())
        elif order_depth.sell_orders:
            return min(order_depth.sell_orders.keys())
        return None

    def calculate_sma(self, prices: List[float], window: int) -> float:
        """
        Calculates the simple moving average (SMA) for a list of prices using the specified window.
        If there are not enough prices, returns the average of all available prices.
        """
        if len(prices) < window:
            return sum(prices) / len(prices)
        return sum(prices[-window:]) / window

    def trim_history(self):
        """
        Trims historical data to avoid excessive memory usage.
        Also removes outdated predictions based on the current timestamp.
        """
        if len(self.resin_prices) > self.max_history:
            self.resin_prices = self.resin_prices[-self.max_history:]
        if len(self.kelp_prices) > self.max_history:
            self.kelp_prices = self.kelp_prices[-self.max_history:]
        if len(self.squid_ink_prices) > self.max_history:
            self.squid_ink_prices = self.squid_ink_prices[-self.max_history:]
        if len(self.timestamps) > self.max_history:
            self.timestamps = self.timestamps[-self.max_history:]
        
        current_timestamp = self.timestamps[-1] if self.timestamps else 0
        keys_to_remove = [ts for ts in self.squid_to_kelp_predictions if int(ts) < current_timestamp]
        for key in keys_to_remove:
            del self.squid_to_kelp_predictions[key]
