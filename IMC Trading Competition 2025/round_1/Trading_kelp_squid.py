import json
import pandas as pd
import numpy as np
from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List

class Trader:
    SUBMISSION_ID = "your-submission-id"

    def __init__(self):
        self.lookback_period = 50
        self.threshold = 0.001
        self.position_limit = 50
        self.products = ["KELP", "SQUID_INK"]

    def run(self, state: TradingState):
        historical_data = self.load_historical_data(state.traderData)
        self.update_historical_data(historical_data, state)
        orders = self.generate_orders(state, historical_data)
        traderData = self.save_historical_data(historical_data)
        return orders, 0, traderData

    def load_historical_data(self, traderData: str) -> Dict[str, pd.DataFrame]:
        if traderData:
            try:
                historical_data = json.loads(traderData)
                for product in self.products:
                    if product in historical_data:
                        historical_data[product] = pd.read_json(historical_data[product])
                    else:
                        historical_data[product] = pd.DataFrame(columns=["mid_price"])
            except json.JSONDecodeError:
                historical_data = {p: pd.DataFrame(columns=["mid_price"]) for p in self.products}
        else:
            historical_data = {p: pd.DataFrame(columns=["mid_price"]) for p in self.products}
        return historical_data

    def save_historical_data(self, historical_data: Dict[str, pd.DataFrame]) -> str:
        data_to_save = {
            product: df.to_json() for product, df in historical_data.items()
        }
        return json.dumps(data_to_save)

    def update_historical_data(self, historical_data: Dict[str, pd.DataFrame], state: TradingState):
        for product in self.products:
            if product in state.order_depths:
                depth = state.order_depths[product]
                best_bid = max(depth.buy_orders.keys(), default=None)
                best_ask = min(depth.sell_orders.keys(), default=None)
                if best_bid is not None and best_ask is not None:
                    mid_price = (best_bid + best_ask) / 2.0
                    new_row = pd.DataFrame({"mid_price": [mid_price]})
                    historical_data[product] = pd.concat(
                        [historical_data[product], new_row], ignore_index=True
                    )
                    if len(historical_data[product]) > self.lookback_period:
                        historical_data[product] = historical_data[product].iloc[-self.lookback_period:]

    def compute_VAR1_forecast(self, df: pd.DataFrame) -> Dict[str, float]:
        # Convert mid-prices to returns
        returns = df.pct_change().dropna()
        if len(returns) < 2:
            return {col: 0.0 for col in df.columns}

        # Setup VAR(1): Y_t = A * Y_{t-1} + e
        Y = returns.values[1:]      # (n-1, k)
        Y_lag = returns.values[:-1] # (n-1, k)

        # Least squares estimation: A = (Y_lag.T @ Y_lag)^-1 @ Y_lag.T @ Y
        A = np.linalg.pinv(Y_lag) @ Y  # shape: (k, k)

        # Forecast next return: A.T @ last_return
        last_return = returns.values[-1]  # shape: (k,)
        forecast = A.T @ last_return      # shape: (k,)

        return {col: forecast[i] for i, col in enumerate(df.columns)}

    def generate_orders(self, state: TradingState, historical_data: Dict[str, pd.DataFrame]) -> Dict[str, List[Order]]:
        orders = {product: [] for product in self.products}

        # Make sure we have data for both products
        if all(len(historical_data[product]) >= self.lookback_period for product in self.products):
            # Build DataFrame with both series aligned
            aligned_data = {
                p: historical_data[p]["mid_price"].values for p in self.products
            }
            df = pd.DataFrame(aligned_data)

            # Predict next return using our own VAR(1) model
            predicted_returns = self.compute_VAR1_forecast(df)

            for product in self.products:
                prediction = predicted_returns.get(product, 0.0)
                current_position = state.position.get(product, 0)
                depth = state.order_depths.get(product)

                if prediction > self.threshold:
                    # BUY signal
                    best_ask = min(depth.sell_orders.keys(), default=None)
                    if best_ask is not None and current_position < self.position_limit:
                        volume = min(self.position_limit - current_position, depth.sell_orders[best_ask])
                        orders[product].append(Order(product, best_ask, volume))

                elif prediction < -self.threshold:
                    # SELL signal
                    best_bid = max(depth.buy_orders.keys(), default=None)
                    if best_bid is not None and current_position > -self.position_limit:
                        volume = min(current_position + self.position_limit, depth.buy_orders[best_bid])
                        orders[product].append(Order(product, best_bid, -volume))

        return orders
