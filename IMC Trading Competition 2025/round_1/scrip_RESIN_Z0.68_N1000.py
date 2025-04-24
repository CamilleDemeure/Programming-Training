import json
import math
from datamodel import OrderDepth, TradingState, Order
from typing import List

class Trader:
    
    SUBMISSION_ID = "59f81e67-f6c6-4254-b61e-39661eac6141"
    
    def __init__(self):
        self.position_limit = {
            "RAINFOREST_RESIN": 50,
            "KELP": 50,
            "SQUID_INK": 50
        }
        
    def rain_forest_resin_trade(self, state: TradingState) -> List[Order]:
        ASSET = "RAINFOREST_RESIN"
        NB_OBS = 1000
        Z_SCORE = 1.3  # Multiplier for standard deviation
        
        # Fixed target mean around which to revert.
        TARGET_PRICE = 10000
        
        #### GET THE DATA
        # Retrieve market trades for the specified asset.
        trades = state.market_trades.get(ASSET, [])
        # Sort trades by timestamp (latest first) and take the last NB_OBS trades.
        trades.sort(key=lambda t: t.timestamp, reverse=True)
        last_trades = trades[:NB_OBS]
        
        if not last_trades:
            return []  # No trades available
        
        #### COMPUTE THE SIGNALS
        # Calculate the mean trade price (for standard deviation calculation).
        mean_price = sum(trade.price for trade in last_trades) / len(last_trades)
        variance = sum((trade.price - mean_price) ** 2 for trade in last_trades) / len(last_trades)
        std_dev = math.sqrt(variance)
        
        # Set buy and sell thresholds around the fixed TARGET_PRICE.
        buy_threshold = TARGET_PRICE - Z_SCORE * std_dev
        sell_threshold = TARGET_PRICE + Z_SCORE * std_dev
        
        #### ADJUST POSITION
        order_depth: OrderDepth = state.order_depths[ASSET]
        orders: List[Order] = []
        
        current_position = state.position.get(ASSET, 0)
        allowed_buy = self.position_limit[ASSET] - current_position
        # For a long-only strategy, you can only sell what you hold.
        allowed_sell = current_position if current_position > 0 else 0
        
        # LOOK FOR WHAT TO BUY (from sell orders)
        for price in sorted(order_depth.sell_orders.keys()):
            if price <= buy_threshold:
                if allowed_buy <= 0:
                    break
                available_qty = -order_depth.sell_orders[price]  # Convert negative volume to positive.
                qty_to_buy = min(available_qty, allowed_buy)
                orders.append(Order(ASSET, int(round(price)), qty_to_buy))
                allowed_buy -= qty_to_buy
                print(f"BUYING: {ASSET} {qty_to_buy} at price: {price} ; Threshold: {buy_threshold}")
                
        # LOOK FOR WHAT TO SELL (from buy orders)
        for price in sorted(order_depth.buy_orders.keys(), reverse=True):
            if price >= sell_threshold:
                if allowed_sell <= 0:
                    break
                available_qty = order_depth.buy_orders[price]
                qty_to_sell = min(available_qty, allowed_sell)
                orders.append(Order(ASSET, int(round(price)), -qty_to_sell))
                allowed_sell -= qty_to_sell
                print(f"SELLING: {ASSET} {qty_to_sell} at price: {price} ; Threshold: {sell_threshold}")
            
        return orders

    def kelp_trade(self, state: TradingState) -> List[Order]:
        # For simplicity, this strategy is not implemented.
        return []
    
    def squid_ink_trade(self, state: TradingState) -> List[Order]:
        # For simplicity, this strategy is not implemented.
        return []
    
    def run(self, state: TradingState):
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))
        
        orders_resin = self.rain_forest_resin_trade(state)
        orders_kelp = self.kelp_trade(state)
        orders_squid = self.squid_ink_trade(state)
        
        result = {
            "RAINFOREST_RESIN": orders_resin,
            "KELP": orders_kelp,
            "SQUID_INK": orders_squid
        }
        traderData = ""  # No persistent state used in this simple version.
        conversions = 0
        
        print(state.market_trades)
        return result, conversions, traderData
