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
    
    def rain_forest_resin_trade(self, state: TradingState, std_bid: float) -> List[Order]:
        ASSET = "RAINFOREST_RESIN"
        NB_OBS = 1000
        Z_SCORE = 0.5  # Multiplier for standard deviation
        
        # Fixed target mean around which to revert.
        TARGET_PRICE = 10000
        
        # Here, instead of computing std from recent trades,
        # we use the std_bid computed from historical first bids.
        buy_threshold = TARGET_PRICE - Z_SCORE * std_bid
        sell_threshold = TARGET_PRICE + Z_SCORE * std_bid
        
        #### ADJUST POSITION
        order_depth: OrderDepth = state.order_depths[ASSET]
        orders: List[Order] = []
        
        current_position = state.position.get(ASSET, 0)
        allowed_buy = self.position_limit[ASSET] - current_position
        allowed_sell = current_position if current_position > 0 else 0
        
        # Process sell orders as potential buys.
        for price in sorted(order_depth.sell_orders.keys()):
            if price <= buy_threshold:
                if allowed_buy <= 0:
                    break
                available_qty = -order_depth.sell_orders[price]  # Convert negative volume to positive.
                qty_to_buy = min(available_qty, allowed_buy)
                orders.append(Order(ASSET, int(round(price)), qty_to_buy))
                allowed_buy -= qty_to_buy
                print(f"BUYING: {ASSET} {qty_to_buy} at price: {price} ; Threshold: {buy_threshold}")
                
        # Process buy orders as potential sells.
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
        """
        At the beginning of each run:
          - Load persistent state from state.traderData.
          - Save the first bid (best bid) and its volume from RAINFOREST_RESIN's order book.
          - Compute the standard deviation (std_bid) based on the history of recorded first bids.
          - Execute strategies using std_bid for threshold calculation.
          - Save updated historical data back into traderData.
        """
        # Load persistent state.
        if state.traderData:
            try:
                historical = json.loads(state.traderData)
            except Exception:
                historical = {}
        else:
            historical = {}
        
        # Ensure a history container exists for RAINFOREST_RESIN.
        asset_hist = historical.setdefault("RAINFOREST_RESIN", {})
        bid_history = asset_hist.setdefault("first_bid_history", [])
        
        # Get order depth for RAINFOREST_RESIN.
        order_depth: OrderDepth = state.order_depths.get("RAINFOREST_RESIN")
        if order_depth and order_depth.buy_orders:
            # "First bid" is defined here as the best bid (highest bid price).
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_volume = order_depth.buy_orders[best_bid]
            # Record the first bid only once per run (using state.timestamp as a marker).
            bid_record = {"timestamp": state.timestamp, "bid": best_bid, "volume": best_bid_volume}
            bid_history.append(bid_record)
            # Optionally, limit the history to the last 100 records.
            bid_history = bid_history[-100:]
            asset_hist["first_bid_history"] = bid_history
            print("Recorded first bid:", bid_record)
        else:
            print("No bid orders available to record first bid.")
        
        # Compute standard deviation from historical first bid prices.
        if bid_history:
            bid_prices = [record["bid"] for record in bid_history]
            mean_bid = sum(bid_prices) / len(bid_prices)
            variance_bid = sum((bid - mean_bid) ** 2 for bid in bid_prices) / len(bid_prices)
            std_bid = math.sqrt(variance_bid)
            print("Computed historical std_bid:", std_bid)
        else:
            std_bid = 1.0  # default value if no history is available
            print("No bid history available. Using default std_bid:", std_bid)
        
        # Run trade strategies using the computed std_bid.
        orders_resin = self.rain_forest_resin_trade(state, std_bid)
        orders_kelp = self.kelp_trade(state)
        orders_squid = self.squid_ink_trade(state)
        
        result = {
            "RAINFOREST_RESIN": orders_resin,
            "KELP": orders_kelp,
            "SQUID_INK": orders_squid
        }
        
        # Save updated historical data.
        traderData = json.dumps(historical)
        conversions = 0
        
        print("Market Trades:", state.market_trades)
        return result, conversions, traderData
