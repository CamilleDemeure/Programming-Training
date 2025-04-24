from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict
import jsonpickle
import math

class Trader:
    def __init__(self):
        # Position limits
        self.POSITION_LIMIT = 100  # Assuming this is the limit for PICNIC_BASKET2
        
        # Coefficients from OLS model for bid prediction
        self.bid_const = 454.4374
        self.bid_coef_croissants = 0.0940
        self.bid_coef_djembes = -0.0353
        self.bid_coef_jams = 0.0448
        self.bid_coef_kelp = -0.0629
        self.bid_coef_sma_25 = 0.9819
        
        # Coefficients from OLS model for ask prediction
        self.ask_const = 433.4733
        self.ask_coef_croissants = 0.0924
        self.ask_coef_djembes = -0.0351
        self.ask_coef_jams = 0.0451
        self.ask_coef_kelp = -0.0558
        self.ask_coef_sma_25 = 0.9823
        
        # Price history for SMA calculation
        self.bid_history = []
        self.ask_history = []
        self.sma_window = 25
        
        # Previous prices for data continuity
        self.prev_prices = {}
        
        # Tracked products
        self.required_products = ["PICNIC_BASKET2", "CROISSANTS", "DJEMBES", "JAMS", "KELP"]
        
        # Optional: Confidence factor for order sizing (0 to 1)
        self.confidence = 0.8
        
        # Optional: Threshold for immediate action (mispricing threshold)
        self.mispricing_threshold = 2  # Consider mispriced if difference is ≥ 2

    def calculate_sma(self, history):
        """Calculate Simple Moving Average from price history"""
        if not history:
            return None
        return sum(history) / len(history)

    def run(self, state: TradingState):
        result = {}
        position = state.position.get("PICNIC_BASKET2", 0)
        
        # Store current prices
        current_prices = {}
        
        # Get current market prices
        for product in self.required_products:
            if product in state.order_depths:
                # For all products, get both bid and ask if available
                if state.order_depths[product].buy_orders:
                    current_prices[f"{product}_bid"] = max(state.order_depths[product].buy_orders.keys())
                else:
                    current_prices[f"{product}_bid"] = self.prev_prices.get(f"{product}_bid", None)
                    
                if state.order_depths[product].sell_orders:
                    current_prices[f"{product}_ask"] = min(state.order_depths[product].sell_orders.keys())
                else:
                    current_prices[f"{product}_ask"] = self.prev_prices.get(f"{product}_ask", None)
        
        # Update price history for PICNIC_BASKET2 if available
        if "PICNIC_BASKET2_bid" in current_prices and current_prices["PICNIC_BASKET2_bid"] is not None:
            self.bid_history.append(current_prices["PICNIC_BASKET2_bid"])
            if len(self.bid_history) > self.sma_window:
                self.bid_history = self.bid_history[-self.sma_window:]
                
        if "PICNIC_BASKET2_ask" in current_prices and current_prices["PICNIC_BASKET2_ask"] is not None:
            self.ask_history.append(current_prices["PICNIC_BASKET2_ask"])
            if len(self.ask_history) > self.sma_window:
                self.ask_history = self.ask_history[-self.sma_window:]
        
        # Calculate SMAs
        bid_sma = self.calculate_sma(self.bid_history)
        ask_sma = self.calculate_sma(self.ask_history)
        
        # Check if we have all required data
        required_fields = [
            "CROISSANTS_ask", "DJEMBES_ask", "JAMS_ask", "KELP_ask",
            "PICNIC_BASKET2_bid", "PICNIC_BASKET2_ask"
        ]
        
        missing = [field for field in required_fields if field not in current_prices or current_prices[field] is None]
        
        if not missing and bid_sma is not None and ask_sma is not None:
            # Current market prices for PICNIC_BASKET2
            current_bid = current_prices["PICNIC_BASKET2_bid"]
            current_ask = current_prices["PICNIC_BASKET2_ask"]
            
            # Calculate predicted next bid
            predicted_next_bid = math.floor(
                self.bid_const
                + self.bid_coef_croissants * current_prices["CROISSANTS_ask"]
                + self.bid_coef_djembes * current_prices["DJEMBES_ask"] 
                + self.bid_coef_jams * current_prices["JAMS_ask"]
                + self.bid_coef_kelp * current_prices["KELP_ask"]
                + self.bid_coef_sma_25 * bid_sma
            )
            
            # Calculate predicted next ask
            predicted_next_ask = math.floor(
                self.ask_const
                + self.ask_coef_croissants * current_prices["CROISSANTS_ask"]
                + self.ask_coef_djembes * current_prices["DJEMBES_ask"] 
                + self.ask_coef_jams * current_prices["JAMS_ask"]
                + self.ask_coef_kelp * current_prices["KELP_ask"]
                + self.ask_coef_sma_25 * ask_sma
            )
            
            # Print the values
            print(f"Timestamp: {state.timestamp}")
            print(f"Current PICNIC_BASKET2 Bid: {current_bid}, Ask: {current_ask}")
            print(f"SMA_25 (Bid): {bid_sma}, SMA_25 (Ask): {ask_sma}")
            print(f"Predicted Next Bid: {predicted_next_bid}, Ask: {predicted_next_ask}")
            
            # Trading logic
            orders = []
            available_to_buy = self.POSITION_LIMIT - position
            available_to_sell = self.POSITION_LIMIT + position
            
            # BUY logic: If market ask < predicted next ask, BUY immediately
            if current_ask < predicted_next_ask and available_to_buy > 0:
                # Calculate volume based on mispricing magnitude and available capacity
                price_diff = predicted_next_ask - current_ask
                if price_diff >= self.mispricing_threshold:
                    # More aggressive for larger mispricing
                    volume = min(
                        available_to_buy,
                        int(available_to_buy * self.confidence * (1 + price_diff/10))
                    )
                    # Ensure at least 1 unit traded
                    volume = max(1, min(volume, available_to_buy))
                    
                    orders.append(Order("PICNIC_BASKET2", current_ask, volume))
                    print(f"BUY {volume} @ {current_ask} (Market is underpriced)")
            
            # SELL logic: If market bid > predicted next bid, SELL immediately
            elif current_bid > predicted_next_bid and available_to_sell > 0:
                # Calculate volume based on mispricing magnitude and available capacity
                price_diff = current_bid - predicted_next_bid
                if price_diff >= self.mispricing_threshold:
                    # More aggressive for larger mispricing
                    volume = min(
                        available_to_sell,
                        int(available_to_sell * self.confidence * (1 + price_diff/10))
                    )
                    # Ensure at least 1 unit traded
                    volume = max(1, min(volume, available_to_sell))
                    
                    orders.append(Order("PICNIC_BASKET2", current_bid, -volume))
                    print(f"SELL {volume} @ {current_bid} (Market is overpriced)")
            
            # Market making logic
            else:
                # If market ask > predicted, make buy offers below current ask
                if current_ask > predicted_next_ask and available_to_buy > 0:
                    # Place buy order at predicted price
                    buy_price = predicted_next_ask
                    buy_volume = min(5, available_to_buy)  # Conservative volume for market making
                    
                    orders.append(Order("PICNIC_BASKET2", buy_price, buy_volume))
                    print(f"MAKE BUY {buy_volume} @ {buy_price} (Providing liquidity)")
                
                # If market bid < predicted, make sell offers above current bid
                if current_bid < predicted_next_bid and available_to_sell > 0:
                    # Place sell order at predicted price
                    sell_price = predicted_next_bid
                    sell_volume = min(5, available_to_sell)  # Conservative volume for market making
                    
                    orders.append(Order("PICNIC_BASKET2", sell_price, -sell_volume))
                    print(f"MAKE SELL {sell_volume} @ {sell_price} (Providing liquidity)")
            
            print("----------------------------")
            
            # Add orders to result
            if orders:
                result["PICNIC_BASKET2"] = orders
        else:
            if bid_sma is None:
                missing.append("SMA_25 (bid) - insufficient history")
            if ask_sma is None:
                missing.append("SMA_25 (ask) - insufficient history")
            print(f"Missing data: {missing}")
        
        # Update previous prices
        self.prev_prices = current_prices
        
        # Return orders, no conversions, and no trader data
        return result, 0, ""