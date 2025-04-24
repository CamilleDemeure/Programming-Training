from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict, Deque
import string
import jsonpickle
import numpy as np
import math
from collections import deque

class Product:
    KELP = "KELP"

PARAMS = {
    Product.KELP: {
        "vwap_window": 25,           # Number of periods for VWAP calculation
        "bid_offset": 0.2716,         # Offset to add to bid VWAP for predicting next bid
        "ask_offset": -0.2587,        # Offset to add to ask VWAP for predicting next ask
        "take_threshold": 0.1,        # Minimum edge required to take existing orders
        "make_threshold": 0.05,       # Minimum edge required for market making
        "position_scaling": 0.8,      # Factor to scale orders based on position
        "clear_threshold": 0.02,      # Threshold to clear positions
        "safety_margin": 0.03,        # Added safety margin for predictions
        "max_position_pct": 0.8,      # Maximum position as percentage of limit
    }
}

class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params
        
        # Position limit for KELP is 50
        self.LIMIT = {Product.KELP: 50}
    
    def calculate_vwap(self, prices, volumes):
        """Calculate Volume Weighted Average Price"""
        if not prices or not volumes or len(prices) != len(volumes) or sum(volumes) == 0:
            return None
        
        return sum(p * v for p, v in zip(prices, volumes)) / sum(volumes)
    
    def predict_next_prices(self, bid_vwap, ask_vwap, params):
        """Predict next bid and ask based on separate VWAPs and offsets"""
        next_bid_prediction = bid_vwap + params["bid_offset"]
        next_ask_prediction = ask_vwap + params["ask_offset"]
        
        return next_bid_prediction, next_ask_prediction
    
    def fair_value_estimation(self, bid_vwap, ask_vwap, next_bid, next_ask):
        """Estimate fair value based on VWAPs and predicted next prices"""
        # Average of mid VWAPs and mid predicted prices
        vwap_mid = (bid_vwap + ask_vwap) / 2
        predicted_mid = (next_bid + next_ask) / 2
        fair_value = (vwap_mid + predicted_mid) / 2
        
        return fair_value
    
    def update_price_history(self, traderObject, product, state):
        """Update separate price histories for best bid and best ask only"""
        # Initialize history if it doesn't exist
        if f"{product}_bid_price_history" not in traderObject:
            traderObject[f"{product}_bid_price_history"] = []
            traderObject[f"{product}_bid_volume_history"] = []
            traderObject[f"{product}_ask_price_history"] = []
            traderObject[f"{product}_ask_volume_history"] = []
            traderObject[f"{product}_bid_vwap_history"] = []
            traderObject[f"{product}_ask_vwap_history"] = []
            traderObject[f"{product}_fair_values"] = []
        
        order_depth = state.order_depths.get(product)
        if order_depth:
            # Update bid history from the best bid only
            if order_depth.buy_orders:
                best_bid = max(order_depth.buy_orders.keys())
                best_bid_volume = order_depth.buy_orders[best_bid]
                traderObject[f"{product}_bid_price_history"].append(best_bid)
                traderObject[f"{product}_bid_volume_history"].append(best_bid_volume)
            
            # Update ask history from the best ask only
            if order_depth.sell_orders:
                best_ask = min(order_depth.sell_orders.keys())
                best_ask_volume = abs(order_depth.sell_orders[best_ask])
                traderObject[f"{product}_ask_price_history"].append(best_ask)
                traderObject[f"{product}_ask_volume_history"].append(best_ask_volume)
        
        # Limit history to window size
        window_size = self.params[product]["vwap_window"]
        if len(traderObject[f"{product}_bid_price_history"]) > window_size:
            traderObject[f"{product}_bid_price_history"] = traderObject[f"{product}_bid_price_history"][-window_size:]
            traderObject[f"{product}_bid_volume_history"] = traderObject[f"{product}_bid_volume_history"][-window_size:]
        
        if len(traderObject[f"{product}_ask_price_history"]) > window_size:
            traderObject[f"{product}_ask_price_history"] = traderObject[f"{product}_ask_price_history"][-window_size:]
            traderObject[f"{product}_ask_volume_history"] = traderObject[f"{product}_ask_volume_history"][-window_size:]
        
        # Calculate separate VWAPs for bid and ask if we have enough data
        bid_vwap = None
        ask_vwap = None
        
        if len(traderObject[f"{product}_bid_price_history"]) > 0:
            bid_vwap = self.calculate_vwap(
                traderObject[f"{product}_bid_price_history"],
                traderObject[f"{product}_bid_volume_history"]
            )
            if bid_vwap:
                traderObject[f"{product}_bid_vwap_history"].append(bid_vwap)
        
        if len(traderObject[f"{product}_ask_price_history"]) > 0:
            ask_vwap = self.calculate_vwap(
                traderObject[f"{product}_ask_price_history"],
                traderObject[f"{product}_ask_volume_history"]
            )
            if ask_vwap:
                traderObject[f"{product}_ask_vwap_history"].append(ask_vwap)
        
        # Only proceed if we have both VWAPs
        if bid_vwap and ask_vwap:
            # Predict next prices using separate VWAPs
            next_bid, next_ask = self.predict_next_prices(bid_vwap, ask_vwap, self.params[product])
            
            # Estimate fair value
            fair_value = self.fair_value_estimation(bid_vwap, ask_vwap, next_bid, next_ask)
            traderObject[f"{product}_fair_values"].append(fair_value)
            
            return bid_vwap, ask_vwap, next_bid, next_ask, fair_value
        
        # If we don't have enough data, return None values
        return None, None, None, None, None
    
    def take_opportunity_orders(self, product, order_depth, fair_value, next_bid, next_ask, position):
        """Take orders that offer immediate profit based on predicted prices"""
        orders = []
        position_limit = self.LIMIT[product]
        max_position = int(position_limit * self.params[product]["max_position_pct"])
        
        # Keep track of running position for multiple orders
        buy_volume = 0
        sell_volume = 0
        
        # Safety margins
        safety = self.params[product]["safety_margin"]
        take_threshold = self.params[product]["take_threshold"]
        
        # Adjust thresholds based on current position
        position_ratio = position / position_limit if position_limit > 0 else 0
        buy_threshold = next_bid - take_threshold - (position_ratio * safety)
        sell_threshold = next_ask + take_threshold - (position_ratio * safety)
        
        # Take sell orders (buy opportunities)
        if order_depth.sell_orders and position < max_position:
            sorted_asks = sorted(order_depth.sell_orders.keys())
            for ask_price in sorted_asks:
                # If price is below our threshold, it's profitable to buy
                if ask_price < buy_threshold:
                    available_volume = -order_depth.sell_orders[ask_price]
                    max_buy = min(available_volume, max_position - position - buy_volume)
                    
                    if max_buy > 0:
                        orders.append(Order(product, ask_price, max_buy))
                        buy_volume += max_buy
                else:
                    # Stop if no longer profitable
                    break
        
        # Take buy orders (sell opportunities)
        if order_depth.buy_orders and position > -max_position:
            sorted_bids = sorted(order_depth.buy_orders.keys(), reverse=True)
            for bid_price in sorted_bids:
                # If price is above our threshold, it's profitable to sell
                if bid_price > sell_threshold:
                    available_volume = order_depth.buy_orders[bid_price]
                    max_sell = min(available_volume, max_position + position - sell_volume)
                    
                    if max_sell > 0:
                        orders.append(Order(product, bid_price, -max_sell))
                        sell_volume += max_sell
                else:
                    # Stop if no longer profitable
                    break
        
        return orders, buy_volume, sell_volume
    
    def clear_position_orders(self, product, order_depth, fair_value, position, buy_volume, sell_volume):
        """Create orders to reduce position when price is favorable"""
        orders = []
        position_limit = self.LIMIT[product]
        
        # Calculate position after current orders
        position_after_orders = position + buy_volume - sell_volume
        
        # If no position to clear, return empty list
        if abs(position_after_orders) < 1:
            return orders, buy_volume, sell_volume
        
        clear_threshold = self.params[product]["clear_threshold"]
        
        # If long, look for favorable sell opportunities
        if position_after_orders > 0:
            if order_depth.buy_orders:
                best_bid = max(order_depth.buy_orders.keys())
                # If bid price is close to or above fair value, sell to reduce position
                if best_bid >= fair_value - clear_threshold:
                    available_volume = order_depth.buy_orders[best_bid]
                    max_sell = min(available_volume, position_after_orders)
                    
                    if max_sell > 0:
                        orders.append(Order(product, best_bid, -max_sell))
                        sell_volume += max_sell
        
        # If short, look for favorable buy opportunities
        if position_after_orders < 0:
            if order_depth.sell_orders:
                best_ask = min(order_depth.sell_orders.keys())
                # If ask price is close to or below fair value, buy to reduce position
                if best_ask <= fair_value + clear_threshold:
                    available_volume = -order_depth.sell_orders[best_ask]
                    max_buy = min(available_volume, abs(position_after_orders))
                    
                    if max_buy > 0:
                        orders.append(Order(product, best_ask, max_buy))
                        buy_volume += max_buy
        
        return orders, buy_volume, sell_volume
    
    def market_making_orders(self, product, order_depth, fair_value, next_bid, next_ask, position, buy_volume, sell_volume):
        """Place market making orders based on predicted prices"""
        orders = []
        position_limit = self.LIMIT[product]
        
        # Calculate position after current orders
        position_after_orders = position + buy_volume - sell_volume
        
        # Calculate position scaling factor based on current position
        # This reduces order size as we approach position limits
        position_ratio = abs(position_after_orders) / position_limit if position_limit > 0 else 0
        position_scale = 1 - position_ratio * self.params[product]["position_scaling"]
        
        # Make sure scale factor is positive
        position_scale = max(0.1, position_scale)
        
        # Calculate bid and ask prices
        make_threshold = self.params[product]["make_threshold"]
        
        # Adjust bid/ask based on position
        if position_after_orders > 0:
            # If long, be more aggressive selling, less aggressive buying
            bid_price = next_bid - make_threshold - (position_ratio * make_threshold)
            ask_price = next_ask - (position_ratio * make_threshold / 2)
        elif position_after_orders < 0:
            # If short, be more aggressive buying, less aggressive selling
            bid_price = next_bid + (position_ratio * make_threshold / 2)
            ask_price = next_ask + make_threshold + (position_ratio * make_threshold)
        else:
            # If neutral, use standard bid/ask
            bid_price = next_bid - make_threshold
            ask_price = next_ask + make_threshold
        
        # Round prices to nearest valid value
        bid_price = round(bid_price)
        ask_price = round(ask_price)
        
        # Calculate buy and sell quantities
        max_buy = max(0, position_limit - position_after_orders)
        max_sell = max(0, position_limit + position_after_orders)
        
        # Scale quantities based on position
        buy_quantity = int(max_buy * position_scale)
        sell_quantity = int(max_sell * position_scale)
        
        # Add orders if quantities are positive
        if buy_quantity > 0:
            orders.append(Order(product, bid_price, buy_quantity))
        
        if sell_quantity > 0:
            orders.append(Order(product, ask_price, -sell_quantity))
        
        return orders
    
    def run(self, state: TradingState):
        # Initialize trader state
        traderObject = {}
        if state.traderData and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)
        
        # Initialize result dictionary
        result = {}
        
        # Check if KELP is available to trade
        if Product.KELP in state.order_depths:
            # Get current position
            position = state.position.get(Product.KELP, 0)
            
            # Get order depth
            order_depth = state.order_depths[Product.KELP]
            
            # Update price history and get separate bid/ask VWAPs + predictions
            bid_vwap, ask_vwap, next_bid, next_ask, fair_value = self.update_price_history(
                traderObject, 
                Product.KELP, 
                state
            )
            
            # If we don't have enough history, use mid price as fair value
            if fair_value is None and order_depth.buy_orders and order_depth.sell_orders:
                best_bid = max(order_depth.buy_orders.keys())
                best_ask = min(order_depth.sell_orders.keys())
                
                # Use current best bid/ask as our bid/ask VWAPs
                bid_vwap = best_bid
                ask_vwap = best_ask
                fair_value = (best_bid + best_ask) / 2
                
                # Use separate bid/ask values to estimate next bid/ask
                next_bid = bid_vwap + self.params[Product.KELP]["bid_offset"]
                next_ask = ask_vwap + self.params[Product.KELP]["ask_offset"]
            
            # Initialize order tracking
            kelp_orders = []
            buy_volume = 0
            sell_volume = 0
            
            # Skip trading if we don't have price predictions
            if fair_value is not None and next_bid is not None and next_ask is not None:
                # STEP 1: Take immediate opportunities
                opportunity_orders, buy_vol, sell_vol = self.take_opportunity_orders(
                    Product.KELP,
                    order_depth,
                    fair_value,
                    next_bid,
                    next_ask,
                    position
                )
                kelp_orders.extend(opportunity_orders)
                buy_volume += buy_vol
                sell_volume += sell_vol
                
                # STEP 2: Clear positions when favorable
                clear_orders, buy_vol, sell_vol = self.clear_position_orders(
                    Product.KELP,
                    order_depth,
                    fair_value,
                    position,
                    buy_volume,
                    sell_volume
                )
                kelp_orders.extend(clear_orders)
                buy_volume += buy_vol
                sell_volume += sell_vol
                
                # STEP 3: Market making
                make_orders = self.market_making_orders(
                    Product.KELP,
                    order_depth,
                    fair_value,
                    next_bid,
                    next_ask,
                    position,
                    buy_volume,
                    sell_volume
                )
                kelp_orders.extend(make_orders)
            
            # Add orders to result
            if kelp_orders:
                result[Product.KELP] = kelp_orders
        
        # Save trader state
        traderData = jsonpickle.encode(traderObject)
        
        # Return result
        return result, 0, traderData