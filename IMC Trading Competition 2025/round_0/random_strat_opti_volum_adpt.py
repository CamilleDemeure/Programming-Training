from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict
import string
import jsonpickle
import numpy as np
import math


class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"


# Configuration parameters for trading RAINFOREST_RESIN
# These parameters control various aspects of the trading strategy
# Parameters optimized based on historical data analysis
PARAMS = {
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000,          # The estimated fair price of RAINFOREST_RESIN (matches mean trade price of 9999.97)
        "take_width": 2,              # Increased to 2 based on standard deviation of 3.15
        "clear_width": 1,             # Increased to 1 to better balance clearing positions vs giving away edge
        "disregard_edge": 1,          # Kept at 1 as appropriate for the 10-point price range
        "join_edge": 3,               # Increased to 3 based on typical spread of ~8 points
        "default_edge": 3,            # Decreased to 3 to position orders in middle of typical price range
        "soft_position_limit": 25,    # Kept at half the hard limit for good position management
        "volume_limit": 15,           # Set to 15 based on typical larger order volumes of 20-30
        "small_volume_threshold": 5,  # Added to identify small volume orders
        "large_volume_threshold": 20, # Added to identify large volume orders
    },
}


class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params

        # (1) Position Limit Adjustment - Updated to match the position limit of 50
        # This allows us to take larger positions when profitable opportunities arise
        self.LIMIT = {Product.RAINFOREST_RESIN: 50}

    # (7) Enhanced Order Execution Logic - Improved take_best_orders implementation
    # This method scans the order book and takes advantage of mispriced orders
    # Now with volume-based adjustment of aggressiveness
    def take_best_orders(
        self,
        product: str,
        fair_value: int,
        take_width: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (int, int):
        # Get the position limit for the product
        position_limit = self.LIMIT[product]
        
        # Get volume thresholds from parameters
        small_volume_threshold = self.params[product].get("small_volume_threshold", 5)
        large_volume_threshold = self.params[product].get("large_volume_threshold", 20)

        # More sophisticated order execution logic for sell orders (we buy)
        if len(order_depth.sell_orders) != 0:
            # Sort sell orders by price (lowest first) to prioritize most profitable orders
            # This ensures we take the best deals first
            sorted_sell_prices = sorted(order_depth.sell_orders.keys())
            
            for price in sorted_sell_prices:
                volume = -1 * order_depth.sell_orders[price]  # Convert negative quantity to positive
                
                # Adjust take width based on volume
                adjusted_take_width = take_width
                if volume <= small_volume_threshold:
                    # More aggressive for small volumes (likely retail)
                    adjusted_take_width = take_width * 1.2
                elif volume >= large_volume_threshold:
                    # Less aggressive for large volumes (likely market makers)
                    adjusted_take_width = take_width * 0.8
                
                # Stop if price is no longer profitable (above our buying threshold)
                if price > fair_value - adjusted_take_width:
                    break
                
                # Calculate how much we can buy at this price level
                # Limit quantity by position limit and what's already been bought
                quantity = min(volume, position_limit - position - buy_order_volume)
                
                if quantity > 0:
                    # Create buy order at this price
                    orders.append(Order(product, price, quantity))
                    buy_order_volume += quantity
                    
                    # Update order depth to reflect our order (for future iterations)
                    order_depth.sell_orders[price] += quantity
                    if order_depth.sell_orders[price] == 0:
                        del order_depth.sell_orders[price]

        # More sophisticated order execution logic for buy orders (we sell)
        if len(order_depth.buy_orders) != 0:
            # Sort buy orders by price (highest first) to prioritize most profitable orders
            sorted_buy_prices = sorted(order_depth.buy_orders.keys(), reverse=True)
            
            for price in sorted_buy_prices:
                volume = order_depth.buy_orders[price]
                
                # Adjust take width based on volume
                adjusted_take_width = take_width
                if volume <= small_volume_threshold:
                    # More aggressive for small volumes (likely retail)
                    adjusted_take_width = take_width * 1.2
                elif volume >= large_volume_threshold:
                    # Less aggressive for large volumes (likely market makers)
                    adjusted_take_width = take_width * 0.8
                
                # Stop if price is no longer profitable (below our selling threshold)
                if price < fair_value + adjusted_take_width:
                    break
                
                # Calculate how much we can sell at this price level
                # Limit quantity by position limit and what's already been sold
                quantity = min(volume, position_limit + position - sell_order_volume)
                
                if quantity > 0:
                    # Create sell order at this price
                    orders.append(Order(product, price, -1 * quantity))
                    sell_order_volume += quantity
                    
                    # Update order depth to reflect our order
                    order_depth.buy_orders[price] -= quantity
                    if order_depth.buy_orders[price] == 0:
                        del order_depth.buy_orders[price]

        return buy_order_volume, sell_order_volume

    # This method places market making orders at calculated bid and ask prices
    # Market making provides liquidity and captures the spread between bids and asks
    def market_make(
        self,
        product: str,
        orders: List[Order],
        bid: int,
        ask: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (int, int):
        # Calculate maximum buy quantity based on position limit and existing buys
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            # Place a bid (buy order) at the calculated bid price
            orders.append(Order(product, round(bid), buy_quantity))

        # Calculate maximum sell quantity based on position limit and existing sells
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            # Place an ask (sell order) at the calculated ask price
            orders.append(Order(product, round(ask), -sell_quantity))
            
        return buy_order_volume, sell_order_volume

    # This method helps reduce unwanted positions by placing orders to move position toward zero
    # It's especially useful after taking advantage of mispriced orders
    def clear_position_order(
        self,
        product: str,
        fair_value: float,
        width: int,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (int, int):
        # Calculate what position will be after the orders in progress
        position_after_take = position + buy_order_volume - sell_order_volume
        
        # Calculate prices at which we're willing to clear positions
        fair_for_bid = round(fair_value - width)  # Price to buy at when clearing short positions
        fair_for_ask = round(fair_value + width)  # Price to sell at when clearing long positions

        # Calculate maximum quantities we can trade
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)

        # If we're long (positive position), try to reduce it by selling
        if position_after_take > 0:
            # Find all buy orders at or above our selling price
            clear_quantity = sum(
                volume
                for price, volume in order_depth.buy_orders.items()
                if price >= fair_for_ask
            )
            # Limit by how much we need to sell and how much we can sell
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            
            if sent_quantity > 0:
                # Create a sell order to reduce position
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        # If we're short (negative position), try to reduce it by buying
        if position_after_take < 0:
            # Find all sell orders at or below our buying price
            clear_quantity = sum(
                abs(volume)
                for price, volume in order_depth.sell_orders.items()
                if price <= fair_for_bid
            )
            # Limit by how much we need to buy and how much we can buy
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            
            if sent_quantity > 0:
                # Create a buy order to reduce position
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)

        return buy_order_volume, sell_order_volume

    # Wrapper method that calls take_best_orders
    # This separates different trading strategies for clarity
    def take_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        take_width: float,
        position: int,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        # Call the enhanced order execution logic
        buy_order_volume, sell_order_volume = self.take_best_orders(
            product,
            fair_value,
            take_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    # Wrapper method that calls clear_position_order
    # This handles reducing unwanted positions
    def clear_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        clear_width: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        
        # Call position clearing method
        buy_order_volume, sell_order_volume = self.clear_position_order(
            product,
            fair_value,
            clear_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    # (4) Improved Position Management - Implemented from amethysts strategy
    # This method handles market making with dynamic pricing based on position
    # Also incorporates volume-based pricing based on market analysis
    def make_resin_orders(
        self,
        order_depth: OrderDepth,
        fair_value: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        volume_limit: int,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        
        # Calculate disregard threshold based on parameters
        disregard_edge = self.params[Product.RAINFOREST_RESIN]["disregard_edge"]
        join_edge = self.params[Product.RAINFOREST_RESIN]["join_edge"]
        default_edge = self.params[Product.RAINFOREST_RESIN]["default_edge"]
        
        # Find best price levels to join or penny
        # First, find all sell orders above fair value + disregard threshold
        ask_prices_above_threshold = [
            price
            for price in order_depth.sell_orders.keys()
            if price > fair_value + disregard_edge
        ]
        
        # Find all buy orders below fair value - disregard threshold
        bid_prices_below_threshold = [
            price 
            for price in order_depth.buy_orders.keys() 
            if price < fair_value - disregard_edge
        ]
        
        # Get the closest price levels or use default if none found
        baaf = min(ask_prices_above_threshold) if ask_prices_above_threshold else fair_value + default_edge
        bbbf = max(bid_prices_below_threshold) if bid_prices_below_threshold else fair_value - default_edge

        # Position-based pricing adjustment
        # Calculate how aggressive to be based on current position relative to limits
        position_ratio = position / self.LIMIT[Product.RAINFOREST_RESIN]  # Between -1 and 1
        
        # Adjust pricing based on current position and market conditions
        if abs(position) <= volume_limit:
            # When position is small, aim for wider spreads (more profit)
            # Adjust ask price - increase when not position constrained
            if baaf <= fair_value + default_edge:
                baaf = fair_value + default_edge
                
            # Adjust bid price - decrease when not position constrained
            if bbbf >= fair_value - default_edge:
                bbbf = fair_value - default_edge
        else:
            # When position is larger, adjust pricing to encourage balancing trades
            if position > 0:  # Long position - be more aggressive on selling
                # Reduce ask price to encourage sells
                position_adjustment = max(0, int(position_ratio * 2))
                baaf = min(baaf, fair_value + default_edge - position_adjustment)
            else:  # Short position - be more aggressive on buying
                # Increase bid price to encourage buys
                position_adjustment = max(0, int(-position_ratio * 2))
                bbbf = max(bbbf, fair_value - default_edge + position_adjustment)

        # Check if we should join or penny existing levels
        for ask_price in sorted(order_depth.sell_orders.keys()):
            if fair_value < ask_price <= fair_value + join_edge:
                # Join this ask level instead of creating our own
                baaf = ask_price
                break
                
        for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
            if fair_value > bid_price >= fair_value - join_edge:
                # Join this bid level instead of creating our own
                bbbf = bid_price
                break

        # Market make with adjusted prices
        # The +1 and -1 adjust the prices to be slightly better than existing levels if not joining
        buy_order_volume, sell_order_volume = self.market_make(
            Product.RAINFOREST_RESIN,
            orders,
            bbbf if bbbf in order_depth.buy_orders.keys() else bbbf + 1,  # Join or penny
            baaf if baaf in order_depth.sell_orders.keys() else baaf - 1,  # Join or penny
            position,
            buy_order_volume,
            sell_order_volume,
        )
        
        return orders, buy_order_volume, sell_order_volume

    # Main method called by the trading environment at each iteration
    def run(self, state: TradingState):
        # Deserialize trader state data if available
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

        # Initialize result dictionary to store orders for each product
        result = {}

        # Check if RAINFOREST_RESIN is available to trade
        if Product.RAINFOREST_RESIN in self.params and Product.RAINFOREST_RESIN in state.order_depths:
            # Get current position in RAINFOREST_RESIN
            resin_position = (
                state.position[Product.RAINFOREST_RESIN]
                if Product.RAINFOREST_RESIN in state.position
                else 0
            )
            
            # STEP 1: Taking orders - enhanced execution logic helps get better prices
            # This finds mispriced orders in the market and takes advantage of them
            resin_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(
                    Product.RAINFOREST_RESIN,
                    state.order_depths[Product.RAINFOREST_RESIN],
                    self.params[Product.RAINFOREST_RESIN]["fair_value"],
                    self.params[Product.RAINFOREST_RESIN]["take_width"],
                    resin_position,
                )
            )
            
            # STEP 2: Clearing positions - improved to reduce exposure when needed
            # This helps manage risk by unwinding positions when favorable opportunities arise
            resin_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(
                    Product.RAINFOREST_RESIN,
                    state.order_depths[Product.RAINFOREST_RESIN],
                    self.params[Product.RAINFOREST_RESIN]["fair_value"],
                    self.params[Product.RAINFOREST_RESIN]["clear_width"],
                    resin_position,
                    buy_order_volume,
                    sell_order_volume,
                )
            )
            
            # STEP 3: Making orders - using the improved position management from amethysts
            # This places market making orders with intelligent pricing based on position
            resin_make_orders, _, _ = self.make_resin_orders(
                state.order_depths[Product.RAINFOREST_RESIN],
                self.params[Product.RAINFOREST_RESIN]["fair_value"],
                resin_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.RAINFOREST_RESIN]["volume_limit"],
            )
            
            # Combine all orders for RAINFOREST_RESIN
            result[Product.RAINFOREST_RESIN] = (
                resin_take_orders + resin_clear_orders + resin_make_orders
            )

        # No conversions needed for RAINFOREST_RESIN (unlike some other products)
        conversions = 0
        
        # Serialize trader state data for next iteration
        traderData = jsonpickle.encode(traderObject)

        # Return orders, conversions, and trader state
        return result, conversions, traderData