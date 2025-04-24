from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict
import string
import jsonpickle
import numpy as np
import math


class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"

'''
RAINFOREST_RESIN Trading Strategy Explanation

Overview
This trading strategy for RAINFOREST_RESIN combines three complementary approaches: opportunity taking, position clearing, and market making. The strategy operates on the premise that RAINFOREST_RESIN has a stable intrinsic value of 10,000, similar to amethysts from previous year. By carefully managing positions and optimizing order execution, the strategy aims to capture small price discrepancies around this fair value.

################################################
Step 1: Opportunity Taking
################################################

The strategy begins by scanning the order book for immediate profit opportunities:

Identify Mispriced Orders: The algorithm looks for sell orders below fair value (10,000) and buy orders above fair value.
Prioritize by Profitability: It sorts all sell orders by price (lowest first) and buy orders by price (highest first) to handle the most profitable opportunities first.
Calculate Trade Quantity: For each price level, the algorithm determines the maximum quantity it can trade while respecting position limits (50 for RAINFOREST_RESIN).
Execute Immediate Trades: It creates orders to immediately trade against these mispriced orders, buying below fair value and selling above.
Track Position: Throughout this process, it carefully tracks running position to ensure it never exceeds limits.

################################################
Step 2: Position Clearing
################################################

After taking immediate opportunities, the strategy focuses on risk management:
Assess Current Position: It calculates what the position will be after all opportunity-taking orders are executed.
Identify Clearing Opportunities: If the position is long (positive), it looks for favorable buy orders to sell to. If the position is short (negative), it looks for favorable sell orders to buy from.
Calculate Clearing Prices: It uses a tighter spread (fair value ± 0.5) for clearing positions compared to opportunity taking (fair value ± 1).
Submit Clearing Orders: It sends orders designed to reduce position when favorable prices are available, helping manage risk.
Respect Position Flow: The clearing logic ensures that selling when long and buying when short doesn't create a new imbalanced position in the opposite direction.

################################################
Step 3: Market Making
################################################

Finally, the strategy provides liquidity through market making:
Analyze Existing Order Levels: It examines the order book to identify the best levels to join or improve upon ("penny").
Position-Based Pricing: It adjusts pricing strategy based on the current position:

When not position constrained (position near zero), it seeks wider spreads for better profit
When approaching position limits, it tightens spreads to encourage position-reducing trades

################################################
Calculate Optimal Bid/Ask: 
################################################

It determines the ideal prices at which to place orders:
When other orders are nearby, it either joins them or improves by 1 (pennies them)
When no good reference prices exist, it uses a default edge of 4 from fair value


Submit Market Making Orders: It places buy orders at its calculated bid price and sell orders at its calculated ask price, with quantities that respect position limits.
Dynamic Adjustment: The pricing logic automatically adjusts as positions and market conditions change.

Strategy Improvements from Amethysts
This strategy incorporates several enhancements from successful amethysts trading:

Increased Position Limits: Utilizing the full 50-position limit (up from 20) to capture more opportunities.
Enhanced Order Execution: Scanning multiple price levels and prioritizing by profitability rather than just taking the best bid/ask.
Risk Management: Implementing position clearing with a tighter spread to reduce risk without sacrificing too much profit.
Careful Position Tracking: Tracking running position across all order types to ensure position limits are never exceeded.

Algorithm Flow
Each time the strategy runs:

It first executes opportunity taking to capture immediate profits
Then runs position clearing to manage risk
Finally places market making orders to provide liquidity

All three components work together to generate profit while carefully managing risk through intelligent position management.
'''

# Configuration parameters for trading RAINFOREST_RESIN
# These parameters control various aspects of the trading strategy


'''
Past data - Trades

bid_price_1  bid_volume_1   bid_price_2  bid_volume_2  bid_price_3  \
count  30000.000000  30000.000000  22311.000000  22311.000000  5141.000000   
mean    9996.609633      8.218933   9995.290395     19.714939  9995.083252   
std        1.937861     10.155552      0.634420     10.142682     0.296664   
min     9995.000000      0.000000   9995.000000      1.000000  9995.000000   
25%     9995.000000      1.000000   9995.000000     20.000000  9995.000000   
50%     9996.000000      2.000000   9995.000000     23.000000  9995.000000   
75%     9996.000000     20.000000   9995.000000     27.000000  9995.000000   
max    10002.000000     30.000000  10000.000000     30.000000  9998.000000   

       bid_volume_3   ask_price_1  ask_volume_1  ask_price_2  ask_volume_2  \
count   5141.000000  30000.000000  30000.000000  22312.00000  22312.000000   
mean      23.099981  10003.383467      8.212767  10004.70433     19.669953   
std        6.997590      1.949362     10.158015      0.64945     10.137293   
min        1.000000   9998.000000      1.000000  10000.00000      1.000000   
25%       22.000000  10004.000000      1.000000  10005.00000     20.000000   
50%       24.000000  10004.000000      2.000000  10005.00000     23.000000   
75%       27.000000  10005.000000     20.000000  10005.00000     27.000000   
max       30.000000  10005.000000     30.000000  10005.00000     30.000000   

        ask_price_3  ask_volume_3  
count   5169.000000   5169.000000  
mean   10004.912749     23.112014  
std        0.304014      7.147199  
min    10002.000000      1.000000  
25%    10005.000000     22.000000  
50%    10005.000000     25.000000  
75%    10005.000000     28.000000  
max    10005.000000     30.000000  
'''

PARAMS = {
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000,          # The estimated fair price of RAINFOREST_RESIN
        "take_width": 2,              # How far from fair value we're willing to buy/sell immediately
        "clear_width": 0.5,           # Increased from 0 to match successful amethyst strategy - controls position clearing
        "disregard_edge": 1,          # Ignores orders for joining/pennying within this distance from fair value
        "join_edge": 2,               # Join orders within this edge rather than pennying
        "default_edge": 4,            # Default spread to use when no good reference prices exist
        "soft_position_limit": 25,    # Reduced position limit to be more conservative than hard limit
        "volume_limit": 0,            # Added for position management like in amethysts
        "penny_val": 1,               # The value to adjust prices by when pennying
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

    # This method scans the order book and takes advantage of mispriced orders
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

        # More sophisticated order execution logic for sell orders (we buy)
        if len(order_depth.sell_orders) != 0:
            # Sort sell orders by price (lowest first) to prioritize most profitable orders
            # This ensures we take the best deals first
            sorted_sell_prices = sorted(order_depth.sell_orders.keys())
            
            for price in sorted_sell_prices:
                # Stop if price is no longer profitable (above our buying threshold)
                if price > fair_value - take_width:
                    break
                
                # Calculate how much we can buy at this price level
                best_ask_amount = -1 * order_depth.sell_orders[price]  # Convert negative quantity to positive
                # Limit quantity by position limit and what's already been bought
                quantity = min(best_ask_amount, position_limit - position - buy_order_volume)
                
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
                # Stop if price is no longer profitable (below our selling threshold)
                if price < fair_value + take_width:
                    break
                
                # Calculate how much we can sell at this price level
                best_bid_amount = order_depth.buy_orders[price]
                # Limit quantity by position limit and what's already been sold
                quantity = min(best_bid_amount, position_limit + position - sell_order_volume)
                
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
    def make_resin_orders(
        self,
        order_depth: OrderDepth,
        fair_value: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        volume_limit: int,
        penny_val = 1
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        
        # Find best price levels to join or penny
        # First, find all sell orders above fair value + some threshold
        baaf = min(
            [
                price
                for price in order_depth.sell_orders.keys()
                if price > fair_value + penny_val
            ]
        )
        
        # Find all buy orders below fair value - some threshold
        bbbf = max(
            [price for price in order_depth.buy_orders.keys() if price < fair_value - penny_val]
        )

        # Adjust pricing based on current position
        # If we're not position constrained on the buy side, we want better edge
        if baaf <= fair_value + 2:
            if position <= volume_limit:
                # Increase ask price to get better profit margin when not position constrained
                baaf = fair_value + 3  # still want edge 2 if position is not a concern

        # If we're not position constrained on the sell side, we want better edge
        if bbbf >= fair_value - 2:
            if position >= -volume_limit:
                # Decrease bid price to get better profit margin when not position constrained
                bbbf = fair_value - 3  # still want edge 2 if position is not a concern

        # Market make with adjusted prices
        # The +1 and -1 adjust the prices to be slightly better than existing levels
        buy_order_volume, sell_order_volume = self.market_make(
            Product.RAINFOREST_RESIN,
            orders,
            bbbf + 1,  # Set bid price just above the best non-competitive bid
            baaf - 1,  # Set ask price just below the best non-competitive ask
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
                self.params[Product.RAINFOREST_RESIN]["penny_val"],
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