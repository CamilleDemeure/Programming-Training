from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict
import string
import jsonpickle
import numpy as np
import math

# =============================================================================
# Product Class
# =============================================================================
# Define product constants for clarity. Here, only PICNIC_BASKET2 is in use.
class Product:
    PICNIC_BASKET2 = "PICNIC_BASKET2"

# =============================================================================
# PARAMS Dictionary
# =============================================================================
# This dictionary holds key trading parameters for PICNIC_BASKET2. Each parameter is
# tuned to balance risk, capture pricing inefficiencies, and manage position
# limits. Below is an explanation for each parameter:
#
# base_fair_value:
#   - The starting estimate of the true market value of PICNIC_BASKET2 based on historical data.
#   - Here set to 2033 (observed median price), serving as an anchor until enough
#     market data refines this value.
#
# take_width:
#   - The price range (distance from fair value) that qualifies an order for
#     immediate execution (opportunity taking).
#   - A wider take_width (here 3) allows the trader to capture orders that are
#     slightly farther away from the current fair value.
#
# clear_width:
#   - The threshold distance from fair value used to trigger position clearing.
#   - A value of 1 means that if the market price moves 1 unit away from the fair
#     value, the trader starts to unwind positions to reduce risk.
#
# disregard_edge:
#   - Orders within this price distance from fair value are ignored for joining
#     or pennying. This avoids orders that are too close to fair value to be profitable.
#
# join_edge:
#   - When orders are within this edge, the strategy decides to join them instead
#     of trying to outbid (penny) the competition.
#
# default_edge:
#   - A fallback price offset used when no better reference prices are available.
#   - Set to 3 here (reduced from 4) to align with the typical market spread.
#
# soft_position_limit:
#   - A dynamically adjustable limit to manage risk. Although the hard limit is 50,
#     this soft limit (set to 35 here) is used for more conservative position sizing.
#
# volume_limit:
#   - A parameter for finer position management (not actively changed in this code).
#
# penny_val:
#   - The adjustment increment for 'pennying'; that is, modifying order prices
#     slightly (by 1 unit) to outcompete other orders.
#
# vwap_window:
#   - The number of historical data points to use when computing a dynamic fair value
#     via a rolling volume-weighted average price (VWAP). A smaller window (10)
#     makes the fair value more responsive to market changes.
# =============================================================================

'''
SPREAD DISTRIBUTION
count    30000.000000
mean        -5.225933
std          1.141285
min         -7.000000
25%         -6.000000
50%         -5.000000
75%         -5.000000
max         -1.000000

ASSET PRICE DISTRIBUTION
count    30000.000000
mean     30410.844400
std        214.993748
min      29990.000000
25%      30239.000000
50%      30429.000000
75%      30592.000000
max      30816.000000
'''

PARAMS = {
    Product.PICNIC_BASKET2: {
        "base_fair_value": 30429,       # Median value
        "take_width": 5,                # Increased for safer entry points
        "clear_width": 0.5,             # Reduced to clear positions more aggressively
        "disregard_edge": 3,            # Maintained
        "join_edge": 5,                 # Maintained
        "default_edge": 7,              # Increased for wider spreads
        "soft_position_limit": 30,      # Significantly reduced from 42
        "volume_limit": 0,              # Unchanged
        "penny_val": 1,                 # Unchanged
        "vwap_window": 100,             # Increased for more stable fair value
        # Aggressiveness parameters
        "aggressiveness": 0.3,          # Significantly reduced for safer approach
        "fair_value_order_size": 0,     # Removed fair value orders (highest risk)
        "tier1_distance": 8,            # Increased distance from fair value
        "tier2_distance": 15,           # Increased distance
        "tier3_distance": 25,           # Increased distance
        "tier1_volume_pct": 0.4,        # Reduced allocation to closest tier
        "tier2_volume_pct": 0.35,       # Increased middle tier allocation
        "tier3_volume_pct": 0.25,       # Increased furthest tier allocation
        "position_scaling": 0.3,        # Increased to reduce position concentration
    },
}


# =============================================================================
# Trader Class
# =============================================================================
class Trader:
    def __init__(self, params=None):
        # Use provided parameters or default to the PARAMS defined above.
        if params is None:
            params = PARAMS
        self.params = params

        # Limit for the maximum allowed position for PICNIC_BASKET2. This is the hard limit,
        # corresponding to the exchange or strategy constraints.
        self.LIMIT = {Product.PICNIC_BASKET2: 50}

    # -------------------------------------------------------------------------
    # Dynamic Fair Value Calculation
    # -------------------------------------------------------------------------
    def calculate_fair_value_PICNIC_BASKET2(
        self,
        product: str,
        order_depth: OrderDepth,
        traderObject: Dict
    ) -> float:
        """
        Calculate a dynamic fair value based on the volume-weighted average price (VWAP)
        of bid and ask orders.
        
        Parameters:
        - product: The trading instrument (here, PICNIC_BASKET2).
        - order_depth: Contains buy and sell order data from the market.
        - traderObject: Persistent storage (dictionary) used to keep historical price data.
        
        Returns:
        - fair_value: The updated market fair value based on recent order data.
        """
        # Initialize historical data storage for the product if it doesn't exist.
        if f"{product}_price_data" not in traderObject:
            traderObject[f"{product}_price_data"] = []
            traderObject[f"{product}_volume_data"] = []
            traderObject[f"{product}_fair_value"] = self.params[product]["base_fair_value"]
        
        # Create a container for the current snapshot of price information.
        current_data = {}
        
        # Extract bid orders: the keys are prices, and the values are volumes.
        bid_prices = list(order_depth.buy_orders.keys())
        bid_volumes = [order_depth.buy_orders[price] for price in bid_prices]
        
        # Extract ask orders: convert negative volumes to positive for calculation.
        ask_prices = list(order_depth.sell_orders.keys())
        ask_volumes = [-order_depth.sell_orders[price] for price in ask_prices]
        
        # If either side of the market is empty, return the most recent fair value.
        if not bid_prices or not ask_prices:
            return traderObject[f"{product}_fair_value"]
        
        # Identify the best bid (highest price) and best ask (lowest price) along with their volumes.
        best_bid = max(bid_prices)
        best_ask = min(ask_prices)
        best_bid_volume = order_depth.buy_orders[best_bid]
        best_ask_volume = abs(order_depth.sell_orders[best_ask])  # Ensure volume is positive
        
        # Calculate bid VWAP: average bid price weighted by volume.
        bid_vwap = (
            sum(p * v for p, v in zip(bid_prices, bid_volumes)) / sum(bid_volumes)
            if sum(bid_volumes) > 0 else best_bid
        )
        
        # Calculate ask VWAP: average ask price weighted by volume.
        ask_vwap = (
            sum(p * v for p, v in zip(ask_prices, ask_volumes)) / sum(ask_volumes)
            if sum(ask_volumes) > 0 else best_ask
        )
        
        # Compute the volume-weighted mid price from the best bid and ask.
        weighted_mid_price = (best_bid * best_bid_volume + best_ask * best_ask_volume) / (best_bid_volume + best_ask_volume)
        
        # Package the current calculated VWAP values and mid price into the snapshot.
        current_data["bid_vwap"] = bid_vwap
        current_data["ask_vwap"] = ask_vwap
        current_data["mid_price"] = weighted_mid_price
        
        # Append the current snapshot to the historical price data.
        traderObject[f"{product}_price_data"].append(current_data)
        
        # Keep only the most recent data points up to the specified vwap_window size.
        window_size = self.params[product]["vwap_window"]
        if len(traderObject[f"{product}_price_data"]) > window_size:
            traderObject[f"{product}_price_data"] = traderObject[f"{product}_price_data"][-window_size:]
        
        # If there is enough historical data, compute the fair value as the average of recent mid prices.
        if len(traderObject[f"{product}_price_data"]) >= window_size:
            recent_mid_prices = [data["mid_price"] for data in traderObject[f"{product}_price_data"]]
            fair_value = sum(recent_mid_prices) / len(recent_mid_prices)
        else:
            # Without enough history, default to the current computed mid price.
            fair_value = current_data["mid_price"]

        # Update and persist the new fair value.
        traderObject[f"{product}_fair_value"] = fair_value
        
        return fair_value

    # -------------------------------------------------------------------------
    # Taking Advantage of Mispriced Orders (Opportunity Taking)
    # -------------------------------------------------------------------------
    def take_best_orders_PICNIC_BASKET2(
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
        """
        Scans the order book and executes orders that are mispriced relative to the fair value.
        
        Parameters:
        - product: Trading instrument (PICNIC_BASKET2).
        - fair_value: Current calculated fair value.
        - take_width: Threshold to determine if an order is priced attractively.
        - orders: List to accumulate new orders.
        - order_depth: Current market order book.
        - position: Current net position in PICNIC_BASKET2.
        - buy_order_volume: Running total of PICNIC_BASKET2 already being bought this round.
        - sell_order_volume: Running total of PICNIC_BASKET2 already being sold this round.
        
        Returns:
        - Updated buy_order_volume and sell_order_volume after processing orders.
        """
        # Retrieve the hard position limit for the product.
        position_limit = self.LIMIT[product]

        # --- Sell Orders (Opportunity to Buy) ---
        # If there are sell orders available, process to capture prices below (fair_value - take_width).
        if len(order_depth.sell_orders) != 0:
            # Sort sell order prices in ascending order to address the lowest priced offers first.
            sorted_sell_prices = sorted(order_depth.sell_orders.keys())
            
            for price in sorted_sell_prices:
                # If the ask price is too high, stop scanning (no more profitable opportunities).
                if price > fair_value - take_width:
                    break
                
                # Determine the available volume at this price level.
                best_ask_amount = -1 * order_depth.sell_orders[price]
                # Limit the order based on remaining position capacity.
                quantity = min(best_ask_amount, position_limit - position - buy_order_volume)
                
                if quantity > 0:
                    # Submit a buy order at the current price.
                    orders.append(Order(product, price, quantity))
                    buy_order_volume += quantity
                    
                    # Update the order book to reflect executed quantity.
                    order_depth.sell_orders[price] += quantity
                    if order_depth.sell_orders[price] == 0:
                        del order_depth.sell_orders[price]

        # --- Buy Orders (Opportunity to Sell) ---
        # Process buy orders to sell at prices above (fair_value + take_width).
        if len(order_depth.buy_orders) != 0:
            # Sort buy order prices in descending order to target the highest bids first.
            sorted_buy_prices = sorted(order_depth.buy_orders.keys(), reverse=True)
            
            for price in sorted_buy_prices:
                # If the bid price is too low, stop processing further.
                if price < fair_value + take_width:
                    break
                
                # Determine available volume at this buy order level.
                best_bid_amount = order_depth.buy_orders[price]
                # Ensure we do not exceed the maximum sell capacity.
                quantity = min(best_bid_amount, position_limit + position - sell_order_volume)
                
                if quantity > 0:
                    # Submit a sell order (negative quantity) at the current price.
                    orders.append(Order(product, price, -1 * quantity))
                    sell_order_volume += quantity
                    
                    # Update the order book after matching orders.
                    order_depth.buy_orders[price] -= quantity
                    if order_depth.buy_orders[price] == 0:
                        del order_depth.buy_orders[price]

        return buy_order_volume, sell_order_volume

    # -------------------------------------------------------------------------
    # Market Making Orders (Providing Liquidity)
    # -------------------------------------------------------------------------
    def market_make_PICNIC_BASKET2(
        self,
        product: str,
        orders: List[Order],
        bid: int,
        ask: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (int, int):
        """
        Places market making orders to supply liquidity. This method computes
        how many shares can be offered on each side while respecting the position limit.
        
        Parameters:
        - product: Trading instrument (PICNIC_BASKET2).
        - orders: List where new market making orders are appended.
        - bid: Bid price to post for buying.
        - ask: Ask price to post for selling.
        - position: Current net position of PICNIC_BASKET2.
        - buy_order_volume: Quantity already bought.
        - sell_order_volume: Quantity already sold.
        
        Returns:
        - Updated buy_order_volume and sell_order_volume after market making orders.
        """
        # Compute how many units can be bought without breaching the upper position limit.
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            # Post a buy order at the given bid price.
            orders.append(Order(product, round(bid), buy_quantity))

        # Compute how many units can be sold without breaching the lower position limit.
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            # Post a sell order at the given ask price (note the negative sign for sells).
            orders.append(Order(product, round(ask), -sell_quantity))
            
        return buy_order_volume, sell_order_volume

    # -------------------------------------------------------------------------
    # Position Clearing (Risk Management)
    # -------------------------------------------------------------------------
    def clear_position_order_PICNIC_BASKET2(
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
        """
        Executes clearing orders to reduce over-sized positions, helping to control risk.
        Uses a tighter spread around fair_value for these orders.
        
        Parameters:
        - product: Trading instrument (PICNIC_BASKET2).
        - fair_value: Current fair value estimate.
        - width: Half-width of the clearing window to determine execution prices.
        - orders: List to append clearing orders.
        - order_depth: Current market order book.
        - position: Current net position.
        - buy_order_volume: Units already committed in buys.
        - sell_order_volume: Units already committed in sells.
        
        Returns:
        - Updated buy_order_volume and sell_order_volume after position clearing.
        """
        # Estimate the net position after considering orders already executed.
        position_after_take = position + buy_order_volume - sell_order_volume
        
        # Define target prices for clearing positions:
        # fair_for_bid: Price at which to buy when short.
        # fair_for_ask: Price at which to sell when long.
        fair_for_bid = round(fair_value - width)
        fair_for_ask = round(fair_value + width)

        # Determine the remaining capacity for buying and selling.
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)

        # If the trader is long, attempt to clear by selling.
        if position_after_take > 0:
            # Sum the volumes available at prices above our target selling price.
            clear_quantity = sum(
                volume
                for price, volume in order_depth.buy_orders.items()
                if price >= fair_for_ask
            )
            # Do not clear more than the excess long position.
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            
            if sent_quantity > 0:
                # Place a sell order to reduce the long position.
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        # If the trader is short, attempt to clear by buying.
        if position_after_take < 0:
            # Sum the volumes available at prices below our target buying price.
            clear_quantity = sum(
                abs(volume)
                for price, volume in order_depth.sell_orders.items()
                if price <= fair_for_bid
            )
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            
            if sent_quantity > 0:
                # Place a buy order to cover the short position.
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)

        return buy_order_volume, sell_order_volume

    # -------------------------------------------------------------------------
    # Wrapper Methods for Order Taking and Clearing
    # -------------------------------------------------------------------------
    def take_orders_PICNIC_BASKET2(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        take_width: float,
        position: int,
    ) -> (List[Order], int, int):
        """
        Wrapper for processing mispriced orders.
        
        Parameters:
        - product: Trading instrument (PICNIC_BASKET2).
        - order_depth: Current market depth.
        - fair_value: Market fair value.
        - take_width: Threshold distance for taking orders.
        - position: Current position.
        
        Returns:
        - A tuple containing the list of orders along with updated buy and sell volumes.
        """
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        # Execute opportunity taking logic.
        buy_order_volume, sell_order_volume = self.take_best_orders_PICNIC_BASKET2(
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

    def clear_orders_PICNIC_BASKET2(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        clear_width: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (List[Order], int, int):
        """
        Wrapper for processing position clearing orders.
        
        Parameters:
        - product: Trading instrument (PICNIC_BASKET2).
        - order_depth: Current market depth.
        - fair_value: Market fair value.
        - clear_width: Distance threshold to trigger clearing orders.
        - position: Current net position.
        - buy_order_volume: Accumulated buy volume.
        - sell_order_volume: Accumulated sell volume.
        
        Returns:
        - A tuple containing the list of clearing orders along with updated volumes.
        """
        orders: List[Order] = []
        
        # Execute position clearing logic.
        buy_order_volume, sell_order_volume = self.clear_position_order_PICNIC_BASKET2(
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

    # -------------------------------------------------------------------------
    # Market Making with Dynamic Pricing for PICNIC_BASKET2
    # -------------------------------------------------------------------------

    def make_PICNIC_BASKET2_orders_PICNIC_BASKET2(
        self,
        order_depth: OrderDepth,
        fair_value: float,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        volume_limit: int,
        penny_val = 1,
        traderObject = None
    ) -> (List[Order], int, int):
        """
        Creates layered market making orders with configurable aggressiveness.
        """
        orders: List[Order] = []
        
        # Extract aggressiveness parameters
        aggressiveness = self.params[Product.PICNIC_BASKET2].get("aggressiveness", 0.5)
        fair_value_order_size = self.params[Product.PICNIC_BASKET2].get("fair_value_order_size", 5)
        tier1_distance = self.params[Product.PICNIC_BASKET2].get("tier1_distance", 1)
        tier2_distance = self.params[Product.PICNIC_BASKET2].get("tier2_distance", 2)
        tier3_distance = self.params[Product.PICNIC_BASKET2].get("tier3_distance", 3)
        tier1_volume_pct = self.params[Product.PICNIC_BASKET2].get("tier1_volume_pct", 0.5)
        tier2_volume_pct = self.params[Product.PICNIC_BASKET2].get("tier2_volume_pct", 0.3)
        tier3_volume_pct = self.params[Product.PICNIC_BASKET2].get("tier3_volume_pct", 0.2)
        position_scaling = self.params[Product.PICNIC_BASKET2].get("position_scaling", 0.2)
        
        # Calculate position ratio (-1 to 1 scale)
        position_ratio = position / self.LIMIT[Product.PICNIC_BASKET2]
        
        # Adjust distances based on aggressiveness
        # Higher aggressiveness = tighter spreads
        effective_tier1_distance = max(1, round(tier1_distance * (2 - aggressiveness)))
        effective_tier2_distance = max(2, round(tier2_distance * (2 - aggressiveness)))
        effective_tier3_distance = max(3, round(tier3_distance * (2 - aggressiveness)))
        
        # Calculate available quantities considering position limits
        available_buy_quantity = self.LIMIT[Product.PICNIC_BASKET2] - (position + buy_order_volume)
        available_sell_quantity = self.LIMIT[Product.PICNIC_BASKET2] + (position - sell_order_volume)
        
        # Adjust tier volumes based on position
        # When long, allocate more to sell side; when short, allocate more to buy side
        buy_tier1_pct = tier1_volume_pct - (position_ratio * position_scaling)
        buy_tier2_pct = tier2_volume_pct
        buy_tier3_pct = 1 - buy_tier1_pct - buy_tier2_pct
        
        sell_tier1_pct = tier1_volume_pct + (position_ratio * position_scaling)
        sell_tier2_pct = tier2_volume_pct
        sell_tier3_pct = 1 - sell_tier1_pct - sell_tier2_pct
        
        # Calculate fair value buy order size with dynamic scaling
        # More aggressive when short, less when long
        fair_value_buy_size = round(fair_value_order_size * (1 - position_ratio))
        fair_value_buy_size = min(fair_value_buy_size, available_buy_quantity)
        
        # Calculate fair value sell order size with dynamic scaling
        # More aggressive when long, less when short
        fair_value_sell_size = round(fair_value_order_size * (1 + position_ratio))
        fair_value_sell_size = min(fair_value_sell_size, available_sell_quantity)
        
        # --- BUY ORDERS ---
        remaining_buy_qty = available_buy_quantity
        
        # Super aggressive tier at fair value if aggressiveness > 0.7
        if aggressiveness > 0.7 and fair_value_buy_size > 0 and remaining_buy_qty > 0:
            actual_qty = min(fair_value_buy_size, remaining_buy_qty)
            orders.append(Order(Product.PICNIC_BASKET2, round(fair_value), actual_qty))
            remaining_buy_qty -= actual_qty
            buy_order_volume += actual_qty
        
        # Tier 1: Closest to fair value
        if remaining_buy_qty > 0:
            tier1_qty = round(available_buy_quantity * buy_tier1_pct)
            tier1_qty = min(tier1_qty, remaining_buy_qty)
            if tier1_qty > 0:
                orders.append(Order(Product.PICNIC_BASKET2, round(fair_value - effective_tier1_distance), tier1_qty))
                remaining_buy_qty -= tier1_qty
                buy_order_volume += tier1_qty
        
        # Tier 2: Middle distance from fair value
        if remaining_buy_qty > 0:
            tier2_qty = round(available_buy_quantity * buy_tier2_pct)
            tier2_qty = min(tier2_qty, remaining_buy_qty)
            if tier2_qty > 0:
                orders.append(Order(Product.PICNIC_BASKET2, round(fair_value - effective_tier2_distance), tier2_qty))
                remaining_buy_qty -= tier2_qty
                buy_order_volume += tier2_qty
        
        # Tier 3: Furthest from fair value
        if remaining_buy_qty > 0:
            orders.append(Order(Product.PICNIC_BASKET2, round(fair_value - effective_tier3_distance), remaining_buy_qty))
            buy_order_volume += remaining_buy_qty
        
        # --- SELL ORDERS ---
        remaining_sell_qty = available_sell_quantity
        
        # Super aggressive tier at fair value if aggressiveness > 0.7
        if aggressiveness > 0.7 and fair_value_sell_size > 0 and remaining_sell_qty > 0:
            actual_qty = min(fair_value_sell_size, remaining_sell_qty)
            orders.append(Order(Product.PICNIC_BASKET2, round(fair_value), -actual_qty))
            remaining_sell_qty -= actual_qty
            sell_order_volume += actual_qty
        
        # Tier 1: Closest to fair value
        if remaining_sell_qty > 0:
            tier1_qty = round(available_sell_quantity * sell_tier1_pct)
            tier1_qty = min(tier1_qty, remaining_sell_qty)
            if tier1_qty > 0:
                orders.append(Order(Product.PICNIC_BASKET2, round(fair_value + effective_tier1_distance), -tier1_qty))
                remaining_sell_qty -= tier1_qty
                sell_order_volume += tier1_qty
        
        # Tier 2: Middle distance from fair value
        if remaining_sell_qty > 0:
            tier2_qty = round(available_sell_quantity * sell_tier2_pct)
            tier2_qty = min(tier2_qty, remaining_sell_qty)
            if tier2_qty > 0:
                orders.append(Order(Product.PICNIC_BASKET2, round(fair_value + effective_tier2_distance), -tier2_qty))
                remaining_sell_qty -= tier2_qty
                sell_order_volume += tier2_qty
        
        # Tier 3: Furthest from fair value
        if remaining_sell_qty > 0:
            orders.append(Order(Product.PICNIC_BASKET2, round(fair_value + effective_tier3_distance), -remaining_sell_qty))
            sell_order_volume += remaining_sell_qty
        
        return orders, buy_order_volume, sell_order_volume
    
    # -------------------------------------------------------------------------
    # Main Run Method Called by Trading Environment
    # -------------------------------------------------------------------------
    def run(self, state: TradingState):
        """
        Main entry point of the trading strategy. This method:
         1. Restores any historical trader state from previous iterations.
         2. Checks if PICNIC_BASKET2 is available for trading.
         3. Dynamically computes fair value from live order book data.
         4. Adjusts parameters like take_width and soft_position_limit based on market volatility.
         5. Executes three phases: Opportunity Taking, Position Clearing, and Market Making.
         6. Serializes the updated trader state for the next iteration.
        
        Parameter:
        - state: The current state object containing order books, positions, and previously stored trader data.
        
        Returns:
        - A tuple containing the orders to be executed, any conversions (here, 0), and the updated trader state.
        """
        # Attempt to restore historical trader state if available.
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

        # Initialize results dictionary for storing orders by product.
        result = {}
        
        # Verify that PICNIC_BASKET2 is tradable in the current state.
        if Product.PICNIC_BASKET2 in self.params and Product.PICNIC_BASKET2 in state.order_depths:
            # Retrieve the current net position for PICNIC_BASKET2.
            PICNIC_BASKET2_position = (
                state.position[Product.PICNIC_BASKET2]
                if Product.PICNIC_BASKET2 in state.position
                else 0
            )
            #print(f"PICNIC_BASKET2 Position: {PICNIC_BASKET2_position}")
            # ----- STEP 1: Dynamic Fair Value Computation -----
            # Compute fair value using live VWAP data from the order book.
            fair_value = self.calculate_fair_value_PICNIC_BASKET2(
                Product.PICNIC_BASKET2,
                state.order_depths[Product.PICNIC_BASKET2],
                traderObject
            )
            
            # Retrieve the current observed spread if available (default to 3 otherwise).
            current_spread = traderObject.get(f"{Product.PICNIC_BASKET2}_current_spread", 3)
            
            # Adjust the take_width dynamically based on current market spread.
            # Ensures that when the market is narrow, the width does not drop below 2.
            dynamic_take_width = min(
                self.params[Product.PICNIC_BASKET2]["take_width"],
                current_spread - 1
            )
            
            # ----- Dynamic Position Limit Adjustment Based on Volatility -----
            # Use recent mid prices to compute rolling volatility.
            recent_prices = [data["mid_price"] for data in traderObject.get(f"{Product.PICNIC_BASKET2}_price_data", [])]
            if len(recent_prices) >= 10:
                # Standard deviation provides a measure of price volatility.
                rolling_stddev = np.std(recent_prices[-10:])
                # A higher volatility suggests tighter position limits to manage risk.
                volatility_factor = min(1.0, 7.5 / (rolling_stddev * 10))
                dynamic_soft_limit = int(self.params[Product.PICNIC_BASKET2]["soft_position_limit"] * volatility_factor)
                # Ensure that the dynamic soft limit does not fall below a conservative minimum.
                dynamic_soft_limit = max(20, dynamic_soft_limit)
            else:
                dynamic_soft_limit = self.params[Product.PICNIC_BASKET2]["soft_position_limit"]
            
            # Log key metrics for debugging purposes.
            #print(f"PICNIC_BASKET2 - Fair Value: {fair_value}, Position: {PICNIC_BASKET2_position}, Spread: {current_spread}")
            #print(f"Dynamic Params - Take Width: {dynamic_take_width}, Soft Limit: {dynamic_soft_limit}")
            
            
            # ----- STEP 2: Taking Orders (Opportunity Taking) -----
            PICNIC_BASKET2_take_orders, buy_order_volume, sell_order_volume = self.take_orders_PICNIC_BASKET2(
                Product.PICNIC_BASKET2,
                state.order_depths[Product.PICNIC_BASKET2],
                fair_value,
                dynamic_take_width,  # Use dynamically adjusted take width
                PICNIC_BASKET2_position,
            )
            
            # ----- STEP 3: Clearing Positions (Risk Management) -----
            PICNIC_BASKET2_clear_orders, buy_order_volume, sell_order_volume = self.clear_orders_PICNIC_BASKET2(
                Product.PICNIC_BASKET2,
                state.order_depths[Product.PICNIC_BASKET2],
                fair_value,
                self.params[Product.PICNIC_BASKET2]["clear_width"],
                PICNIC_BASKET2_position,
                buy_order_volume,
                sell_order_volume,
            )
            '''
            # ----- STEP 4: Market Making (Providing Liquidity) -----
            PICNIC_BASKET2_make_orders, _, _ = self.make_PICNIC_BASKET2_orders_PICNIC_BASKET2(
                state.order_depths[Product.PICNIC_BASKET2],
                fair_value,
                PICNIC_BASKET2_position,
                buy_order_volume,
                sell_order_volume,
                dynamic_soft_limit,  # Use dynamically computed soft limit for position sizing
                self.params[Product.PICNIC_BASKET2]["penny_val"],
            )
            '''
            
            # Combine all generated orders for PICNIC_BASKET2 into the result.
            result[Product.PICNIC_BASKET2] = (
                PICNIC_BASKET2_take_orders + PICNIC_BASKET2_clear_orders #+ PICNIC_BASKET2_make_orders
            )
            #print(f"PICNIC_BASKET2 Orders: {result[Product.PICNIC_BASKET2]}")
        # In this strategy, no currency or conversion orders are required.
        conversions = 0
        
        # Serialize updated trader state for use in the next trading iteration.
        traderData = jsonpickle.encode(traderObject)

        return result, conversions, traderData
