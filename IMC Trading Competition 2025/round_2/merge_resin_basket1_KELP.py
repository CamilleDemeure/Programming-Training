from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict, Any
import string
import jsonpickle
import numpy as np
import math

class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    PICNIC_BASKET1 = "PICNIC_BASKET1"
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    DJEMBES = "DJEMBES"
    SYNTHETIC = "SYNTHETIC"  # Used internally to represent synthetic basket
    SPREAD = "SPREAD"        # Used internally to represent the basket-synthetic spread
    KELP = "KELP"

PARAMS = {
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000,          # The estimated fair price of RAINFOREST_RESIN
        "take_width": 1,              # How far from fair value we're willing to buy/sell immediately
        "clear_width": 0.5,           # Increased from 0 to match successful amethyst strategy - controls position clearing
        "disregard_edge": 1,          # Ignores orders for joining/pennying within this distance from fair value
        "join_edge": 2,               # Join orders within this edge rather than pennying
        "default_edge": 4,            # Default spread to use when no good reference prices exist
        "soft_position_limit": 45,    # Reduced position limit to be more conservative than hard limit
        "volume_limit": 0,            # Added for position management like in amethysts
        "penny_val": 1,               # The value to adjust prices by when pennying
    },

    Product.SPREAD: {
        "default_spread_mean": 37.441800,
        "default_spread_std": 85.148401,
        "spread_std_window": 55,
        "zscore_threshold": 3.3,
        "target_position": 58,
    },
    
    Product.KELP: {
        "base_fair_value": 2033,
        "take_width": 4,
        "clear_width": 1,
        "disregard_edge": 1,
        "join_edge": 2,
        "default_edge": 3,
        "soft_position_limit": 45,
        "volume_limit": 0,
        "penny_val": 1,
        "vwap_window": 50,
        # New aggressiveness parameters
        "aggressiveness": 1,           # 0.0 to 1.0, higher = more aggressive
        "fair_value_order_size": 0,      # Size of orders placed at fair value
        "tier1_distance": 1,             # Distance of first tier from fair value
        "tier2_distance": 2,             # Distance of second tier from fair value  
        "tier3_distance": 3,             # Distance of third tier from fair value
        "tier1_volume_pct": 0.75,         # Percentage of available volume for tier 1
        "tier2_volume_pct": 0.2,         # Percentage of available volume for tier 2
        "tier3_volume_pct": 0.05,         # Percentage of available volume for tier 3
        "position_scaling": 0,         # How much to adjust volume distribution based on position
    },
}

BASKET_WEIGHTS = {
    Product.CROISSANTS: 6,
    Product.JAMS: 3,
    Product.DJEMBES: 1,
}

class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params

        self.LIMIT = {
            Product.RAINFOREST_RESIN: 50,
            Product.PICNIC_BASKET1: 60,
            Product.CROISSANTS: 250,
            Product.JAMS: 350,
            Product.DJEMBES: 60,
            Product.KELP: 50
        }

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
    
    def get_swmid(self, order_depth) -> float:
        """
        Calculate the size-weighted midpoint price from an order book.
        
        This gives a more accurate price representation than a simple midpoint
        by accounting for the volume at each price level.
        
        Args:
            order_depth: The order book containing buy and sell orders
            
        Returns:
            float: The size-weighted midpoint price
        """
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        best_bid_vol = abs(order_depth.buy_orders[best_bid])
        best_ask_vol = abs(order_depth.sell_orders[best_ask])
        
        # Calculate weighted average of bid and ask prices based on their volumes
        return (best_bid * best_ask_vol + best_ask * best_bid_vol) / (
            best_bid_vol + best_ask_vol
        )

    def get_synthetic_basket_order_depth(
        self, order_depths: Dict[str, OrderDepth]
    ) -> OrderDepth:
        """
        Create a synthetic order book for a basket based on component order books.
        
        This method calculates what the implied prices and volumes would be if you 
        were to create or disassemble baskets using the individual components.
        
        Args:
            order_depths: Dictionary of order books for all products
            
        Returns:
            OrderDepth: A synthetic order book for the basket
        """
        # Get the component weights from the basket definition
        CROISSANTS_PER_BASKET = BASKET_WEIGHTS[Product.CROISSANTS]
        JAMS_PER_BASKET = BASKET_WEIGHTS[Product.JAMS]
        DJEMBES_PER_BASKET = BASKET_WEIGHTS[Product.DJEMBES]

        # Initialize the synthetic order book
        synthetic_order_price = OrderDepth()

        # Extract the best prices for each component
        # For buying components (to create a synthetic basket)
        CROISSANTS_best_bid = (
            max(order_depths[Product.CROISSANTS].buy_orders.keys())
            if order_depths[Product.CROISSANTS].buy_orders
            else 0
        )
        CROISSANTS_best_ask = (
            min(order_depths[Product.CROISSANTS].sell_orders.keys())
            if order_depths[Product.CROISSANTS].sell_orders
            else float("inf")
        )
        JAMS_best_bid = (
            max(order_depths[Product.JAMS].buy_orders.keys())
            if order_depths[Product.JAMS].buy_orders
            else 0
        )
        JAMS_best_ask = (
            min(order_depths[Product.JAMS].sell_orders.keys())
            if order_depths[Product.JAMS].sell_orders
            else float("inf")
        )
        DJEMBES_best_bid = (
            max(order_depths[Product.DJEMBES].buy_orders.keys())
            if order_depths[Product.DJEMBES].buy_orders
            else 0
        )
        DJEMBES_best_ask = (
            min(order_depths[Product.DJEMBES].sell_orders.keys())
            if order_depths[Product.DJEMBES].sell_orders
            else float("inf")
        )

        # Calculate the implied prices for buying and selling a synthetic basket
        # Synthetic bid price = what you can sell components for
        implied_bid = (
            CROISSANTS_best_bid * CROISSANTS_PER_BASKET
            + JAMS_best_bid * JAMS_PER_BASKET
            + DJEMBES_best_bid * DJEMBES_PER_BASKET
        )
        # Synthetic ask price = what you must pay to buy components
        implied_ask = (
            CROISSANTS_best_ask * CROISSANTS_PER_BASKET
            + JAMS_best_ask * JAMS_PER_BASKET
            + DJEMBES_best_ask * DJEMBES_PER_BASKET
        )

        # Calculate the maximum quantity available at these prices
        if implied_bid > 0:
            # How many baskets can we create from component buy orders?
            CROISSANTS_bid_volume = (
                order_depths[Product.CROISSANTS].buy_orders[CROISSANTS_best_bid]
                // CROISSANTS_PER_BASKET
            )
            JAMS_bid_volume = (
                order_depths[Product.JAMS].buy_orders[JAMS_best_bid]
                // JAMS_PER_BASKET
            )
            DJEMBES_bid_volume = (
                order_depths[Product.DJEMBES].buy_orders[DJEMBES_best_bid]
                // DJEMBES_PER_BASKET
            )
            # We are limited by the component with the least available volume
            implied_bid_volume = min(
                CROISSANTS_bid_volume, JAMS_bid_volume, DJEMBES_bid_volume
            )
            synthetic_order_price.buy_orders[implied_bid] = implied_bid_volume

        if implied_ask < float("inf"):
            # How many baskets can we break down from component sell orders?
            CROISSANTS_ask_volume = (
                -order_depths[Product.CROISSANTS].sell_orders[CROISSANTS_best_ask]
                // CROISSANTS_PER_BASKET
            )
            JAMS_ask_volume = (
                -order_depths[Product.JAMS].sell_orders[JAMS_best_ask]
                // JAMS_PER_BASKET
            )
            DJEMBES_ask_volume = (
                -order_depths[Product.DJEMBES].sell_orders[DJEMBES_best_ask]
                // DJEMBES_PER_BASKET
            )
            # We are limited by the component with the least available volume
            implied_ask_volume = min(
                CROISSANTS_ask_volume, JAMS_ask_volume, DJEMBES_ask_volume
            )
            synthetic_order_price.sell_orders[implied_ask] = -implied_ask_volume

        return synthetic_order_price

    def convert_synthetic_basket_orders(
        self, synthetic_orders: List[Order], order_depths: Dict[str, OrderDepth]
    ) -> Dict[str, List[Order]]:
        """
        Convert orders for a synthetic basket into equivalent component orders.
        
        This breaks down basket orders into the correct proportions of components.
        
        Args:
            synthetic_orders: List of orders for the synthetic basket
            order_depths: Dictionary of order books for all products
            
        Returns:
            Dict[str, List[Order]]: Dictionary mapping each component to its orders
        """
        # Initialize the dictionary to store orders for each component
        component_orders = {
            Product.CROISSANTS: [],
            Product.JAMS: [],
            Product.DJEMBES: [],
        }

        # Get the current synthetic basket prices
        synthetic_basket_order_depth = self.get_synthetic_basket_order_depth(
            order_depths
        )
        best_bid = (
            max(synthetic_basket_order_depth.buy_orders.keys())
            if synthetic_basket_order_depth.buy_orders
            else 0
        )
        best_ask = (
            min(synthetic_basket_order_depth.sell_orders.keys())
            if synthetic_basket_order_depth.sell_orders
            else float("inf")
        )

        # Process each synthetic basket order
        for order in synthetic_orders:
            # Extract order details
            price = order.price
            quantity = order.quantity

            # Determine appropriate prices for component orders
            if quantity > 0 and price >= best_ask:
                # When buying synthetic basket, we need to buy components at ask prices
                CROISSANTS_price = min(
                    order_depths[Product.CROISSANTS].sell_orders.keys()
                )
                JAMS_price = min(
                    order_depths[Product.JAMS].sell_orders.keys()
                )
                DJEMBES_price = min(order_depths[Product.DJEMBES].sell_orders.keys())
            elif quantity < 0 and price <= best_bid:
                # When selling synthetic basket, we need to sell components at bid prices
                CROISSANTS_price = max(order_depths[Product.CROISSANTS].buy_orders.keys())
                JAMS_price = max(
                    order_depths[Product.JAMS].buy_orders.keys()
                )
                DJEMBES_price = max(order_depths[Product.DJEMBES].buy_orders.keys())
            else:
                # If order price doesn't match current market, skip it
                continue

            # Create component orders with the correct weights
            CROISSANTS_order = Order(
                Product.CROISSANTS,
                CROISSANTS_price,
                quantity * BASKET_WEIGHTS[Product.CROISSANTS],
            )
            JAMS_order = Order(
                Product.JAMS,
                JAMS_price,
                quantity * BASKET_WEIGHTS[Product.JAMS],
            )
            DJEMBES_order = Order(
                Product.DJEMBES, 
                DJEMBES_price, 
                quantity * BASKET_WEIGHTS[Product.DJEMBES]
            )

            # Add orders to their respective lists
            component_orders[Product.CROISSANTS].append(CROISSANTS_order)
            component_orders[Product.JAMS].append(JAMS_order)
            component_orders[Product.DJEMBES].append(DJEMBES_order)

        return component_orders

    def execute_spread_orders(
        self,
        target_position: int,
        basket_position: int,
        order_depths: Dict[str, OrderDepth],
    ):
        """
        Execute trades to move toward the target basket position.
        
        This method calculates and places the orders needed to reach the desired
        basket position while maintaining the correct component ratios.
        
        Args:
            target_position: The desired position in PICNIC_BASKET1
            basket_position: The current position in PICNIC_BASKET1
            order_depths: Dictionary of order books for all products
            
        Returns:
            Dict or None: Dictionary of orders for each product, or None if no action needed
        """
        # If we're already at the target position, do nothing
        if target_position == basket_position:
            return None

        # Calculate how many baskets we need to buy or sell
        target_quantity = abs(target_position - basket_position)
        
        # Get order books for the real basket and synthetic basket
        basket_order_depth = order_depths[Product.PICNIC_BASKET1]
        synthetic_order_depth = self.get_synthetic_basket_order_depth(order_depths)

        # If we need to increase our position (buy baskets)
        if target_position > basket_position:
            # Find best prices and volumes for buying baskets and selling components
            basket_ask_price = min(basket_order_depth.sell_orders.keys())
            basket_ask_volume = abs(basket_order_depth.sell_orders[basket_ask_price])

            synthetic_bid_price = max(synthetic_order_depth.buy_orders.keys())
            synthetic_bid_volume = abs(
                synthetic_order_depth.buy_orders[synthetic_bid_price]
            )

            # Determine the maximum volume we can trade
            orderbook_volume = min(basket_ask_volume, synthetic_bid_volume)
            execute_volume = min(orderbook_volume, target_quantity)

            # Create orders for the basket and synthetic components
            basket_orders = [
                Order(Product.PICNIC_BASKET1, basket_ask_price, execute_volume)
            ]
            synthetic_orders = [
                Order(Product.SYNTHETIC, synthetic_bid_price, -execute_volume)
            ]

            # Convert the synthetic orders to actual component orders
            aggregate_orders = self.convert_synthetic_basket_orders(
                synthetic_orders, order_depths
            )
            aggregate_orders[Product.PICNIC_BASKET1] = basket_orders
            return aggregate_orders

        # If we need to decrease our position (sell baskets)
        else:
            # Find best prices and volumes for selling baskets and buying components
            basket_bid_price = max(basket_order_depth.buy_orders.keys())
            basket_bid_volume = abs(basket_order_depth.buy_orders[basket_bid_price])

            synthetic_ask_price = min(synthetic_order_depth.sell_orders.keys())
            synthetic_ask_volume = abs(
                synthetic_order_depth.sell_orders[synthetic_ask_price]
            )

            # Determine the maximum volume we can trade
            orderbook_volume = min(basket_bid_volume, synthetic_ask_volume)
            execute_volume = min(orderbook_volume, target_quantity)

            # Create orders for the basket and synthetic components
            basket_orders = [
                Order(Product.PICNIC_BASKET1, basket_bid_price, -execute_volume)
            ]
            synthetic_orders = [
                Order(Product.SYNTHETIC, synthetic_ask_price, execute_volume)
            ]

            # Convert the synthetic orders to actual component orders
            aggregate_orders = self.convert_synthetic_basket_orders(
                synthetic_orders, order_depths
            )
            aggregate_orders[Product.PICNIC_BASKET1] = basket_orders
            return aggregate_orders

    def spread_orders(
        self,
        order_depths: Dict[str, OrderDepth],
        product: Product,
        basket_position: int,
        spread_data: Dict[str, Any],
    ):
        """
        Main strategy method that monitors spread and executes trades when appropriate.
        
        This method calculates the spread between the basket and synthetic prices,
        tracks historical spread data, and triggers trades when statistical
        thresholds are crossed.
        
        Args:
            order_depths: Dictionary of order books for all products
            product: The basket product being traded
            basket_position: Current position in the basket
            spread_data: Dictionary tracking historical spread information
            
        Returns:
            Dict or None: Dictionary of orders for each product, or None if no action needed
        """
        # Verify that the basket product is available to trade
        if Product.PICNIC_BASKET1 not in order_depths.keys():
            return None

        # Get order books for the basket and calculate synthetic basket
        basket_order_depth = order_depths[Product.PICNIC_BASKET1]
        synthetic_order_depth = self.get_synthetic_basket_order_depth(order_depths)
        
        # Calculate size-weighted mid prices for both
        basket_swmid = self.get_swmid(basket_order_depth)
        synthetic_swmid = self.get_swmid(synthetic_order_depth)
        
        # Calculate the current spread and add to history
        spread = basket_swmid - synthetic_swmid
        spread_data["spread_history"].append(spread)

        # Wait until we have enough data for statistical analysis
        if (
            len(spread_data["spread_history"])
            < self.params[Product.SPREAD]["spread_std_window"]
        ):
            return None
        # Keep the history at a fixed length
        elif len(spread_data["spread_history"]) > self.params[Product.SPREAD]["spread_std_window"]:
            spread_data["spread_history"].pop(0)

        # Calculate the standard deviation of the spread
        spread_std = np.std(spread_data["spread_history"])

        # Calculate z-score: how many standard deviations from the mean
        zscore = (
            spread - self.params[Product.SPREAD]["default_spread_mean"]
        ) / spread_std

        # If spread is significantly above average (basket expensive, synthetic cheap)
        if zscore >= self.params[Product.SPREAD]["zscore_threshold"]:
            # Check if we need to adjust position
            if basket_position != -self.params[Product.SPREAD]["target_position"]:
                # Execute trades to sell basket and buy components
                return self.execute_spread_orders(
                    -self.params[Product.SPREAD]["target_position"],
                    basket_position,
                    order_depths,
                )

        # If spread is significantly below average (basket cheap, synthetic expensive)
        if zscore <= -self.params[Product.SPREAD]["zscore_threshold"]:
            # Check if we need to adjust position
            if basket_position != self.params[Product.SPREAD]["target_position"]:
                # Execute trades to buy basket and sell components
                return self.execute_spread_orders(
                    self.params[Product.SPREAD]["target_position"],
                    basket_position,
                    order_depths,
                )

        # Store the current z-score for reference
        spread_data["prev_zscore"] = zscore
        return None
    
    ###############################################################################
    #######  KELP
    ###############################################################################
    
    # -------------------------------------------------------------------------
    # Dynamic Fair Value Calculation
    # -------------------------------------------------------------------------
    def calculate_fair_value_KELP(
        self,
        product: str,
        order_depth: OrderDepth,
        traderObject: Dict
    ) -> float:
        """
        Calculate a dynamic fair value based on the volume-weighted average price (VWAP)
        of bid and ask orders.
        
        Parameters:
        - product: The trading instrument (here, KELP).
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
    def take_best_orders_KELP(
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
        - product: Trading instrument (KELP).
        - fair_value: Current calculated fair value.
        - take_width: Threshold to determine if an order is priced attractively.
        - orders: List to accumulate new orders.
        - order_depth: Current market order book.
        - position: Current net position in KELP.
        - buy_order_volume: Running total of KELP already being bought this round.
        - sell_order_volume: Running total of KELP already being sold this round.
        
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
    def market_make_KELP(
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
        - product: Trading instrument (KELP).
        - orders: List where new market making orders are appended.
        - bid: Bid price to post for buying.
        - ask: Ask price to post for selling.
        - position: Current net position of KELP.
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
    def clear_position_order_KELP(
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
        - product: Trading instrument (KELP).
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
    def take_orders_KELP(
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
        - product: Trading instrument (KELP).
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
        buy_order_volume, sell_order_volume = self.take_best_orders_KELP(
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

    def clear_orders_KELP(
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
        - product: Trading instrument (KELP).
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
        buy_order_volume, sell_order_volume = self.clear_position_order_KELP(
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
    # Market Making with Dynamic Pricing for KELP
    # -------------------------------------------------------------------------

    def make_kelp_orders_KELP(
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
        aggressiveness = self.params[Product.KELP].get("aggressiveness", 0.5)
        fair_value_order_size = self.params[Product.KELP].get("fair_value_order_size", 5)
        tier1_distance = self.params[Product.KELP].get("tier1_distance", 1)
        tier2_distance = self.params[Product.KELP].get("tier2_distance", 2)
        tier3_distance = self.params[Product.KELP].get("tier3_distance", 3)
        tier1_volume_pct = self.params[Product.KELP].get("tier1_volume_pct", 0.5)
        tier2_volume_pct = self.params[Product.KELP].get("tier2_volume_pct", 0.3)
        tier3_volume_pct = self.params[Product.KELP].get("tier3_volume_pct", 0.2)
        position_scaling = self.params[Product.KELP].get("position_scaling", 0.2)
        
        # Calculate position ratio (-1 to 1 scale)
        position_ratio = position / self.LIMIT[Product.KELP]
        
        # Adjust distances based on aggressiveness
        # Higher aggressiveness = tighter spreads
        effective_tier1_distance = max(1, round(tier1_distance * (2 - aggressiveness)))
        effective_tier2_distance = max(2, round(tier2_distance * (2 - aggressiveness)))
        effective_tier3_distance = max(3, round(tier3_distance * (2 - aggressiveness)))
        
        # Calculate available quantities considering position limits
        available_buy_quantity = self.LIMIT[Product.KELP] - (position + buy_order_volume)
        available_sell_quantity = self.LIMIT[Product.KELP] + (position - sell_order_volume)
        
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
            orders.append(Order(Product.KELP, round(fair_value), actual_qty))
            remaining_buy_qty -= actual_qty
            buy_order_volume += actual_qty
        
        # Tier 1: Closest to fair value
        if remaining_buy_qty > 0:
            tier1_qty = round(available_buy_quantity * buy_tier1_pct)
            tier1_qty = min(tier1_qty, remaining_buy_qty)
            if tier1_qty > 0:
                orders.append(Order(Product.KELP, round(fair_value - effective_tier1_distance), tier1_qty))
                remaining_buy_qty -= tier1_qty
                buy_order_volume += tier1_qty
        
        # Tier 2: Middle distance from fair value
        if remaining_buy_qty > 0:
            tier2_qty = round(available_buy_quantity * buy_tier2_pct)
            tier2_qty = min(tier2_qty, remaining_buy_qty)
            if tier2_qty > 0:
                orders.append(Order(Product.KELP, round(fair_value - effective_tier2_distance), tier2_qty))
                remaining_buy_qty -= tier2_qty
                buy_order_volume += tier2_qty
        
        # Tier 3: Furthest from fair value
        if remaining_buy_qty > 0:
            orders.append(Order(Product.KELP, round(fair_value - effective_tier3_distance), remaining_buy_qty))
            buy_order_volume += remaining_buy_qty
        
        # --- SELL ORDERS ---
        remaining_sell_qty = available_sell_quantity
        
        # Super aggressive tier at fair value if aggressiveness > 0.7
        if aggressiveness > 0.7 and fair_value_sell_size > 0 and remaining_sell_qty > 0:
            actual_qty = min(fair_value_sell_size, remaining_sell_qty)
            orders.append(Order(Product.KELP, round(fair_value), -actual_qty))
            remaining_sell_qty -= actual_qty
            sell_order_volume += actual_qty
        
        # Tier 1: Closest to fair value
        if remaining_sell_qty > 0:
            tier1_qty = round(available_sell_quantity * sell_tier1_pct)
            tier1_qty = min(tier1_qty, remaining_sell_qty)
            if tier1_qty > 0:
                orders.append(Order(Product.KELP, round(fair_value + effective_tier1_distance), -tier1_qty))
                remaining_sell_qty -= tier1_qty
                sell_order_volume += tier1_qty
        
        # Tier 2: Middle distance from fair value
        if remaining_sell_qty > 0:
            tier2_qty = round(available_sell_quantity * sell_tier2_pct)
            tier2_qty = min(tier2_qty, remaining_sell_qty)
            if tier2_qty > 0:
                orders.append(Order(Product.KELP, round(fair_value + effective_tier2_distance), -tier2_qty))
                remaining_sell_qty -= tier2_qty
                sell_order_volume += tier2_qty
        
        # Tier 3: Furthest from fair value
        if remaining_sell_qty > 0:
            orders.append(Order(Product.KELP, round(fair_value + effective_tier3_distance), -remaining_sell_qty))
            sell_order_volume += remaining_sell_qty
        
        return orders, buy_order_volume, sell_order_volume
    
    
    
    
    def run(self, state: TradingState):
        # Deserialize trader state data if available
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

        # Initialize result dictionary to store orders for each product
        result = {}
        conversions = 0

        # Verify that KELP is tradable in the current state.
        if Product.KELP in self.params and Product.KELP in state.order_depths:
            # Retrieve the current net position for KELP.
            kelp_position = (
                state.position[Product.KELP]
                if Product.KELP in state.position
                else 0
            )
            #print(f"KELP Position: {kelp_position}")
            # ----- STEP 1: Dynamic Fair Value Computation -----
            # Compute fair value using live VWAP data from the order book.
            fair_value = self.calculate_fair_value_KELP(
                Product.KELP,
                state.order_depths[Product.KELP],
                traderObject
            )
            
            # Retrieve the current observed spread if available (default to 3 otherwise).
            current_spread = traderObject.get(f"{Product.KELP}_current_spread", 3)
            
            # Adjust the take_width dynamically based on current market spread.
            # Ensures that when the market is narrow, the width does not drop below 2.
            dynamic_take_width = min(
                self.params[Product.KELP]["take_width"],
                current_spread - 1
            )
            
            # ----- Dynamic Position Limit Adjustment Based on Volatility -----
            # Use recent mid prices to compute rolling volatility.
            recent_prices = [data["mid_price"] for data in traderObject.get(f"{Product.KELP}_price_data", [])]
            if len(recent_prices) >= 10:
                # Standard deviation provides a measure of price volatility.
                rolling_stddev = np.std(recent_prices[-10:])
                # A higher volatility suggests tighter position limits to manage risk.
                volatility_factor = min(1.0, 7.5 / (rolling_stddev * 10))
                dynamic_soft_limit = int(self.params[Product.KELP]["soft_position_limit"] * volatility_factor)
                # Ensure that the dynamic soft limit does not fall below a conservative minimum.
                dynamic_soft_limit = max(20, dynamic_soft_limit)
            else:
                dynamic_soft_limit = self.params[Product.KELP]["soft_position_limit"]
            
            # Log key metrics for debugging purposes.
            #print(f"KELP - Fair Value: {fair_value}, Position: {kelp_position}, Spread: {current_spread}")
            #print(f"Dynamic Params - Take Width: {dynamic_take_width}, Soft Limit: {dynamic_soft_limit}")
            
            
            # ----- STEP 2: Taking Orders (Opportunity Taking) -----
            kelp_take_orders_KELP, buy_order_volume, sell_order_volume = self.take_orders_KELP(
                Product.KELP,
                state.order_depths[Product.KELP],
                fair_value,
                dynamic_take_width,  # Use dynamically adjusted take width
                kelp_position,
            )
            
            # ----- STEP 3: Clearing Positions (Risk Management) -----
            kelp_clear_orders_KELP, buy_order_volume, sell_order_volume = self.clear_orders_KELP(
                Product.KELP,
                state.order_depths[Product.KELP],
                fair_value,
                self.params[Product.KELP]["clear_width"],
                kelp_position,
                buy_order_volume,
                sell_order_volume,
            )
            
            # ----- STEP 4: Market Making (Providing Liquidity) -----
            kelp_make_orders, _, _ = self.make_kelp_orders_KELP(
                state.order_depths[Product.KELP],
                fair_value,
                kelp_position,
                buy_order_volume,
                sell_order_volume,
                dynamic_soft_limit,  # Use dynamically computed soft limit for position sizing
                self.params[Product.KELP]["penny_val"],
            )
            
            # Combine all generated orders for KELP into the result.
            result[Product.KELP] = (
                kelp_take_orders_KELP + kelp_clear_orders_KELP + kelp_make_orders
            )
            #print(f"KELP Orders: {result[Product.KELP]}")
        
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

        # Initialize spread tracking data structure if it doesn't exist
        if Product.SPREAD not in traderObject:
            traderObject[Product.SPREAD] = {
                "spread_history": [],     # List to track historical spread values
                "prev_zscore": 0,         # Previous z-score for reference
                "clear_flag": False,      # Flag to indicate if we're in clearing mode
                "curr_avg": 0,            # Current average spread
            }

        # Get current basket position
        basket_position = (
            state.position[Product.PICNIC_BASKET1]
            if Product.PICNIC_BASKET1 in state.position
            else 0
        )
        
        # Run the spread trading strategy
        spread_orders = self.spread_orders(
            state.order_depths,
            Product.PICNIC_BASKET1,
            basket_position,
            traderObject[Product.SPREAD],
        )
        
        # If we have orders to execute, add them to the result
        if spread_orders != None:
            result[Product.CROISSANTS] = spread_orders[Product.CROISSANTS]
            result[Product.JAMS] = spread_orders[Product.JAMS]
            result[Product.DJEMBES] = spread_orders[Product.DJEMBES]
            result[Product.PICNIC_BASKET1] = spread_orders[Product.PICNIC_BASKET1]

        # Save updated trader data for next iteration
        traderData = jsonpickle.encode(traderObject)

        return result, conversions, traderData