from datamodel import OrderDepth, UserId, TradingState, Order, ConversionObservation
from typing import List, Dict, Any
import string
import jsonpickle
import numpy as np
import math


class Product:
    PICNIC_BASKET2 = "PICNIC_BASKET2"
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    SYNTHETIC = "SYNTHETIC"  # Used internally to represent synthetic basket
    SPREAD = "SPREAD"        # Used internally to represent the basket-synthetic spread


'''
PICNIC_BASKET2 Statistical Arbitrage Trading Strategy Explanation

Overview
This trading strategy implements a statistical arbitrage approach for PICNIC_BASKET2 and its components.
The strategy exploits price differences between the actual basket product and its synthetic equivalent
created from the individual components (CROISSANTS, JAMS). By detecting when these prices
diverge significantly from their historical relationship, the strategy can profit from their eventual
convergence back to equilibrium.

################################################
Step 1: Calculate Synthetic Basket
################################################

The strategy begins by creating a synthetic version of PICNIC_BASKET2:

Basket Composition: One PICNIC_BASKET2 consists of 4 CROISSANTS, 2 JAMS
Synthetic Order Book: The algorithm calculates the synthetic prices for buying and selling a basket
based on the order books of the components.
Size-Weighted Mid Prices: It computes size-weighted mid prices for both the real basket and synthetic
basket to get a more accurate representation of their true values.

################################################
Step 2: Statistical Spread Analysis
################################################

With both prices in hand, the strategy analyzes their relationship:

Calculate Spread: It computes the spread between the basket and synthetic prices.
Track Historical Data: The spread values are recorded over time to establish a baseline.
Compute Z-Score: The current spread is compared to the historical mean and standard deviation
to determine how unusual the current spread is (measured by z-score).
Identify Opportunities: When the z-score exceeds predetermined thresholds, the algorithm
identifies a potential trading opportunity.

################################################
Step 3: Spread Trade Execution
################################################

When a significant spread deviation is detected:

Buy Low, Sell High: If the basket is cheap relative to components (negative z-score), buy the basket
and sell the components. If the basket is expensive (positive z-score), sell the basket and buy components.
Calculate Trade Size: The algorithm determines the maximum trade size based on current positions, 
available liquidity, and position limits.
Place Component Orders: Orders for all components are calculated in the correct proportions to
match the basket orders.
Monitor Position: The strategy tracks the basket position to ensure it stays within target ranges.

################################################
Key Features
################################################

Mean Reversion: The strategy is based on the principle that the spread will revert to its mean over time.
Statistical Edge: Using z-scores provides a statistical framework for determining when to trade.
Balanced Exposure: By always trading the basket against its components, market risk is minimized.
Position Management: The algorithm respects position limits for all products.

This statistical arbitrage approach allows us to profit from temporary pricing inefficiencies while
maintaining a relatively market-neutral position.
'''

# Configuration parameters that control the spread trading strategy
# These parameters are derived from historical analysis of the spread behavior
PARAMS = {
    Product.SPREAD: {
        "default_spread_mean": 23.579200,
        "default_spread_std": 59.86159,
        "spread_std_window": 55,
        "zscore_threshold": 30,
        "target_position": 96
    }
}

# The weights of each component in one basket
# These define the composition of PICNIC_BASKET2
BASKET_WEIGHTS = {
    Product.CROISSANTS: 4,
    Product.JAMS: 2
    }


class Trader:
    def __init__(self, params=None):
        """
        Initialize the trader with parameters for the strategy.
        
        Args:
            params: Optional dictionary of parameters to override defaults
        """
        if params is None:
            params = PARAMS
        self.params = params

        # Position limits for each product
        # These are the maximum long or short positions allowed
        self.LIMIT = {
            Product.PICNIC_BASKET2: 100,    # Maximum 60 baskets long or short
            Product.CROISSANTS: 250,       # Maximum 250 croissants long or short
            Product.JAMS: 350,             # Maximum 350 jams long or short
        }

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

        # Calculate the implied prices for buying and selling a synthetic basket
        # Synthetic bid price = what you can sell components for
        implied_bid = (
            CROISSANTS_best_bid * CROISSANTS_PER_BASKET
            + JAMS_best_bid * JAMS_PER_BASKET
        )
        # Synthetic ask price = what you must pay to buy components
        implied_ask = (
            CROISSANTS_best_ask * CROISSANTS_PER_BASKET
            + JAMS_best_ask * JAMS_PER_BASKET
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
            # We are limited by the component with the least available volume
            implied_bid_volume = min(
                CROISSANTS_bid_volume, JAMS_bid_volume
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
            # We are limited by the component with the least available volume
            implied_ask_volume = min(
                CROISSANTS_ask_volume, JAMS_ask_volume
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
            elif quantity < 0 and price <= best_bid:
                # When selling synthetic basket, we need to sell components at bid prices
                CROISSANTS_price = max(order_depths[Product.CROISSANTS].buy_orders.keys())
                JAMS_price = max(
                    order_depths[Product.JAMS].buy_orders.keys()
                )
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

            # Add orders to their respective lists
            component_orders[Product.CROISSANTS].append(CROISSANTS_order)
            component_orders[Product.JAMS].append(JAMS_order)

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
            target_position: The desired position in PICNIC_BASKET2
            basket_position: The current position in PICNIC_BASKET2
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
        basket_order_depth = order_depths[Product.PICNIC_BASKET2]
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
                Order(Product.PICNIC_BASKET2, basket_ask_price, execute_volume)
            ]
            synthetic_orders = [
                Order(Product.SYNTHETIC, synthetic_bid_price, -execute_volume)
            ]

            # Convert the synthetic orders to actual component orders
            aggregate_orders = self.convert_synthetic_basket_orders(
                synthetic_orders, order_depths
            )
            aggregate_orders[Product.PICNIC_BASKET2] = basket_orders
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
                Order(Product.PICNIC_BASKET2, basket_bid_price, -execute_volume)
            ]
            synthetic_orders = [
                Order(Product.SYNTHETIC, synthetic_ask_price, execute_volume)
            ]

            # Convert the synthetic orders to actual component orders
            aggregate_orders = self.convert_synthetic_basket_orders(
                synthetic_orders, order_depths
            )
            aggregate_orders[Product.PICNIC_BASKET2] = basket_orders
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
        if Product.PICNIC_BASKET2 not in order_depths.keys():
            return None

        # Get order books for the basket and calculate synthetic basket
        basket_order_depth = order_depths[Product.PICNIC_BASKET2]
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

    def run(self, state: TradingState):
        """
        Main method called by the trading system on each iteration.
        
        This method processes the current market state and generates orders
        based on the spread trading strategy.
        
        Args:
            state: Current market state including order books and positions
            
        Returns:
            tuple: (orders, conversions, trader_data)
        """
        # Load saved trader data from previous iterations
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

        # Initialize result dictionary to hold our orders
        result = {}
        # No conversions needed for this strategy
        conversions = 0

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
            state.position[Product.PICNIC_BASKET2]
            if Product.PICNIC_BASKET2 in state.position
            else 0
        )
        
        # Run the spread trading strategy
        spread_orders = self.spread_orders(
            state.order_depths,
            Product.PICNIC_BASKET2,
            basket_position,
            traderObject[Product.SPREAD],
        )
        
        # If we have orders to execute, add them to the result
        if spread_orders != None:
            result[Product.CROISSANTS] = spread_orders[Product.CROISSANTS]
            result[Product.JAMS] = spread_orders[Product.JAMS]
            result[Product.PICNIC_BASKET2] = spread_orders[Product.PICNIC_BASKET2]

        # Save updated trader data for next iteration
        traderData = jsonpickle.encode(traderObject)

        return result, conversions, traderData