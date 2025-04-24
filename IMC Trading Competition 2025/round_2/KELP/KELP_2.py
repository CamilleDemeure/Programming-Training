from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict
import string
import jsonpickle
import numpy as np
import math


class Product:
    KELP = "KELP"

PARAMS = {
    Product.KELP: {
        "base_fair_value": 2033,      # Base fair value set to median of observed prices (~2033)
        "take_width": 3,              # Increased from 2 to 3 based on typical spread
        "clear_width": 1,             # Increased from 0.5 to 1 based on price volatility
        "disregard_edge": 1,          # Ignores orders for joining/pennying within this distance from fair value
        "join_edge": 2,               # Join orders within this edge rather than pennying
        "default_edge": 3,            # Reduced from 4 to 3 based on typical spread
        "soft_position_limit": 35,    # Increased from 25 to 35 to utilize more of the 50 position limit
        "volume_limit": 0,            # For position management
        "penny_val": 1,               # The value to adjust prices by when pennying
        "vwap_window": 20,            # Reduced from 25 to 20 for more responsive fair value
    },
}


class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params

        # Position Limit - Matches the position limit of 50
        self.LIMIT = {Product.KELP: 50}

    def calculate_fair_value(
        self,
        product: str,
        order_depth: OrderDepth,
        traderObject: Dict
    ) -> float:
        """
        Calculate a dynamic fair value based on VWAP of orderbook data
        """
        # Initialize fair value tracking if it doesn't exist
        if f"{product}_price_data" not in traderObject:
            traderObject[f"{product}_price_data"] = []
            traderObject[f"{product}_volume_data"] = []
            traderObject[f"{product}_fair_value"] = self.params[product]["base_fair_value"]
        
        # Get current timestamp data
        current_data = {}
        
        # Calculate VWAP for all available bid levels
        bid_prices = list(order_depth.buy_orders.keys())
        bid_volumes = [order_depth.buy_orders[price] for price in bid_prices]
        
        # Calculate VWAP for all available ask levels
        ask_prices = list(order_depth.sell_orders.keys())
        ask_volumes = [-order_depth.sell_orders[price] for price in ask_prices]  # Convert negative volumes to positive
        
        # Skip if there are no orders
        if not bid_prices or not ask_prices:
            return traderObject[f"{product}_fair_value"]
        
        # Calculate best bid and best ask and their volumes
        best_bid = max(bid_prices)
        best_ask = min(ask_prices)
        best_bid_volume = order_depth.buy_orders[best_bid]
        best_ask_volume = abs(order_depth.sell_orders[best_ask])  # Convert to positive
        
        # Calculate bid VWAP
        bid_vwap = sum(p * v for p, v in zip(bid_prices, bid_volumes)) / sum(bid_volumes) if sum(bid_volumes) > 0 else best_bid
        
        # Calculate ask VWAP
        ask_vwap = sum(p * v for p, v in zip(ask_prices, ask_volumes)) / sum(ask_volumes) if sum(ask_volumes) > 0 else best_ask
        
        # Calculate volume-weighted mid price
        weighted_mid_price = (best_bid*best_bid_volume + best_ask*best_ask_volume) / (best_bid_volume + best_ask_volume)
        
        # Store current price and volume data
        current_data["bid_vwap"] = bid_vwap
        current_data["ask_vwap"] = ask_vwap
        current_data["mid_price"] = weighted_mid_price
        
        # Add current data to history
        traderObject[f"{product}_price_data"].append(current_data)
        
        # Keep only the last window_size data points
        window_size = self.params[product]["vwap_window"]
        if len(traderObject[f"{product}_price_data"]) > window_size:
            traderObject[f"{product}_price_data"] = traderObject[f"{product}_price_data"][-window_size:]
        
        if len(traderObject[f"{product}_price_data"]) >= window_size:
            # Collect recent VWAP values (if you intend to use these averages, you might want to combine them,
            # for example, as the average of bid and ask VWAPs – or simply use the mid prices as you do below)
            recent_bid_vwaps = [data["bid_vwap"] for data in traderObject[f"{product}_price_data"]]
            recent_ask_vwaps = [data["ask_vwap"] for data in traderObject[f"{product}_price_data"]]
            
            # Option 1: Use VWAP averages (if desired)
            # avg_bid_vwap = sum(recent_bid_vwaps) / len(recent_bid_vwaps)
            # avg_ask_vwap = sum(recent_ask_vwaps) / len(recent_ask_vwaps)
            # fair_value = (avg_bid_vwap + avg_ask_vwap) / 2
            
            # Option 2: Use the average of the mid prices
            recent_mid_prices = [data["mid_price"] for data in traderObject[f"{product}_price_data"]]
            fair_value = sum(recent_mid_prices) / len(recent_mid_prices)
        else:
            # Not enough historical data, so simply use the current weighted mid-price
            fair_value = current_data["mid_price"]

        
        # Update stored fair value
        traderObject[f"{product}_fair_value"] = fair_value
        
        return fair_value

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
        position_limit = self.LIMIT[product]

        # More sophisticated order execution logic for sell orders (we buy)
        if len(order_depth.sell_orders) != 0:
            # Sort sell orders by price (lowest first) to prioritize most profitable orders
            sorted_sell_prices = sorted(order_depth.sell_orders.keys())
            
            for price in sorted_sell_prices:
                # Stop if price is no longer profitable
                if price > fair_value - take_width:
                    break
                
                # Calculate how much we can buy at this price level
                best_ask_amount = -1 * order_depth.sell_orders[price]
                quantity = min(best_ask_amount, position_limit - position - buy_order_volume)
                
                if quantity > 0:
                    # Create buy order at this price
                    orders.append(Order(product, price, quantity))
                    buy_order_volume += quantity
                    
                    # Update order depth to reflect our order
                    order_depth.sell_orders[price] += quantity
                    if order_depth.sell_orders[price] == 0:
                        del order_depth.sell_orders[price]

        # More sophisticated order execution logic for buy orders (we sell)
        if len(order_depth.buy_orders) != 0:
            # Sort buy orders by price (highest first) to prioritize most profitable orders
            sorted_buy_prices = sorted(order_depth.buy_orders.keys(), reverse=True)
            
            for price in sorted_buy_prices:
                # Stop if price is no longer profitable
                if price < fair_value + take_width:
                    break
                
                # Calculate how much we can sell at this price level
                best_bid_amount = order_depth.buy_orders[price]
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

    # Market making method
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
        # Calculate maximum buy quantity
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            # Place a bid (buy order)
            orders.append(Order(product, round(bid), buy_quantity))

        # Calculate maximum sell quantity
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            # Place an ask (sell order)
            orders.append(Order(product, round(ask), -sell_quantity))
            
        return buy_order_volume, sell_order_volume

    # Position clearing method
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
        fair_for_bid = round(fair_value - width)
        fair_for_ask = round(fair_value + width)

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

    # Wrapper method for taking orders
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

    # Wrapper method for clearing positions
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

    # Market making with dynamic pricing
    def make_kelp_orders(
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
        
        # Find price levels to join or penny
        # First, find all sell orders above fair value + threshold
        sells_above_threshold = [
            price for price in order_depth.sell_orders.keys()
            if price > fair_value + penny_val
        ]
        
        if not sells_above_threshold:
            # If no appropriate sell orders, use default edge
            baaf = fair_value + self.params[Product.KELP]["default_edge"]
        else:
            baaf = min(sells_above_threshold)
        
        # Find all buy orders below fair value - threshold
        buys_below_threshold = [
            price for price in order_depth.buy_orders.keys()
            if price < fair_value - penny_val
        ]
        
        if not buys_below_threshold:
            # If no appropriate buy orders, use default edge
            bbbf = fair_value - self.params[Product.KELP]["default_edge"]
        else:
            bbbf = max(buys_below_threshold)

        # Adjust pricing based on current position
        # If we're not position constrained on the buy side, we want better edge
        if baaf <= fair_value + 2:
            if position <= volume_limit:
                # Increase ask price to get better profit margin
                baaf = fair_value + 3

        # If we're not position constrained on the sell side, we want better edge
        if bbbf >= fair_value - 2:
            if position >= -volume_limit:
                # Decrease bid price to get better profit margin
                bbbf = fair_value - 3

        # Market make with adjusted prices
        buy_order_volume, sell_order_volume = self.market_make(
            Product.KELP,
            orders,
            bbbf + 1,  # Set bid price just above the best non-competitive bid
            baaf - 1,  # Set ask price just below the best non-competitive ask
            position,
            buy_order_volume,
            sell_order_volume,
        )
        
        return orders, buy_order_volume, sell_order_volume

    # Main method called by the trading environment
    def run(self, state: TradingState):
        # Deserialize trader state data if available
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

        # Initialize result dictionary to store orders for each product
        result = {}
        
        # Check if KELP is available to trade
        if Product.KELP in self.params and Product.KELP in state.order_depths:
            # Get current position in KELP
            kelp_position = (
                state.position[Product.KELP]
                if Product.KELP in state.position
                else 0
            )
            
            # STEP 1: Calculate dynamic fair value
            fair_value = self.calculate_fair_value(
                Product.KELP,
                state.order_depths[Product.KELP],
                traderObject
            )
            
            # Dynamically adjust trading parameters based on current market conditions
            current_spread = traderObject.get(f"{Product.KELP}_current_spread", 3)
            
            # Adjust take_width based on spread - use tighter width when spread is narrow
            dynamic_take_width = min(
                self.params[Product.KELP]["take_width"],
                max(2, current_spread - 1)  # Ensure take_width is at least 2
            )
            
            # Adjust position limit based on volatility
            recent_prices = [data["mid_price"] for data in traderObject.get(f"{Product.KELP}_price_data", [])]
            if len(recent_prices) >= 10:
                # Calculate rolling stddev of prices
                rolling_stddev = np.std(recent_prices[-10:])
                # Adjust soft position limit inversely with volatility
                # Higher volatility -> lower position limit
                volatility_factor = min(1.0, 7.5 / (rolling_stddev * 10))  # Based on typical stddev of ~0.75
                dynamic_soft_limit = int(self.params[Product.KELP]["soft_position_limit"] * volatility_factor)
                dynamic_soft_limit = max(20, dynamic_soft_limit)  # Never go below 20
            else:
                dynamic_soft_limit = self.params[Product.KELP]["soft_position_limit"]
            
            # Log important values for debugging
            print(f"KELP - Fair Value: {fair_value}, Position: {kelp_position}, Spread: {current_spread}")
            print(f"Dynamic Params - Take Width: {dynamic_take_width}, Soft Limit: {dynamic_soft_limit}")
            
            # STEP 2: Taking orders - exploit mispriced orders
            kelp_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(
                    Product.KELP,
                    state.order_depths[Product.KELP],
                    fair_value,
                    dynamic_take_width,  # Use dynamic width
                    kelp_position,
                )
            )
            
            # STEP 3: Clearing positions - reduce risk
            kelp_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(
                    Product.KELP,
                    state.order_depths[Product.KELP],
                    fair_value,
                    self.params[Product.KELP]["clear_width"],
                    kelp_position,
                    buy_order_volume,
                    sell_order_volume,
                )
            )
            
            # STEP 4: Market making - provide liquidity
            kelp_make_orders, _, _ = self.make_kelp_orders(
                state.order_depths[Product.KELP],
                fair_value,
                kelp_position,
                buy_order_volume,
                sell_order_volume,
                dynamic_soft_limit,  # Use dynamic soft limit
                self.params[Product.KELP]["penny_val"],
            )
            
            # Combine all orders for KELP
            result[Product.KELP] = (
                kelp_take_orders + kelp_clear_orders + kelp_make_orders
            )

        # No conversions needed for KELP
        conversions = 0
        
        # Serialize trader state data for next iteration
        traderData = jsonpickle.encode(traderObject)

        # Return orders, conversions, and trader state
        return result, conversions, traderData