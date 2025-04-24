from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict
import string
import jsonpickle
import numpy as np
import math

class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"

PARAMS = {
    Product.RAINFOREST_RESIN: {
        "fair_value_initial": 10000,  # Initial fair value, will be adjusted dynamically
        "take_width": 1,
        "clear_width": 0,
        "disregard_edge": 1,  # disregards orders for joining or pennying within this value from fair
        "join_edge": 2,  # joins orders within this edge
        "default_edge": 4,
        "soft_position_limit": 25,  # Increased from 10 to align with new limit
        "ema_alpha": 0.75,  # Weight for EMA calculation (higher = more weight to recent prices)
        "price_history_max": 1000,  # Maximum number of prices to store in history
        "vwap_lookback": 1000,  # Number of trades to include in VWAP calculation
    },
}

class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params

        # Updated position limit to 50 as per the provided context
        self.LIMIT = {Product.RAINFOREST_RESIN: 50}

    def calculate_fair_value(
        self,
        product: str,
        order_depth: OrderDepth,
        market_trades: List,
        own_trades: List,
        traderObject: Dict
    ) -> float:
        """
        Calculate a dynamic fair value based on market data and trading history
        """
        # Initialize fair value history if it doesn't exist
        if f"{product}_fair_value_history" not in traderObject:
            traderObject[f"{product}_fair_value_history"] = []
            traderObject[f"{product}_last_fair_value"] = self.params[product]["fair_value_initial"]
            traderObject[f"{product}_trade_prices"] = []
            traderObject[f"{product}_trade_volumes"] = []
        
        # Get previous fair value
        prev_fair_value = traderObject[f"{product}_last_fair_value"]
        
        # Get mid price from order book if available
        mid_price = None
        if order_depth.buy_orders and order_depth.sell_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            mid_price = (best_bid + best_ask) / 2
        
        # Record all new trade prices and volumes
        if market_trades:
            for trade in market_trades:
                traderObject[f"{product}_trade_prices"].append(trade.price)
                traderObject[f"{product}_trade_volumes"].append(trade.quantity)
                
                # Keep history within limits
                if len(traderObject[f"{product}_trade_prices"]) > self.params[product]["price_history_max"]:
                    traderObject[f"{product}_trade_prices"].pop(0)
                    traderObject[f"{product}_trade_volumes"].pop(0)
        
        if own_trades:
            for trade in own_trades:
                traderObject[f"{product}_trade_prices"].append(trade.price)
                traderObject[f"{product}_trade_volumes"].append(trade.quantity)
                
                # Keep history within limits
                if len(traderObject[f"{product}_trade_prices"]) > self.params[product]["price_history_max"]:
                    traderObject[f"{product}_trade_prices"].pop(0)
                    traderObject[f"{product}_trade_volumes"].pop(0)
        
        # Calculate VWAP if we have trade data
        vwap = None
        if traderObject[f"{product}_trade_prices"]:
            # Use the most recent trades up to vwap_lookback
            lookback = min(self.params[product]["vwap_lookback"], len(traderObject[f"{product}_trade_prices"]))
            recent_prices = traderObject[f"{product}_trade_prices"][-lookback:]
            recent_volumes = traderObject[f"{product}_trade_volumes"][-lookback:]
            
            # Calculate VWAP
            if sum(recent_volumes) > 0:
                vwap = sum(p * v for p, v in zip(recent_prices, recent_volumes)) / sum(recent_volumes)
        
        # Combine different price signals to get final fair value
        if vwap is not None and mid_price is not None:
            # Weighted average between VWAP, mid price, and previous fair value
            alpha = self.params[product]["ema_alpha"]
            fair_value = 0.5 * vwap + 0.3 * mid_price + 0.2 * prev_fair_value
            
            # Apply EMA to smooth transitions
            fair_value = alpha * fair_value + (1 - alpha) * prev_fair_value
        elif mid_price is not None:
            # If we only have mid price, use EMA with previous fair value
            alpha = self.params[product]["ema_alpha"]
            fair_value = alpha * mid_price + (1 - alpha) * prev_fair_value
        elif vwap is not None:
            # If we only have VWAP, use EMA with previous fair value
            alpha = self.params[product]["ema_alpha"]
            fair_value = alpha * vwap + (1 - alpha) * prev_fair_value
        else:
            # If no price signals available, use previous fair value
            fair_value = prev_fair_value
        
        # Store the calculated fair value
        traderObject[f"{product}_fair_value_history"].append(fair_value)
        traderObject[f"{product}_last_fair_value"] = fair_value
        
        # Keep history within limits
        if len(traderObject[f"{product}_fair_value_history"]) > self.params[product]["price_history_max"]:
            traderObject[f"{product}_fair_value_history"].pop(0)
        
        return fair_value

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

        # Handle best ask (sell orders)
        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]

            if best_ask <= fair_value - take_width:
                quantity = min(
                    best_ask_amount, position_limit - position
                )  # max amt to buy
                if quantity > 0:
                    orders.append(Order(product, best_ask, quantity))
                    buy_order_volume += quantity
                    order_depth.sell_orders[best_ask] += quantity
                    if order_depth.sell_orders[best_ask] == 0:
                        del order_depth.sell_orders[best_ask]

        # Handle best bid (buy orders)
        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]

            if best_bid >= fair_value + take_width:
                quantity = min(
                    best_bid_amount, position_limit + position
                )  # should be the max we can sell
                if quantity > 0:
                    orders.append(Order(product, best_bid, -1 * quantity))
                    sell_order_volume += quantity
                    order_depth.buy_orders[best_bid] -= quantity
                    if order_depth.buy_orders[best_bid] == 0:
                        del order_depth.buy_orders[best_bid]

        return buy_order_volume, sell_order_volume

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
        # Enhanced market making for better position management with increased limits
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, round(bid), buy_quantity))  # Buy order

        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, round(ask), -sell_quantity))  # Sell order
        return buy_order_volume, sell_order_volume

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
    ) -> List[Order]:
        # Improved clearing of positions based on refined strategy
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value - width)
        fair_for_ask = round(fair_value + width)

        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)

        # Clear buy orders if position is long
        if position_after_take > 0:
            clear_quantity = sum(
                volume for price, volume in order_depth.buy_orders.items() if price >= fair_for_ask
            )
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        # Clear sell orders if position is short
        if position_after_take < 0:
            clear_quantity = sum(
                abs(volume) for price, volume in order_depth.sell_orders.items() if price <= fair_for_bid
            )
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)

        return buy_order_volume, sell_order_volume

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

    def make_orders(
        self,
        product,
        order_depth: OrderDepth,
        fair_value: float,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        disregard_edge: float,  # disregard trades within this edge for pennying or joining
        join_edge: float,  # join trades within this edge
        default_edge: float,  # default edge to request if there are no levels to penny or join
        manage_position: bool = True,  # Set to True to enable position management
        soft_position_limit: int = 25,  # Updated soft position limit based on new limits
    ):
        orders: List[Order] = []
        asks_above_fair = [
            price
            for price in order_depth.sell_orders.keys()
            if price > fair_value + disregard_edge
        ]
        bids_below_fair = [
            price
            for price in order_depth.buy_orders.keys()
            if price < fair_value - disregard_edge
        ]

        best_ask_above_fair = min(asks_above_fair) if len(asks_above_fair) > 0 else None
        best_bid_below_fair = max(bids_below_fair) if len(bids_below_fair) > 0 else None

        ask = round(fair_value + default_edge)
        if best_ask_above_fair != None:
            if abs(best_ask_above_fair - fair_value) <= join_edge:
                ask = best_ask_above_fair  # join
            else:
                ask = best_ask_above_fair - 1  # penny

        bid = round(fair_value - default_edge)
        if best_bid_below_fair != None:
            if abs(fair_value - best_bid_below_fair) <= join_edge:
                bid = best_bid_below_fair
            else:
                bid = best_bid_below_fair + 1

        # Position management logic
        if manage_position:
            if position > soft_position_limit:
                # More aggressive sell when long
                position_factor = min(5, int(np.floor(0.05 * abs(position))))
                ask -= position_factor
            elif position < -1 * soft_position_limit:
                # More aggressive buy when short
                position_factor = min(5, int(np.floor(0.05 * abs(position))))
                bid += position_factor

        buy_order_volume, sell_order_volume = self.market_make(
            product,
            orders,
            bid,
            ask,
            position,
            buy_order_volume,
            sell_order_volume,
        )

        return orders, buy_order_volume, sell_order_volume

    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

        result = {}

        if Product.RAINFOREST_RESIN in self.params and Product.RAINFOREST_RESIN in state.order_depths:
            resin_position = (
                state.position[Product.RAINFOREST_RESIN]
                if Product.RAINFOREST_RESIN in state.position
                else 0
            )
            
            # Get market trades for the product
            market_trades = state.market_trades.get(Product.RAINFOREST_RESIN, [])
            own_trades = state.own_trades.get(Product.RAINFOREST_RESIN, [])
            
            # Calculate dynamic fair value
            fair_value = self.calculate_fair_value(
                Product.RAINFOREST_RESIN,
                state.order_depths[Product.RAINFOREST_RESIN],
                market_trades,
                own_trades,
                traderObject
            )
            
            resin_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(
                    Product.RAINFOREST_RESIN,
                    state.order_depths[Product.RAINFOREST_RESIN],
                    fair_value,
                    self.params[Product.RAINFOREST_RESIN]["take_width"],
                    resin_position,
                )
            )
            resin_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(
                    Product.RAINFOREST_RESIN,
                    state.order_depths[Product.RAINFOREST_RESIN],
                    fair_value,
                    self.params[Product.RAINFOREST_RESIN]["clear_width"],
                    resin_position,
                    buy_order_volume,
                    sell_order_volume,
                )
            )
            resin_make_orders, _, _ = self.make_orders(
                Product.RAINFOREST_RESIN,
                state.order_depths[Product.RAINFOREST_RESIN],
                fair_value,
                resin_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.RAINFOREST_RESIN]["disregard_edge"],
                self.params[Product.RAINFOREST_RESIN]["join_edge"],
                self.params[Product.RAINFOREST_RESIN]["default_edge"],
                True,  # Enable position management
                self.params[Product.RAINFOREST_RESIN]["soft_position_limit"],
            )
            result[Product.RAINFOREST_RESIN] = (
                resin_take_orders + resin_clear_orders + resin_make_orders
            )

        conversions = 0  # No conversions needed for this product
        traderData = jsonpickle.encode(traderObject)

        return result, conversions, traderData