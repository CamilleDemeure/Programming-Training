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
        "fair_value_window": 20,       # Number of mid-prices to average over
        "take_width": 2,
        "clear_width": 0.5,
        "disregard_edge": 1,
        "join_edge": 1,
        "default_edge": 4,
        "soft_position_limit": 25,
        "volume_limit": 0,
        "penny_val": 1,
    },
}


class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params
        self.LIMIT = {Product.KELP: 50}
        self.mid_prices = {Product.KELP: []}

    def update_fair_value(self, product: str, order_depth: OrderDepth) -> float | None:
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None  # No mid price if either side is missing

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid_price = (best_bid + best_ask) / 2

        # Update rolling window of mid prices
        self.mid_prices[product].append(mid_price)
        window = self.params[product]["fair_value_window"]
        if len(self.mid_prices[product]) > window:
            self.mid_prices[product].pop(0)

        if len(self.mid_prices[product]) < window:
            return None  # Not enough data to compute fair value

        return np.mean(self.mid_prices[product])


    def take_best_orders(self, product, fair_value, take_width, orders, order_depth, position, buy_order_volume, sell_order_volume):
        position_limit = self.LIMIT[product]

        if len(order_depth.sell_orders) != 0:
            sorted_sell_prices = sorted(order_depth.sell_orders.keys())
            for price in sorted_sell_prices:
                if price > fair_value - take_width:
                    break
                best_ask_amount = -1 * order_depth.sell_orders[price]
                quantity = min(best_ask_amount, position_limit - position - buy_order_volume)
                if quantity > 0:
                    orders.append(Order(product, price, quantity))
                    buy_order_volume += quantity
                    order_depth.sell_orders[price] += quantity
                    if order_depth.sell_orders[price] == 0:
                        del order_depth.sell_orders[price]

        if len(order_depth.buy_orders) != 0:
            sorted_buy_prices = sorted(order_depth.buy_orders.keys(), reverse=True)
            for price in sorted_buy_prices:
                if price < fair_value + take_width:
                    break
                best_bid_amount = order_depth.buy_orders[price]
                quantity = min(best_bid_amount, position_limit + position - sell_order_volume)
                if quantity > 0:
                    orders.append(Order(product, price, -1 * quantity))
                    sell_order_volume += quantity
                    order_depth.buy_orders[price] -= quantity
                    if order_depth.buy_orders[price] == 0:
                        del order_depth.buy_orders[price]

        return buy_order_volume, sell_order_volume

    def market_make(self, product, orders, bid, ask, position, buy_order_volume, sell_order_volume):
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, round(bid), buy_quantity))

        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, round(ask), -sell_quantity))

        return buy_order_volume, sell_order_volume

    def clear_position_order(self, product, fair_value, width, orders, order_depth, position, buy_order_volume, sell_order_volume):
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value - width)
        fair_for_ask = round(fair_value + width)
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)

        if position_after_take > 0:
            clear_quantity = sum(
                volume for price, volume in order_depth.buy_orders.items() if price >= fair_for_ask
            )
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

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

    def take_orders(self, product, order_depth, fair_value, take_width, position):
        orders = []
        buy_order_volume = 0
        sell_order_volume = 0
        buy_order_volume, sell_order_volume = self.take_best_orders(
            product, fair_value, take_width, orders, order_depth, position, buy_order_volume, sell_order_volume
        )
        return orders, buy_order_volume, sell_order_volume

    def clear_orders(self, product, order_depth, fair_value, clear_width, position, buy_order_volume, sell_order_volume):
        orders = []
        buy_order_volume, sell_order_volume = self.clear_position_order(
            product, fair_value, clear_width, orders, order_depth, position, buy_order_volume, sell_order_volume
        )
        return orders, buy_order_volume, sell_order_volume

    def make_resin_orders(self, order_depth, fair_value, position, buy_order_volume, sell_order_volume, volume_limit, penny_val=1):
        orders = []

        try:
            baaf = min([price for price in order_depth.sell_orders.keys() if price > fair_value + penny_val])
        except:
            baaf = fair_value + 4

        try:
            bbbf = max([price for price in order_depth.buy_orders.keys() if price < fair_value - penny_val])
        except:
            bbbf = fair_value - 4

        if baaf <= fair_value + 2 and position <= volume_limit:
            baaf = fair_value + 3
        if bbbf >= fair_value - 2 and position >= -volume_limit:
            bbbf = fair_value - 3

        buy_order_volume, sell_order_volume = self.market_make(
            Product.KELP,
            orders,
            bbbf + 1,
            baaf - 1,
            position,
            buy_order_volume,
            sell_order_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData:
            traderObject = jsonpickle.decode(state.traderData)

        result = {}

        product = Product.KELP
        if product in self.params and product in state.order_depths:
            resin_position = state.position.get(product, 0)
            order_depth = state.order_depths[product]

            fair_value = self.update_fair_value(product, order_depth)

            if fair_value is None:
                return {}, 0, jsonpickle.encode(traderObject)

            resin_take_orders, buy_order_volume, sell_order_volume = self.take_orders(
                product, order_depth, fair_value, self.params[product]["take_width"], resin_position
            )

            resin_clear_orders, buy_order_volume, sell_order_volume = self.clear_orders(
                product, order_depth, fair_value, self.params[product]["clear_width"], resin_position, buy_order_volume, sell_order_volume
            )

            resin_make_orders, _, _ = self.make_resin_orders(
                order_depth,
                fair_value,
                resin_position,
                buy_order_volume,
                sell_order_volume,
                self.params[product]["volume_limit"],
                self.params[product]["penny_val"],
            )

            result[product] = resin_take_orders + resin_clear_orders + resin_make_orders

        conversions = 0
        traderData = jsonpickle.encode(traderObject)
        return result, conversions, traderData
