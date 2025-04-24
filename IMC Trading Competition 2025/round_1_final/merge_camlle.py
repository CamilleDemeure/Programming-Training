from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Tuple
import jsonpickle
import numpy as np
from collections import deque

class Product: 
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    SQUID_INK = "SQUID_INK"

class Trader:
    def __init__(self, params: Dict = None):
        # Default parameters for each product
        resin_defaults = {
            "fair_value": 10000,
            "take_width": 2,
            "clear_width": 0.5,
            "disregard_edge": 1,
            "join_edge": 2,
            "default_edge": 4,
            "soft_position_limit": 25,
            "volume_limit": 0,
            "penny_val": 1,
        }
        squid_defaults = {
            "vwap_window": 10,
            "bid_offset": 0,
            "ask_offset": 0,
            "take_threshold": 0.1,
            "make_threshold": 0.05,
            "position_scaling": 0.8,
            "clear_threshold": 0.02,
            "safety_margin": 0.03,
            "max_position_pct": 0.8,
        }
        self.params = params if params is not None else {}
        # Set default parameters if not provided
        self.params.setdefault(Product.RAINFOREST_RESIN, resin_defaults)
        self.params.setdefault(Product.SQUID_INK, squid_defaults)
        self.LIMIT = {Product.RAINFOREST_RESIN: 50, Product.SQUID_INK: 50}
    
    # ------------------------ RAINFOREST_RESIN methods ------------------------
    def resin_take_best_orders(self,
                               order_depth: OrderDepth,
                               fair_value: int,
                               take_width: float,
                               position: int,
                               buy_order_volume: int,
                               sell_order_volume: int) -> Tuple[List[Order], int, int]:
        orders: List[Order] = []
        # Process sell orders (buy opportunities)
        sell_orders = order_depth.sell_orders  # local reference
        for price in sorted(sell_orders.keys()):
            if price > fair_value - take_width:
                break
            available = -sell_orders.get(price, 0)
            qty = min(available, self.LIMIT[Product.RAINFOREST_RESIN] - position - buy_order_volume)
            if qty > 0:
                orders.append(Order(Product.RAINFOREST_RESIN, price, qty))
                buy_order_volume += qty
                sell_orders[price] += qty
                if sell_orders[price] == 0:
                    del sell_orders[price]
        # Process buy orders (sell opportunities)
        buy_orders = order_depth.buy_orders  # local reference
        for price in sorted(buy_orders.keys(), reverse=True):
            if price < fair_value + take_width:
                break
            available = buy_orders.get(price, 0)
            qty = min(available, self.LIMIT[Product.RAINFOREST_RESIN] + position - sell_order_volume)
            if qty > 0:
                orders.append(Order(Product.RAINFOREST_RESIN, price, -qty))
                sell_order_volume += qty
                buy_orders[price] -= qty
                if buy_orders[price] == 0:
                    del buy_orders[price]
        return orders, buy_order_volume, sell_order_volume

    def resin_clear_position_order(self,
                                   order_depth: OrderDepth,
                                   fair_value: float,
                                   width: float,
                                   position: int,
                                   buy_order_volume: int,
                                   sell_order_volume: int) -> Tuple[List[Order], int, int]:
        orders: List[Order] = []
        new_position = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value - width)
        fair_for_ask = round(fair_value + width)
        buy_quantity = self.LIMIT[Product.RAINFOREST_RESIN] - (position + buy_order_volume)
        sell_quantity = self.LIMIT[Product.RAINFOREST_RESIN] + (position - sell_order_volume)
        if new_position > 0 and order_depth.buy_orders:
            clear_qty = sum(vol for p, vol in order_depth.buy_orders.items() if p >= fair_for_ask)
            clear_qty = min(clear_qty, new_position)
            executed = min(sell_quantity, clear_qty)
            if executed > 0:
                orders.append(Order(Product.RAINFOREST_RESIN, fair_for_ask, -executed))
                sell_order_volume += executed
        if new_position < 0 and order_depth.sell_orders:
            clear_qty = sum(abs(vol) for p, vol in order_depth.sell_orders.items() if p <= fair_for_bid)
            clear_qty = min(clear_qty, abs(new_position))
            executed = min(buy_quantity, clear_qty)
            if executed > 0:
                orders.append(Order(Product.RAINFOREST_RESIN, fair_for_bid, executed))
                buy_order_volume += executed
        return orders, buy_order_volume, sell_order_volume

    def resin_market_make(self,
                          order_depth: OrderDepth,
                          fair_value: int,
                          position: int,
                          buy_order_volume: int,
                          sell_order_volume: int) -> Tuple[List[Order], int, int]:
        orders: List[Order] = []
        penny_val = self.params[Product.RAINFOREST_RESIN].get("penny_val", 1)
        default_edge = self.params[Product.RAINFOREST_RESIN].get("default_edge", 4)
        volume_limit = self.params[Product.RAINFOREST_RESIN].get("volume_limit", 0)
        sell_orders = order_depth.sell_orders
        buy_orders = order_depth.buy_orders
        prices_above = [price for price in sell_orders.keys() if price > fair_value + penny_val]
        baaf = min(prices_above) if prices_above else fair_value + penny_val + default_edge
        prices_below = [price for price in buy_orders.keys() if price < fair_value - penny_val]
        bbbf = max(prices_below) if prices_below else fair_value - penny_val - default_edge
        if baaf <= fair_value + 2 and position <= volume_limit:
            baaf = fair_value + 3
        if bbbf >= fair_value - 2 and position >= -volume_limit:
            bbbf = fair_value - 3
        buy_qty = self.LIMIT[Product.RAINFOREST_RESIN] - (position + buy_order_volume)
        if buy_qty > 0:
            orders.append(Order(Product.RAINFOREST_RESIN, round(bbbf + 1), buy_qty))
        sell_qty = self.LIMIT[Product.RAINFOREST_RESIN] + (position - sell_order_volume)
        if sell_qty > 0:
            orders.append(Order(Product.RAINFOREST_RESIN, round(baaf - 1), -sell_qty))
        return orders, buy_order_volume, sell_order_volume

    # ------------------------ SQUID_INK methods ------------------------
    def calculate_vwap(self, prices: deque, volumes: deque) -> float:
        if not prices or not volumes or len(prices) != len(volumes) or sum(volumes) == 0:
            return None
        prices_arr = np.array(prices)
        volumes_arr = np.array(volumes)
        return float(np.dot(prices_arr, volumes_arr) / np.sum(volumes_arr))

    def predict_next_prices(self, bid_vwap: float, ask_vwap: float) -> Tuple[float, float]:
        bid_off = self.params[Product.SQUID_INK].get("bid_offset", 0)
        ask_off = self.params[Product.SQUID_INK].get("ask_offset", 0)
        return bid_vwap + bid_off, ask_vwap + ask_off

    def fair_value_estimation(self, bid_vwap: float, ask_vwap: float, next_bid: float, next_ask: float) -> float:
        return ((bid_vwap + ask_vwap) / 2 + (next_bid + next_ask) / 2) / 2

    def update_price_history(self, traderObject: Dict, product: str, state: TradingState) -> Tuple[float, float, float, float, float]:
        # Use a fixed window defined in the parameters
        window_size = self.params[Product.SQUID_INK].get("vwap_window", 10)
        history_keys = ["bid_price_history", "bid_volume_history", "ask_price_history", "ask_volume_history", "bid_vwap_history", "ask_vwap_history", "fair_values"]
        for key in history_keys:
            hist_key = f"{product}_{key}"
            if hist_key not in traderObject:
                traderObject[hist_key] = deque(maxlen=window_size)
        od = state.order_depths.get(product)
        if od:
            if od.buy_orders:
                best_bid = max(od.buy_orders.keys())
                traderObject[f"{product}_bid_price_history"].append(best_bid)
                traderObject[f"{product}_bid_volume_history"].append(od.buy_orders[best_bid])
            if od.sell_orders:
                best_ask = min(od.sell_orders.keys())
                traderObject[f"{product}_ask_price_history"].append(best_ask)
                traderObject[f"{product}_ask_volume_history"].append(abs(od.sell_orders[best_ask]))
        bid_prices = traderObject[f"{product}_bid_price_history"]
        bid_volumes = traderObject[f"{product}_bid_volume_history"]
        ask_prices = traderObject[f"{product}_ask_price_history"]
        ask_volumes = traderObject[f"{product}_ask_volume_history"]
        bid_vwap = self.calculate_vwap(bid_prices, bid_volumes) if bid_prices else None
        ask_vwap = self.calculate_vwap(ask_prices, ask_volumes) if ask_prices else None
        if bid_vwap is not None:
            traderObject[f"{product}_bid_vwap_history"].append(bid_vwap)
        if ask_vwap is not None:
            traderObject[f"{product}_ask_vwap_history"].append(ask_vwap)
        if bid_vwap is not None and ask_vwap is not None:
            next_bid, next_ask = self.predict_next_prices(bid_vwap, ask_vwap)
            fair_value = self.fair_value_estimation(bid_vwap, ask_vwap, next_bid, next_ask)
            traderObject[f"{product}_fair_values"].append(fair_value)
            return bid_vwap, ask_vwap, next_bid, next_ask, fair_value
        return None, None, None, None, None

    def squid_take_opportunity_orders(self,
                                      product: str,
                                      order_depth: OrderDepth,
                                      fair_value: float,
                                      next_bid: float,
                                      next_ask: float,
                                      position: int) -> Tuple[List[Order], int, int]:
        orders: List[Order] = []
        buy_vol = 0
        sell_vol = 0
        limit = self.LIMIT[Product.SQUID_INK]
        max_position = int(limit * self.params[Product.SQUID_INK].get("max_position_pct", 0.8))
        safety = self.params[Product.SQUID_INK].get("safety_margin", 0.03)
        take_threshold = self.params[Product.SQUID_INK].get("take_threshold", 0.1)
        pos_ratio = position / limit if limit > 0 else 0
        buy_threshold = next_bid - take_threshold - (pos_ratio * safety)
        sell_threshold = next_ask + take_threshold - (pos_ratio * safety)
        sell_orders = order_depth.sell_orders
        for price in sorted(sell_orders.keys()):
            if price < buy_threshold:
                available = -sell_orders[price]
                max_buy = min(available, max_position - position - buy_vol)
                if max_buy > 0:
                    orders.append(Order(product, price, max_buy))
                    buy_vol += max_buy
            else:
                break
        buy_orders = order_depth.buy_orders
        for price in sorted(buy_orders.keys(), reverse=True):
            if price > sell_threshold:
                available = buy_orders[price]
                max_sell = min(available, max_position + position - sell_vol)
                if max_sell > 0:
                    orders.append(Order(product, price, -max_sell))
                    sell_vol += max_sell
            else:
                break
        return orders, buy_vol, sell_vol

    def squid_clear_position_orders(self,
                                    product: str,
                                    order_depth: OrderDepth,
                                    fair_value: float,
                                    position: int,
                                    buy_vol: int,
                                    sell_vol: int) -> Tuple[List[Order], int, int]:
        orders: List[Order] = []
        pos_after = position + buy_vol - sell_vol
        if abs(pos_after) < 1:
            return orders, buy_vol, sell_vol
        clear_threshold = self.params[Product.SQUID_INK].get("clear_threshold", 0.02)
        if pos_after > 0 and order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            if best_bid >= fair_value - clear_threshold:
                available = order_depth.buy_orders[best_bid]
                max_sell = min(available, pos_after)
                if max_sell > 0:
                    orders.append(Order(product, best_bid, -max_sell))
                    sell_vol += max_sell
        if pos_after < 0 and order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            if best_ask <= fair_value + clear_threshold:
                available = -order_depth.sell_orders[best_ask]
                max_buy = min(available, abs(pos_after))
                if max_buy > 0:
                    orders.append(Order(product, best_ask, max_buy))
                    buy_vol += max_buy
        return orders, buy_vol, sell_vol

    def squid_market_making_orders(self,
                                   product: str,
                                   order_depth: OrderDepth,
                                   fair_value: float,
                                   next_bid: float,
                                   next_ask: float,
                                   position: int,
                                   buy_vol: int,
                                   sell_vol: int) -> List[Order]:
        orders: List[Order] = []
        pos_after = position + buy_vol - sell_vol
        limit = self.LIMIT[Product.SQUID_INK]
        pos_ratio = abs(pos_after) / limit if limit > 0 else 0
        pos_scaling = self.params[Product.SQUID_INK].get("position_scaling", 0.8)
        position_scale = 1 - pos_ratio * pos_scaling
        position_scale = max(0.1, position_scale)
        make_threshold = self.params[Product.SQUID_INK].get("make_threshold", 0.05)
        if pos_after > 0:
            bid_price = next_bid - make_threshold - (pos_ratio * make_threshold)
            ask_price = next_ask - (pos_ratio * make_threshold / 2)
        elif pos_after < 0:
            bid_price = next_bid + (pos_ratio * make_threshold / 2)
            ask_price = next_ask + make_threshold + (pos_ratio * make_threshold)
        else:
            bid_price = next_bid - make_threshold
            ask_price = next_ask + make_threshold
        bid_price = round(bid_price)
        ask_price = round(ask_price)
        max_buy = max(0, limit - pos_after)
        max_sell = max(0, limit + pos_after)
        buy_qty = int(max_buy * position_scale)
        sell_qty = int(max_sell * position_scale)
        if buy_qty > 0:
            orders.append(Order(product, bid_price, buy_qty))
        if sell_qty > 0:
            orders.append(Order(product, ask_price, -sell_qty))
        return orders

    # ------------------------ Main run() method ------------------------
    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData:
            try:
                traderObject = jsonpickle.decode(state.traderData)
            except Exception as e:
                print("Error decoding traderData:", e)
        result = {}

        # -------- RAINFOREST_RESIN Strategy --------
        if Product.RAINFOREST_RESIN in state.order_depths:
            od_resin = state.order_depths[Product.RAINFOREST_RESIN]
            resin_position = state.position.get(Product.RAINFOREST_RESIN, 0)
            resin_params = self.params[Product.RAINFOREST_RESIN]
            buy_vol = 0
            sell_vol = 0
            # Take best orders for resin
            take_orders, buy_vol, sell_vol = self.resin_take_best_orders(
                od_resin,
                resin_params.get("fair_value", 10000),
                resin_params.get("take_width", 2),
                resin_position,
                buy_vol,
                sell_vol
            )
            # Clear position orders for resin
            clear_orders, buy_vol, sell_vol = self.resin_clear_position_order(
                od_resin,
                resin_params.get("fair_value", 10000),
                resin_params.get("clear_width", 0.5),
                resin_position,
                buy_vol,
                sell_vol
            )
            # Market making orders for resin
            mm_orders, buy_vol, sell_vol = self.resin_market_make(
                od_resin,
                resin_params.get("fair_value", 10000),
                resin_position,
                buy_vol,
                sell_vol
            )
            orders_resin = take_orders + clear_orders + mm_orders
            if orders_resin:
                result[Product.RAINFOREST_RESIN] = orders_resin

        # -------- SQUID_INK Strategy --------
        if Product.SQUID_INK in state.order_depths:
            od_squid = state.order_depths[Product.SQUID_INK]
            squid_position = state.position.get(Product.SQUID_INK, 0)
            # Update or initialize price history
            bid_vwap, ask_vwap, next_bid, next_ask, fair_value = self.update_price_history(traderObject, Product.SQUID_INK, state)
            if fair_value is None and od_squid.buy_orders and od_squid.sell_orders:
                best_bid = max(od_squid.buy_orders.keys())
                best_ask = min(od_squid.sell_orders.keys())
                bid_vwap = best_bid
                ask_vwap = best_ask
                fair_value = (best_bid + best_ask) / 2
                next_bid = bid_vwap + self.params[Product.SQUID_INK].get("bid_offset", 0)
                next_ask = ask_vwap + self.params[Product.SQUID_INK].get("ask_offset", 0)
            squid_orders = []
            buy_vol = 0
            sell_vol = 0
            if fair_value is not None and next_bid is not None and next_ask is not None:
                opp_orders, bvol, svol = self.squid_take_opportunity_orders(
                    Product.SQUID_INK, od_squid, fair_value, next_bid, next_ask, squid_position
                )
                squid_orders.extend(opp_orders)
                buy_vol += bvol
                sell_vol += svol
                clear_orders, bvol, svol = self.squid_clear_position_orders(
                    Product.SQUID_INK, od_squid, fair_value, squid_position, buy_vol, sell_vol
                )
                squid_orders.extend(clear_orders)
                buy_vol += bvol
                sell_vol += svol
                # If desired, market making orders can be added by uncommenting the following lines:
                mm_orders = self.squid_market_making_orders(
                    Product.SQUID_INK, od_squid, fair_value, next_bid, next_ask, squid_position, buy_vol, sell_vol
                )
                squid_orders.extend(mm_orders)
            if squid_orders:
                result[Product.SQUID_INK] = squid_orders

        traderData = jsonpickle.encode(traderObject)
        return result, 0, traderData
