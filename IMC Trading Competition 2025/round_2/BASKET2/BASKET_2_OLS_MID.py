from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict
import jsonpickle
import numpy as np
import math

# =============================================================================
# Product Class
# =============================================================================
class Product:
    PICNIC_BASKET2 = "PICNIC_BASKET2"

# =============================================================================
# PARAMS Dictionary
# =============================================================================
PARAMS = {
    Product.PICNIC_BASKET2: {
        "base_fair_value": 30429,       # Median value used as a fallback
        "take_width": 5,                # Threshold around fair value for order taking
        "clear_width": 0.5,             # Threshold for triggering position clearing
        "disregard_edge": 3,            # Orders within this distance from fair value are ignored
        "join_edge": 5,                 # Orders within this edge are joined instead of outbid
        "default_edge": 7,              # Fallback edge if no better reference is available
        "soft_position_limit": 30,      # Soft position limit for risk management
        "volume_limit": 0,              # Parameter for finer position management
        "penny_val": 1,                 # Penny increment for adjusting order prices
        "vwap_window": 100,             # Not used for fair value computation now
        # Aggressiveness parameters (for market making orders)
        "aggressiveness": 0.3,
        "fair_value_order_size": 0,     # Fair value orders removed since we now derive fair value from OLS
        "tier1_distance": 8,
        "tier2_distance": 15,
        "tier3_distance": 25,
        "tier1_volume_pct": 0.4,
        "tier2_volume_pct": 0.35,
        "tier3_volume_pct": 0.25,
        "position_scaling": 0.3,
    },
}

# =============================================================================
# Trader Class
# =============================================================================
class Trader:
    def __init__(self, params=None):
        self.params = params if params is not None else PARAMS
        # Hard limit according to exchange/strategy constraints.
        self.LIMIT = {Product.PICNIC_BASKET2: 50}

    # -------------------------------------------------------------------------
    # OLS Prediction Functions for Fair Value Calculation
    # -------------------------------------------------------------------------
    def predict_next_bid(self, features: Dict) -> float:
        """
        Predicts the next bid price using pre-trained OLS coefficients.
        """
        intercept = 50.2339
        coef_JAMS = 0.0031
        coef_KELP = -0.0141
        coef_fair_vwap = 0.0066
        coef_lag = 0.9920

        predicted_bid = (
            intercept +
            coef_JAMS * features['JAMS'] +
            coef_KELP * features['KELP'] +
            coef_fair_vwap * features['fair_VWAP_ask_VWAP'] +
            coef_lag * features['lag1_PICNIC_BASKET2']
        )
        print(f"[Predict Bid] Features: {features} => Predicted Bid: {predicted_bid:.2f}")
        return predicted_bid

    def predict_next_ask(self, features: Dict) -> float:
        """
        Predicts the next ask price using pre-trained OLS coefficients.
        """
        intercept = 41.9161
        coef_JAMS = 0.0030
        coef_KELP = -0.0114
        coef_fair_vwap = 0.0074
        coef_lag = 0.9914

        predicted_ask = (
            intercept +
            coef_JAMS * features['JAMS'] +
            coef_KELP * features['KELP'] +
            coef_fair_vwap * features['fair_VWAP_ask_VWAP'] +
            coef_lag * features['lag1_PICNIC_BASKET2']
        )
        print(f"[Predict Ask] Features: {features} => Predicted Ask: {predicted_ask:.2f}")
        return predicted_ask

    # -------------------------------------------------------------------------
    # Feature Extraction Functions
    # -------------------------------------------------------------------------
    def compute_JAMS(self, order_depth: OrderDepth) -> float:
        """
        Computes the 'JAMS' feature using the best bid price.
        """
        if order_depth.buy_orders:
            jams = max(order_depth.buy_orders.keys())
            print(f"[Compute JAMS] Best Bid Price: {jams}")
            return jams
        print("[Compute JAMS] No buy orders; using default 30429")
        return 30429  # Default fallback value

    def compute_KELP(self, order_depth: OrderDepth) -> float:
        """
        Computes the 'KELP' feature using the volume at the best bid.
        """
        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            kelp = order_depth.buy_orders.get(best_bid, 0)
            print(f"[Compute KELP] Volume at Best Bid ({best_bid}): {kelp}")
            return kelp
        print("[Compute KELP] No buy orders; using default volume 100")
        return 100  # Default fallback volume

    def get_current_features(self, order_depth: OrderDepth, traderObject: Dict) -> Dict:
        """
        Constructs the feature vector for OLS prediction.
        """
        features = {}
        features['JAMS'] = self.compute_JAMS(order_depth)
        features['KELP'] = self.compute_KELP(order_depth)
        # Use previously computed rolling fair ask VWAP if available; otherwise, set a default.
        features['fair_VWAP_ask_VWAP'] = traderObject.get("fair_VWAP_ask_VWAP", 30430)
        # Use the last known PICNIC_BASKET2 price as a lag feature.
        features['lag1_PICNIC_BASKET2'] = traderObject.get("last_PICNIC_BASKET2_price", 30429)
        print(f"[Get Features] Constructed features: {features}")
        return features

    # -------------------------------------------------------------------------
    # Updated Fair Value Calculation using OLS Predictions
    # -------------------------------------------------------------------------
    def calculate_fair_value_PICNIC_BASKET2(
        self,
        product: str,
        order_depth: OrderDepth,
        traderObject: Dict
    ) -> float:
        """
        Computes the dynamic fair value from OLS-based predictions.
        """
        features = self.get_current_features(order_depth, traderObject)
        predicted_bid = self.predict_next_bid(features)
        predicted_ask = self.predict_next_ask(features)
        fair_value = (predicted_bid + predicted_ask) / 2
        print(f"[Fair Value] Predicted Bid: {predicted_bid:.2f}, Predicted Ask: {predicted_ask:.2f} => Fair Value: {fair_value:.2f}")
        # Update persistent state with the fair value and lag.
        traderObject[f"{product}_fair_value"] = fair_value
        traderObject[f"last_{product}_price"] = fair_value
        return fair_value

    # -------------------------------------------------------------------------
    # Taking Advantage of Mispriced Orders (Opportunity Taking)
    # -------------------------------------------------------------------------
    def take_best_orders_PICNIC_BASKET2(
        self,
        product: str,
        fair_value: float,
        take_width: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (int, int):
        """
        Executes orders for mispricings relative to the computed fair value.
        """
        position_limit = self.LIMIT[product]
        print(f"[Take Orders] Fair Value: {fair_value}, Take Width: {take_width}, Position: {position}")

        # --- Sell Orders (Buying Opportunity) ---
        if order_depth.sell_orders:
            sorted_sell_prices = sorted(order_depth.sell_orders.keys())
            for price in sorted_sell_prices:
                if price > fair_value - take_width:
                    break
                best_ask_amount = -order_depth.sell_orders[price]
                quantity = min(best_ask_amount, position_limit - position - buy_order_volume)
                if quantity > 0:
                    orders.append(Order(product, price, quantity))
                    print(f"[Take Orders] Submitting BUY order: Price: {price}, Quantity: {quantity}")
                    buy_order_volume += quantity
                    order_depth.sell_orders[price] += quantity
                    if order_depth.sell_orders[price] == 0:
                        del order_depth.sell_orders[price]

        # --- Buy Orders (Selling Opportunity) ---
        if order_depth.buy_orders:
            sorted_buy_prices = sorted(order_depth.buy_orders.keys(), reverse=True)
            for price in sorted_buy_prices:
                if price < fair_value + take_width:
                    break
                best_bid_amount = order_depth.buy_orders[price]
                quantity = min(best_bid_amount, position_limit + position - sell_order_volume)
                if quantity > 0:
                    orders.append(Order(product, price, -quantity))
                    print(f"[Take Orders] Submitting SELL order: Price: {price}, Quantity: {quantity}")
                    sell_order_volume += quantity
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
        bid: float,
        ask: float,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (int, int):
        """
        Places market making orders at the computed fair value.
        """
        position_limit = self.LIMIT[product]
        available_buy_quantity = position_limit - (position + buy_order_volume)
        available_sell_quantity = position_limit + (position - sell_order_volume)
        print(f"[Market Make] Position: {position}, Available BUY: {available_buy_quantity}, Available SELL: {available_sell_quantity}")
        if available_buy_quantity > 0:
            orders.append(Order(product, round(bid), available_buy_quantity))
            print(f"[Market Make] Placing BUY market making order at {bid} for {available_buy_quantity}")
        if available_sell_quantity > 0:
            orders.append(Order(product, round(ask), -available_sell_quantity))
            print(f"[Market Make] Placing SELL market making order at {ask} for {available_sell_quantity}")
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
    ) -> (List[Order], int, int):
        """
        Clears positions if the net position deviates too much from fair value.
        Now returns (orders, buy_order_volume, sell_order_volume).
        """
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value - width)
        fair_for_ask = round(fair_value + width)
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        print(f"[Clear Orders] Position After Taking: {position_after_take}, Clear Range: [{fair_for_bid}, {fair_for_ask}]")

        if position_after_take > 0:
            clear_quantity = sum(volume for price, volume in order_depth.buy_orders.items() if price >= fair_for_ask)
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                print(f"[Clear Orders] Clearing LONG position: Selling {abs(sent_quantity)} at {fair_for_ask}")
                sell_order_volume += abs(sent_quantity)

        if position_after_take < 0:
            clear_quantity = sum(abs(volume) for price, volume in order_depth.sell_orders.items() if price <= fair_for_bid)
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                print(f"[Clear Orders] Clearing SHORT position: Buying {abs(sent_quantity)} at {fair_for_bid}")
                buy_order_volume += abs(sent_quantity)

        return orders, buy_order_volume, sell_order_volume

    # -------------------------------------------------------------------------
    # Wrapper Method for Order Taking
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
        Combines opportunity-taking orders for the asset.
        """
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0
        print(f"[Take Orders Wrapper] Starting order-taking with fair value: {fair_value} and position: {position}")
        buy_order_volume, sell_order_volume = self.take_best_orders_PICNIC_BASKET2(
            product, fair_value, take_width, orders, order_depth, position, buy_order_volume, sell_order_volume
        )
        return orders, buy_order_volume, sell_order_volume

    # -------------------------------------------------------------------------
    # Main Run Method Called by Trading Environment
    # -------------------------------------------------------------------------
    def run(self, state: TradingState):
        """
        Main entry point for the trading strategy.
          1. Restores persistent state.
          2. Computes fair value using OLS predictions.
          3. Executes opportunity-taking, position clearing, and (optionally) market-making orders.
          4. Updates persistent state.
        """
        traderObject = {}
        if state.traderData is not None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)
            print(f"[Run] Restored trader state: {traderObject}")

        result = {}
        if Product.PICNIC_BASKET2 in self.params and Product.PICNIC_BASKET2 in state.order_depths:
            PICNIC_BASKET2_position = state.position.get(Product.PICNIC_BASKET2, 0)
            order_depth = state.order_depths[Product.PICNIC_BASKET2]
            print(f"[Run] Current Position for {Product.PICNIC_BASKET2}: {PICNIC_BASKET2_position}")
            print(f"[Run] Order Depth: BUY: {order_depth.buy_orders}, SELL: {order_depth.sell_orders}")

            # ----- STEP 1: Compute Fair Value using OLS Predictions -----
            fair_value = self.calculate_fair_value_PICNIC_BASKET2(
                Product.PICNIC_BASKET2,
                order_depth,
                traderObject
            )
            
            current_spread = traderObject.get(f"{Product.PICNIC_BASKET2}_current_spread", 3)
            dynamic_take_width = min(self.params[Product.PICNIC_BASKET2]["take_width"], current_spread - 1)
            print(f"[Run] Dynamic Take Width: {dynamic_take_width}, Current Spread: {current_spread}")
            
            # ----- STEP 2: Opportunity Taking -----
            take_orders, buy_order_volume, sell_order_volume = self.take_orders_PICNIC_BASKET2(
                Product.PICNIC_BASKET2,
                order_depth,
                fair_value,
                dynamic_take_width,
                PICNIC_BASKET2_position
            )
            print(f"[Run] Opportunity Taking Orders: {take_orders}")
            
            # ----- STEP 3: Clearing Positions -----
            clear_orders, buy_order_volume, sell_order_volume = self.clear_position_order_PICNIC_BASKET2(
                Product.PICNIC_BASKET2,
                fair_value,
                self.params[Product.PICNIC_BASKET2]["clear_width"],
                [],
                order_depth,
                PICNIC_BASKET2_position,
                buy_order_volume,
                sell_order_volume
            )
            print(f"[Run] Clearing Orders: {clear_orders}")

            # ----- STEP 4: Market Making (Optional) -----
            # Uncomment the following lines to include market making orders:
            # mm_orders, _, _ = self.market_make_PICNIC_BASKET2(
            #     Product.PICNIC_BASKET2,
            #     order_depth,
            #     fair_value,  # Using the OLS-derived fair value as a reference for both sides
            #     fair_value,
            #     PICNIC_BASKET2_position,
            #     buy_order_volume,
            #     sell_order_volume
            # )
            # print(f"[Run] Market Making Orders: {mm_orders}")

            # Combine orders from opportunity taking and clearing (add mm_orders if enabled)
            result[Product.PICNIC_BASKET2] = take_orders + clear_orders  # + mm_orders if market making is enabled
            print(f"[Run] Combined Orders for {Product.PICNIC_BASKET2}: {result[Product.PICNIC_BASKET2]}")

        conversions = 0
        traderData = jsonpickle.encode(traderObject)
        print(f"[Run] Updated trader state: {traderObject}")
        return result, conversions, traderData
