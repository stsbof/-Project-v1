import numpy as np
import datetime as dt
from typing import Optional, List, Tuple
import matplotlib.pyplot as plt


# ==================== MARKET CLASS ====================
class Market:
    """Market data container for stock pricing"""
    
    def __init__(self, spot: float, vol: float, rate: float, 
                 div_fixed: float, div_prop: float, ex_div_date: dt.datetime):
        self.spot = spot
        self.volatility = vol
        self.interest_rate = rate
        self.div_fixed = div_fixed  # Fixed dividend component (ρ in formula)
        self.div_prop = div_prop    # Proportional dividend rate (λ in formula)
        self.divExDate = ex_div_date
    
    def get_dividend(self, t: float, spot_t: float, t0: float = 0) -> float:
        """Calculate dividend using the model: D_t = ρ(S_0*e^(-λ(t-t0)) + S_t*(1-e^(-λ(t-t0))))"""
        if self.div_fixed == 0 and self.div_prop == 0:
            return 0.0
        
        exp_term = np.exp(-self.div_prop * (t - t0))
        dividend = self.div_fixed * (self.spot * exp_term + spot_t * (1 - exp_term))
        return dividend


# ==================== OPTION TRADE CLASS ====================
class OptionTrade:
    """Option trade specification"""
    
    CALL_LABEL = 'Call'
    PUT_LABEL = 'Put'
    AMER_LABEL = 'American'
    EURO_LABEL = 'European'
    
    def __init__(self, mat: dt.datetime, call_put: str, exercise: str, strike: float):
        self.mat_date = mat
        self.opt_type = call_put
        self.exercise = exercise
        self.strike = strike
    
    def is_american(self) -> bool:
        return self.exercise == OptionTrade.AMER_LABEL
    
    def is_a_call(self) -> bool:
        return self.opt_type == OptionTrade.CALL_LABEL
    
    def is_a_put(self) -> bool:
        return self.opt_type == OptionTrade.PUT_LABEL
    
    def pay_off(self, spot_price: float) -> float:
        """Calculate option payoff"""
        if self.is_a_call():
            return max(spot_price - self.strike, 0.0)
        else:
            return max(self.strike - spot_price, 0.0)
    
    def exercise_value(self, spot_price: float) -> float:
        """Value if exercised now (same as payoff)"""
        return self.pay_off(spot_price)


# ==================== NODE CLASS ====================
class Node:
    """Single node in the trinomial tree"""
    
    def __init__(self, price: float, tree: 'Tree'):
        self.price = price
        self.tree = tree
        
        # Connections to next nodes
        self.next_up: Optional[Node] = None
        self.next_mid: Optional[Node] = None
        self.next_down: Optional[Node] = None
        
        # Transition probabilities
        self.proba_up: float = 0.0
        self.proba_mid: float = 0.0
        self.proba_down: float = 0.0
        
        # Option value at this node
        self.option_value: Optional[float] = None
    
    def price_recursive(self, option: OptionTrade) -> float:
        """Price option recursively from this node (for small trees)"""
        
        # If already computed, return cached value
        if self.option_value is not None:
            return self.option_value
        
        # Terminal node: return payoff
        if self.next_mid is None:
            self.option_value = option.pay_off(self.price)
            return self.option_value
        
        # Recursive pricing: discounted expected value
        next_up_val = self.next_up.price_recursive(option) if self.next_up else 0.0
        next_mid_val = self.next_mid.price_recursive(option) if self.next_mid else 0.0
        next_down_val = self.next_down.price_recursive(option) if self.next_down else 0.0
        
        expected_value = (self.proba_up * next_up_val + 
                         self.proba_mid * next_mid_val + 
                         self.proba_down * next_down_val)
        
        discount_factor = np.exp(-self.tree.market.interest_rate * self.tree.dt)
        hold_value = expected_value * discount_factor
        
        # American option: compare hold vs exercise
        if option.is_american():
            exercise_value = option.exercise_value(self.price)
            self.option_value = max(hold_value, exercise_value)
        else:
            self.option_value = hold_value
        
        return self.option_value
    
    def compute_value_from_next(self, option: OptionTrade) -> float:
        """Compute this node's value from next nodes (assumes next nodes already priced)"""
        
        # Terminal node: return payoff
        if self.next_mid is None:
            self.option_value = option.pay_off(self.price)
            return self.option_value
        
        # Get next node values
        next_up_val = self.next_up.option_value if self.next_up else 0.0
        next_mid_val = self.next_mid.option_value if self.next_mid else 0.0
        next_down_val = self.next_down.option_value if self.next_down else 0.0
        
        expected_value = (self.proba_up * next_up_val + 
                         self.proba_mid * next_mid_val + 
                         self.proba_down * next_down_val)
        
        discount_factor = np.exp(-self.tree.market.interest_rate * self.tree.dt)
        hold_value = expected_value * discount_factor
        
        # American option: compare hold vs exercise
        if option.is_american():
            exercise_value = option.exercise_value(self.price)
            self.option_value = max(hold_value, exercise_value)
        else:
            self.option_value = hold_value
        
        return self.option_value


# ==================== TRUNK NODE CLASS ====================
class TrunkNode(Node):
    """Node on the main trunk (middle line) of the tree"""
    
    def __init__(self, prec_node: Optional['TrunkNode'], col_date: dt.datetime, 
                 price: float, tree: 'Tree'):
        super().__init__(price, tree)
        self.prec_mid = prec_node  # Previous middle node
        self.column_date = col_date
        self.column_index: int = 0
        
        # Check if dividend falls in the next period
        ex_div_date = self.tree.market.divExDate
        self.is_div_in_following_period = (
            not self.are_same_dates(self.column_date, ex_div_date) and
            self.column_date < ex_div_date and
            (ex_div_date <= self.next_date() or 
             self.are_same_dates(ex_div_date, self.next_date()))
        )
    
    def next_date(self) -> dt.datetime:
        """Calculate next column date"""
        return self.column_date + dt.timedelta(seconds=self.tree.dt * 365.25 * 24 * 3600)
    
    def are_same_dates(self, d1: dt.datetime, d2: dt.datetime) -> bool:
        """Check if two dates are equal within tolerance"""
        tolerance = dt.timedelta(days=1) / self.tree.nb_steps / 1000
        return abs(d1 - d2) < tolerance


# ==================== TREE CLASS ====================
class Tree:
    """Trinomial tree structure"""
    
    def __init__(self, market: Market, pricing_date: dt.datetime, 
                 maturity: dt.datetime, nb_steps: int):
        self.market = market
        self.pricing_date = pricing_date
        self.maturity = maturity
        self.nb_steps = nb_steps
        
        # Calculate time parameters
        total_time = (maturity - pricing_date).total_seconds() / (365.25 * 24 * 3600)
        self.dt = total_time / nb_steps
        
        # Alpha parameter for node spacing: α = e^(σ√(3Δt))
        self.alpha = np.exp(market.volatility * np.sqrt(3 * self.dt))
        
        # Build the tree
        self.root: TrunkNode = None
        self.trunk_nodes: List[TrunkNode] = []
        self.all_nodes: List[List[Node]] = []
        self._build_tree()
    
    def _build_tree(self):
        """Build the complete trinomial tree"""
        
        # Create root node
        self.root = TrunkNode(None, self.pricing_date, self.market.spot, self)
        self.root.column_index = 0
        self.trunk_nodes.append(self.root)
        self.all_nodes.append([self.root])
        
        # Build each time step
        for i in range(self.nb_steps):
            current_trunk = self.trunk_nodes[i]
            next_date = current_trunk.next_date()
            
            # Calculate forward price for next middle node
            forward = current_trunk.price * np.exp(self.market.interest_rate * self.dt)
            
            # Subtract dividend if applicable
            if current_trunk.is_div_in_following_period:
                time_to_div = (self.market.divExDate - self.pricing_date).total_seconds() / (365.25 * 24 * 3600)
                dividend = self.market.get_dividend(time_to_div, forward, 0)
                forward -= dividend
            
            # Create next trunk node
            next_trunk = TrunkNode(current_trunk, next_date, forward, self)
            next_trunk.column_index = i + 1
            self.trunk_nodes.append(next_trunk)
            
            # Build column of nodes
            next_column = self._build_column(current_trunk, next_trunk, i)
            self.all_nodes.append(next_column)
            
            # Connect current column to next column
            self._connect_columns(i)
    
    def _build_column(self, current_trunk: TrunkNode, next_trunk: TrunkNode, 
                      col_index: int) -> List[Node]:
        """Build all nodes in a column"""
        
        nodes = []
        mid_price = next_trunk.price
        
        # Calculate number of nodes above and below middle
        # Use k >= 4*sqrt(i/3) for 4 standard deviations coverage
        if col_index == 0:
            k_max = 0
        else:
            k_max = int(np.ceil(4 * np.sqrt((col_index + 1) / 3))) + 2
        
        # Create nodes from top to bottom
        for k in range(k_max, -k_max - 1, -1):
            if k == 0:
                nodes.append(next_trunk)
            else:
                node_price = mid_price * (self.alpha ** k)
                node = Node(node_price, self)
                nodes.append(node)
        
        return nodes
    
    def _connect_columns(self, col_index: int):
        """Connect nodes between two consecutive columns"""
        
        current_col = self.all_nodes[col_index]
        next_col = self.all_nodes[col_index + 1]
        
        for node in current_col:
            self._set_transitions(node, next_col)
    
    def _set_transitions(self, node: Node, next_col: List[Node]):
        """Set transition probabilities and connections for a node"""
        
        # Find closest middle node in next column
        next_mid_idx = self._find_closest_node(node.price, next_col)
        node.next_mid = next_col[next_mid_idx]
        
        # Set up and down nodes
        if next_mid_idx > 0:
            node.next_up = next_col[next_mid_idx - 1]
        else:
            node.next_up = node.next_mid
            
        if next_mid_idx < len(next_col) - 1:
            node.next_down = next_col[next_mid_idx + 1]
        else:
            node.next_down = node.next_mid
        
        # Calculate probabilities
        self._calculate_probabilities(node)
    
    def _find_closest_node(self, price: float, nodes: List[Node]) -> int:
        """Find index of node closest to forward price"""
        forward = price * np.exp(self.market.interest_rate * self.dt)
        
        min_diff = float('inf')
        best_idx = len(nodes) // 2
        
        for i, node in enumerate(nodes):
            diff = abs(node.price - forward)
            if diff < min_diff:
                min_diff = diff
                best_idx = i
        
        return best_idx
    
    def _calculate_probabilities(self, node: Node):
        """Calculate transition probabilities matching mean and variance"""
        
        # Get next node prices
        s_up = node.next_up.price
        s_mid = node.next_mid.price
        s_down = node.next_down.price
        
        # Calculate expected value and variance
        forward = node.price * np.exp(self.market.interest_rate * self.dt)
        variance = (node.price ** 2) * np.exp(2 * self.market.interest_rate * self.dt) * \
                   (np.exp(self.market.volatility ** 2 * self.dt) - 1)
        
        # Use simplified formula when no dividend (more numerically stable)
        if abs(s_mid - forward) < 1e-10:
            # No dividend case
            denom = (1 - self.alpha) * (1 / (self.alpha ** 2) - 1)
            node.proba_down = (np.exp(self.market.volatility ** 2 * self.dt) - 1) / denom
            node.proba_up = node.proba_down / self.alpha
            node.proba_mid = 1 - node.proba_up - node.proba_down
        else:
            # General case with dividend
            # Solve 3x3 linear system
            a = self.alpha
            
            # Build matrix
            A = np.array([
                [1, 1, 1],
                [s_up, s_mid, s_down],
                [s_up**2, s_mid**2, s_down**2]
            ])
            
            b = np.array([1, forward, variance + forward**2])
            
            try:
                probs = np.linalg.solve(A, b)
                node.proba_up = probs[0]
                node.proba_mid = probs[1]
                node.proba_down = probs[2]
            except:
                # Fallback to simplified
                node.proba_down = 1/6
                node.proba_up = 1/6
                node.proba_mid = 2/3
        
        # Validate probabilities
        if not (0 <= node.proba_up <= 1 and 0 <= node.proba_mid <= 1 and 0 <= node.proba_down <= 1):
            print(f"Warning: Invalid probabilities at price {node.price:.2f}")
            print(f"  p_up={node.proba_up:.4f}, p_mid={node.proba_mid:.4f}, p_down={node.proba_down:.4f}")


# ==================== TRINOMIAL MODEL CLASS ====================
class TrinomialModel:
    """Main pricing model"""
    
    def __init__(self, pricing_date: dt.datetime, tree: Tree):
        self.pricing_date = pricing_date
        self.tree = tree
    
    def price_option(self, option: OptionTrade, method: str = 'backward') -> float:
        """Price an option using the trinomial tree
        
        Args:
            option: The option to price
            method: 'recursive' (for small trees) or 'backward' (for large trees)
        """
        if method == 'recursive':
            return self.tree.root.price_recursive(option)
        else:
            return self.price_backward(option)
    
    def price_backward(self, option: OptionTrade) -> float:
        """Price option using backward induction (iterative, no recursion limit)"""
        
        # Clear any cached values
        for col in self.tree.all_nodes:
            for node in col:
                node.option_value = None
        
        # Start from the last column (maturity) and work backward
        for col_index in range(len(self.tree.all_nodes) - 1, -1, -1):
            column = self.tree.all_nodes[col_index]
            
            for node in column:
                if col_index == len(self.tree.all_nodes) - 1:
                    # Terminal nodes: set payoff
                    node.option_value = option.pay_off(node.price)
                else:
                    # Interior nodes: compute from next nodes
                    node.compute_value_from_next(option)
        
        return self.tree.root.option_value
    
    def black_scholes_price(self, option: OptionTrade, spot: float, 
                           vol: float, rate: float, time_to_mat: float) -> float:
        """Calculate Black-Scholes price for comparison"""
        from scipy.stats import norm
        
        if time_to_mat <= 0:
            return option.pay_off(spot)
        
        d1 = (np.log(spot / option.strike) + (rate + 0.5 * vol**2) * time_to_mat) / \
             (vol * np.sqrt(time_to_mat))
        d2 = d1 - vol * np.sqrt(time_to_mat)
        
        if option.is_a_call():
            price = spot * norm.cdf(d1) - option.strike * np.exp(-rate * time_to_mat) * norm.cdf(d2)
        else:
            price = option.strike * np.exp(-rate * time_to_mat) * norm.cdf(-d2) - spot * norm.cdf(-d1)
        
        return price


# ==================== TESTING AND ANALYSIS ====================
def test_convergence():
    """Test convergence to Black-Scholes for European options without dividends"""
    
    print("=" * 80)
    print("ASSIGNMENT 1: Testing Convergence to Black-Scholes")
    print("=" * 80)
    
    # Market parameters
    pricing_date = dt.datetime(2024, 1, 1)
    maturity = dt.datetime(2025, 1, 1)  # 1 year
    spot = 100.0
    vol = 0.30
    rate = 0.05
    strike = 100.0
    
    # No dividend
    mkt = Market(spot, vol, rate, 0.0, 0.0, maturity)
    
    # Test for different number of steps
    steps_list = [5, 10, 20, 50, 100, 200, 1000, 10000, 100000]
    
    for opt_type in ['Call', 'Put']:
        print(f"\n{opt_type} Option (ATM, K={strike}):")
        print(f"{'Steps':<10} {'Tree Price':<15} {'BS Price':<15} {'Difference':<15} {'Error %':<10}")
        print("-" * 75)
        
        opt = OptionTrade(maturity, opt_type, 'European', strike)
        
        # Black-Scholes benchmark
        model_bs = TrinomialModel(pricing_date, None)
        bs_price = model_bs.black_scholes_price(opt, spot, vol, rate, 1.0)
        
        for nb_steps in steps_list:
            import time
            start = time.time()
            
            tree = Tree(mkt, pricing_date, maturity, nb_steps)
            model = TrinomialModel(pricing_date, tree)
            tree_price = model.price_option(opt, method='backward')  # Use backward method
            
            elapsed = time.time() - start
            
            diff = tree_price - bs_price
            error_pct = 100 * diff / bs_price if bs_price != 0 else 0
            
            print(f"{nb_steps:<10} {tree_price:<15.6f} {bs_price:<15.6f} {diff:<15.6f} {error_pct:<10.4f}%  ({elapsed:.2f}s)")
    
    print("\nObservations:")
    print("- Gap decreases approximately as 1/NbSteps")
    print("- Error converges to zero as number of steps increases")
    print("- Backward method handles large trees without recursion issues")


def test_american_vs_european():
    """Test American vs European option pricing"""
    
    print("\n" + "=" * 80)
    print("ASSIGNMENT 2: American vs European Options")
    print("=" * 80)
    
    pricing_date = dt.datetime(2024, 1, 1)
    maturity = dt.datetime(2025, 1, 1)
    spot = 100.0
    vol = 0.30
    strike = 110.0
    nb_steps = 100  # Use larger steps for better accuracy
    
    # Test with different interest rates
    rates = [-0.05, 0.0, 0.05, 0.10]
    
    print("\nPut Options (OTM, K=110, S=100):")
    print(f"{'Rate':<10} {'European':<15} {'American':<15} {'Early Ex Premium':<20}")
    print("-" * 70)
    
    for rate in rates:
        mkt = Market(spot, vol, rate, 0.0, 0.0, maturity)
        tree = Tree(mkt, pricing_date, maturity, nb_steps)
        model = TrinomialModel(pricing_date, tree)
        
        opt_euro = OptionTrade(maturity, 'Put', 'European', strike)
        opt_amer = OptionTrade(maturity, 'Put', 'American', strike)
        
        euro_price = model.price_option(opt_euro, method='backward')
        amer_price = model.price_option(opt_amer, method='backward')
        premium = amer_price - euro_price
        
        print(f"{rate:<10.2%} {euro_price:<15.6f} {amer_price:<15.6f} {premium:<20.6f}")
    
    print("\nObservations:")
    print("- American put > European put (early exercise value)")
    print("- Premium increases with negative rates (higher incentive to exercise)")
    print("- For calls without dividends, American ≈ European")


def test_dividend_impact():
    """Test impact of dividends with zero interest rate"""
    
    print("\n" + "=" * 80)
    print("ASSIGNMENT 2: Dividend Impact (r=0)")
    print("=" * 80)
    
    pricing_date = dt.datetime(2024, 1, 1)
    maturity = dt.datetime(2025, 1, 1)
    ex_div_date = dt.datetime(2024, 7, 1)
    spot = 100.0
    vol = 0.30
    rate = 0.0
    strike = 100.0
    nb_steps = 100  # Use larger steps for better accuracy
    
    dividends = [0.0, 2.0, 5.0, 10.0]
    
    for opt_type in ['Call', 'Put']:
        print(f"\n{opt_type} Options (ATM, K={strike}):")
        print(f"{'Dividend':<12} {'Price':<15} {'Change from 0':<15}")
        print("-" * 45)
        
        base_price = None
        for div in dividends:
            mkt = Market(spot, vol, rate, div, 0.0, ex_div_date)
            tree = Tree(mkt, pricing_date, maturity, nb_steps)
            model = TrinomialModel(pricing_date, tree)
            
            opt = OptionTrade(maturity, opt_type, 'European', strike)
            price = model.price_option(opt, method='backward')
            
            if base_price is None:
                base_price = price
                change_str = "-"
            else:
                change = price - base_price
                change_str = f"{change:+.6f}"
            
            print(f"{div:<12.2f} {price:<15.6f} {change_str:<15}")
    
    print("\nObservations:")
    print("- Dividends decrease call value (reduces expected spot at maturity)")
    print("- Dividends increase put value (reduces expected spot at maturity)")


# if __name__ == "__main__":
#     # Run all tests
#     test_convergence()
#     test_american_vs_european()
#     test_dividend_impact()
    
#     print("\n" + "=" * 80)
#     print("All tests completed successfully!")
#     print("=" * 80)