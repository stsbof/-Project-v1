import math as ma
from datetime import datetime
from scipy.stats import norm


# ======================
# 🔹 Classe Market
# ======================
class Market:
    def __init__(self, stock_price, int_rate, vol, div_date=None, dividende=0.0):
        self.stock_price = stock_price
        self.int_rate = int_rate
        self.vol = vol
        self.div_date = div_date
        self.dividende = dividende


# ======================
# 🔹 Classe Contract (Option)
# ======================
class Contract:
    def __init__(self, strike, maturity, pricing_date, op_type="Call", op_exercice="EU"):
        """
        maturity : soit un float (en années), soit une datetime (date d'échéance)
        op_type : "Call" ou "Put"
        op_exercice : "EU" ou "US"
        """
        self.strike = strike
        self.pricing_date = pricing_date
        self.op_type = op_type
        self.op_exercice = op_exercice

        # ✅ Si maturity est une date, on calcule la durée en années
        if isinstance(maturity, datetime):
            self.maturity_date = maturity
            self.maturity = (maturity - pricing_date).days / 365.0
        else:
            self.maturity_date = None
            self.maturity = float(maturity)



# ======================
# 🔹 Classe Noeud (Node)
# ======================
class Noeud:
    def __init__(self, v, arbre=None):
        self.v = v
        self.v2 = None
        self.arbre = arbre
        self.next_mid = None
        self.voisin_up = None
        self.voisin_down = None
        self.voisin_behind = None
        self.proba_next_up = None
        self.proba_next_mid = None
        self.proba_next_down = None

    def move_up(self, alpha):
        if self.voisin_up is not None:
            return self.voisin_up
        else:
            self.voisin_up = Noeud(self.v * alpha, arbre=self.arbre)
            self.voisin_up.voisin_down = self
            return self.voisin_up

    def move_down(self, alpha):
        if self.voisin_down is not None:
            return self.voisin_down
        else:
            self.voisin_down = Noeud(self.v / alpha, arbre=self.arbre)
            self.voisin_down.voisin_up = self
            return self.voisin_down

    def add_probability(self, D):
        r = self.arbre.market.int_rate
        dt = self.arbre.dt
        alpha = self.arbre.alpha
        sigma = self.arbre.market.vol

        Espe = self.v * ma.exp(r * dt) - D
        var = (self.v ** 2) * ma.exp(2 * r * dt) * (ma.exp(sigma ** 2 * dt) - 1)

        p_down = (
            (self.next_mid.v ** (-2) * (var + Espe ** 2))
            - 1
            - (alpha + 1) * (self.next_mid.v ** (-1) * Espe - 1)
        ) / ((1 - alpha) * (alpha ** (-2) - 1))

        p_up = (
            (self.next_mid.v ** (-1) * Espe - 1 - (alpha ** (-1) - 1) * p_down)
            / (alpha - 1)
        )

        p_mid = 1 - p_up - p_down

        self.proba_next_up = p_up
        self.proba_next_mid = p_mid
        self.proba_next_down = p_down

    def good_next_mid(self, forward, last_next_mid, D):
        alpha = self.arbre.alpha
        self.next_mid = find_next_mid(forward, alpha, last_next_mid)
        self.next_mid.voisin_behind = self
        self.next_mid.voisin_up = self.next_mid.move_up(alpha)
        self.next_mid.voisin_down = self.next_mid.move_down(alpha)
        self.add_probability(D)
        return self.next_mid


# ======================
# 🔹 Classe Arbre
# ======================
class Arbre:
    def __init__(self, market, contract, n_steps):
        self.market = market
        self.contract = contract
        self.n_steps = n_steps
        self.dt = contract.maturity / n_steps
        self.alpha = ma.exp(market.vol * ma.sqrt(3 * self.dt))
        self.racine = None
        self.generer_arbre(n_steps)

    def generer_arbre(self, N):
        market = self.market
        contract = self.contract

        if market.div_date is not None:
            T_div_date = (market.div_date - contract.pricing_date).days / 365.0
        else:
            T_div_date = contract.maturity + 1000000

        S0 = market.stock_price
        r = market.int_rate
        dividende = market.dividende
        D = 0.0
        dt = self.dt
        current_date = 0.0

        noeud = Noeud(S0, arbre=self)
        self.racine = noeud
        noeud_tronc = noeud

        for k in range(1, N + 1):
            if T_div_date > current_date and T_div_date <= current_date + dt:
                D = dividende

            last_next_mid = Noeud(noeud_tronc.v * ma.exp(r * dt) - D, arbre=self)

            noeud = noeud_tronc
            while noeud is not None:
                last_next_mid = noeud.good_next_mid(
                    noeud.v * ma.exp(r * dt) - D, last_next_mid, D
                )
                noeud = noeud.voisin_up

            noeud = noeud_tronc
            last_next_mid = noeud_tronc.next_mid
            while noeud is not None:
                last_next_mid = noeud.good_next_mid(
                    noeud.v * ma.exp(r * dt) - D, last_next_mid, D
                )
                noeud = noeud.voisin_down

            noeud_tronc = noeud_tronc.next_mid
            current_date += dt
            D = 0.0


# ======================
# 🔹 Fonctions utilitaires
# ======================
def find_next_mid(forward, alpha, node):
    while forward > node.v * (1 + alpha) / 2:
        node = node.move_up(alpha)
    while forward <= node.v * (1 + 1 / alpha) / 2:
        if forward < 0:
            break
        node = node.move_down(alpha)
    return node


# ======================
# 🔹 Fonctions de pricing
# ======================
def comput_payoff(op_multiplicator, last_node, K):
    current_node = last_node
    while current_node is not None:
        current_node.v2 = max((current_node.v - K) * op_multiplicator, 0)
        current_node = current_node.voisin_up
    current_node = last_node
    while current_node is not None:
        current_node.v2 = max((current_node.v - K) * op_multiplicator, 0)
        current_node = current_node.voisin_down


def pricing(last_node, d_f, op_exercice, K, op_multiplicator):
    while last_node is not None:
        last_node = last_node.voisin_behind
        current_node = last_node

        while current_node is not None:
            u = current_node.proba_next_up
            pm = current_node.proba_next_mid
            d = current_node.proba_next_down
            val = (
                u * current_node.next_mid.voisin_up.v2
                + pm * current_node.next_mid.v2
                + d * current_node.next_mid.voisin_down.v2
            ) * d_f

            if op_exercice == "US":
                current_node.v2 = max(
                    val, max((current_node.v - K) * op_multiplicator, 0)
                )
            else:
                current_node.v2 = val
            current_node = current_node.voisin_up

        current_node = last_node
        while current_node is not None:
            u = current_node.proba_next_up
            pm = current_node.proba_next_mid
            d = current_node.proba_next_down
            val = (
                u * current_node.next_mid.voisin_up.v2
                + pm * current_node.next_mid.v2
                + d * current_node.next_mid.voisin_down.v2
            ) * d_f

            if op_exercice == "US":
                current_node.v2 = max(
                    val, max((current_node.v - K) * op_multiplicator, 0)
                )
            else:
                current_node.v2 = val
            current_node = current_node.voisin_down


def pricer(arbre):
    r = arbre.market.int_rate
    N = arbre.n_steps
    T = arbre.contract.maturity
    K = arbre.contract.strike
    dt = T / N
    d_f = ma.exp(-r * dt)
    op_type = arbre.contract.op_type
    op_exercice = arbre.contract.op_exercice
    op_multiplicator = 1 if op_type == "Call" else -1

    last_node = arbre.racine
    while last_node.next_mid is not None:
        last_node = last_node.next_mid

    comput_payoff(op_multiplicator, last_node, K)
    pricing(last_node, d_f, op_exercice, K, op_multiplicator)
    return arbre


# ======================
# 🔹 Black–Scholes
# ======================
def BS(S, K, T, r, sigma, type_op, exercice="EU"):
    N = norm.cdf
    n = norm.pdf

    d1 = (ma.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * ma.sqrt(T))
    d2 = d1 - sigma * ma.sqrt(T)

    if type_op == "Call":
        price = S * N(d1) - K * ma.exp(-r * T) * N(d2)
        delta = N(d1)
    else:
        price = K * ma.exp(-r * T) * N(-d2) - S * N(-d1)
        delta = N(d1) - 1

    gamma = n(d1) / (S * sigma * ma.sqrt(T))
    vega = S * n(d1) * ma.sqrt(T) / 100
    vomma = (vega * d1 * d2 / sigma) / 100
    vanna = (-n(d1) * d2 / sigma) / 100

    return {
        "price": price,
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "vomma": vomma,
        "vanna": vanna,
        "note": "⚠️ Pour une option américaine, ces valeurs sont approximatives."
                if exercice == "US" else "Option européenne : formules exactes."
    }


# ======================
# 🔹 Calcul du Delta à partir de l’arbre
# ======================
def delta_from_tree(arbre, step=2):
    node = arbre.racine
    for _ in range(step):
        if node.next_mid is None:
            return None
        node = node.next_mid

    node_up = node.voisin_up
    node_down = node.voisin_down
    if node_up is None or node_down is None:
        return None

    S_up, S_down = node_up.v, node_down.v
    V_up, V_down = node_up.v2, node_down.v2
    delta = (V_up - V_down) / (S_up - S_down)

    return delta, (S_up, S_down, V_up, V_down)


def vega_from_tree(market, contract, n_steps=400, bump=0.01):
    """
    Calcule le Vega numérique à partir de l’arbre :
    Vega ≈ (V(sigma + bump) - V(sigma - bump)) / (2 * bump)
    bump exprimé en décimal (ex: 0.01 = 1%)
    """
    sigma0 = market.vol

    # Monte un arbre avec sigma + bump
    market_up = Market(market.stock_price, market.int_rate, sigma0 + bump,
                       market.div_date, market.dividende)
    arbre_up = pricer(Arbre(market_up, contract, n_steps))
    price_up = arbre_up.racine.v2

    # Monte un arbre avec sigma - bump
    market_down = Market(market.stock_price, market.int_rate, sigma0 - bump,
                         market.div_date, market.dividende)
    arbre_down = pricer(Arbre(market_down, contract, n_steps))
    price_down = arbre_down.racine.v2

    # Approximation du Vega
    vega_tree = (price_up - price_down) / (2 * bump) / 100  # divisé par 100 pour exprimer en "par %"
    return vega_tree



# ======================
# 🔹 Programme principal
# ======================
if __name__ == "__main__":
    t0 = datetime(2025, 9, 1)
    div_date = datetime(2026, 4, 21)

    market = Market(100, 0.05, 0.3, div_date, 3.0)
    contract = Contract(102, datetime(2026, 9, 1), t0, "Call", "US")

    arbre = pricer(Arbre(market, contract, 400))
    price_tree = arbre.racine.v2
    bs = BS(market.stock_price, contract.strike, contract.maturity,
            market.int_rate, market.vol, contract.op_type, contract.op_exercice)

    # --- Delta basique à partir du 2e step ---
    delta_tree, (S_up, S_down, V_up, V_down) = delta_from_tree(arbre, step=2)

    print("\n==============================")
    print(f"   OPTION {contract.op_type} {contract.op_exercice}")
    print("==============================")
    print(f"Prix trinomial : {price_tree:.6f}")
    print(f"Prix Black–Scholes : {bs['price']:.6f}")

    print(f"\n--- 📉 Delta basique (à partir du 2ᵉ step) ---")
    print(f"S_up = {S_up:.4f}, S_down = {S_down:.4f}")
    print(f"V_up = {V_up:.6f}, V_down = {V_down:.6f}")
    print(f"➡️ Delta (arbre) = {delta_tree:.6f}")

    print(f"\n--- 📈 Delta analytique (Black–Scholes) ---")
    print(f"Delta BS : {bs['delta']:.6f}")
    print(f"\nNote : {bs['note']}")
    # --- Vega numérique à partir de l’arbre ---
    vega_tree = vega_from_tree(market, contract, 400)

    print(f"\n--- 🌪 Vega ---")
    print(f"Vega (arbre trinomial) ≈ {vega_tree:.6f}")
    print(f"Vega (Black–Scholes)  = {bs['vega']:.6f}")