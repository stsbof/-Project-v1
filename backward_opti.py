import math as ma
from datetime import datetime
from scipy.stats import norm



class Market:
    def __init__(self, stock_price, int_rate, vol, div_date=None, dividende=0.0):
        self.stock_price = stock_price
        self.int_rate = int_rate
        self.vol = vol
        self.div_date = div_date
        self.dividende = dividende



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

        # Si maturity est une date, on calcule la durée en années
        if isinstance(maturity, datetime):
            self.maturity_date = maturity
            self.maturity = (maturity - pricing_date).days / 365.0
        else:
            self.maturity_date = None
            self.maturity = float(maturity)



class Noeud:
    __slots__ = ['v', 'v2', 'arbre', 'next_mid', 'voisin_up', 'voisin_down',
                 'voisin_behind', 'proba_next_up', 'proba_next_mid', 'proba_next_down',
                 'proba_reach']

    def __init__(self, v, arbre=None, proba_reach=None):
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
        self.proba_reach = proba_reach

    def move_up(self, alpha):
        if self.voisin_up is None:
            self.voisin_up = Noeud(self.v * alpha, arbre=self.arbre)
            self.voisin_up.voisin_down = self
        return self.voisin_up

    def move_down(self, alpha):
        if self.voisin_down is None:
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

        inv_mid_v = 1.0 / self.next_mid.v
        inv_mid_v_sq = inv_mid_v * inv_mid_v

        p_down = (
                         (inv_mid_v_sq * (var + Espe ** 2))
                         - 1
                         - (alpha + 1) * (inv_mid_v * Espe - 1)
                 ) / ((1 - alpha) * (alpha ** (-2) - 1))

        p_up = (
                (inv_mid_v * Espe - 1 - (alpha ** (-1) - 1) * p_down)
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



class Arbre:
    def __init__(self, market, contract, n_steps, pruning_threshold=1e-15):
        """
        pruning_threshold : seuil de probabilité en dessous duquel on NE CRÉE PAS le nœud
        """
        self.market = market
        self.contract = contract
        self.n_steps = n_steps
        self.dt = contract.maturity / n_steps
        self.alpha = ma.exp(market.vol * ma.sqrt(3 * self.dt))
        self.racine = None
        self.pruning_threshold = pruning_threshold
        self.pruned_nodes_count = 0
        self.total_nodes_count = 0

        # Cache pour éviter recalculs
        self.exp_r_dt = ma.exp(market.int_rate * self.dt)
        self.inv_alpha = 1.0 / self.alpha

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

        # Racine avec probabilité 1
        noeud = Noeud(S0, arbre=self, proba_reach=1.0)
        self.racine = noeud
        noeud_tronc = noeud
        self.total_nodes_count = 1

        for k in range(1, N + 1):
            if T_div_date > current_date and T_div_date <= current_date + dt:
                D = dividende

            forward = noeud_tronc.v * self.exp_r_dt - D
            last_next_mid = Noeud(forward, arbre=self)

            # Construire l'arbre et propager les probabilités en UNE SEULE PASSE
            # Montée
            noeud = noeud_tronc
            while noeud is not None:
                last_next_mid = noeud.good_next_mid(noeud.v * self.exp_r_dt - D, last_next_mid, D)

                # Propager les probabilités d'atteinte immédiatement
                if noeud.proba_reach is not None and noeud.proba_reach >= self.pruning_threshold:
                    self._update_next_probabilities(noeud)

                noeud = noeud.voisin_up

            # Descente
            noeud = noeud_tronc
            last_next_mid = noeud_tronc.next_mid
            while noeud is not None:
                last_next_mid = noeud.good_next_mid(noeud.v * self.exp_r_dt - D, last_next_mid, D)

                # Propager les probabilités d'atteinte immédiatement
                if noeud.proba_reach is not None and noeud.proba_reach >= self.pruning_threshold:
                    self._update_next_probabilities(noeud)

                noeud = noeud.voisin_down

            # Appliquer le pruning : transformer en monomial branching
            self._apply_pruning(noeud_tronc)

            noeud_tronc = noeud_tronc.next_mid
            current_date += dt
            D = 0.0

    def _update_next_probabilities(self, node):
        """Mise à jour optimisée des probabilités d'atteinte"""
        if node.next_mid is None:
            return

        current_proba = node.proba_reach

        # Mise à jour next_mid
        if node.next_mid.proba_reach is None:
            node.next_mid.proba_reach = 0.0
        node.next_mid.proba_reach += current_proba * node.proba_next_mid

        # Mise à jour voisin_up
        if node.next_mid.voisin_up is not None:
            if node.next_mid.voisin_up.proba_reach is None:
                node.next_mid.voisin_up.proba_reach = 0.0
            node.next_mid.voisin_up.proba_reach += current_proba * node.proba_next_up

        # Mise à jour voisin_down
        if node.next_mid.voisin_down is not None:
            if node.next_mid.voisin_down.proba_reach is None:
                node.next_mid.voisin_down.proba_reach = 0.0
            node.next_mid.voisin_down.proba_reach += current_proba * node.proba_next_down

    def _apply_pruning(self, trunk_node):
        """Applique le pruning en transformant les nœuds à faible probabilité en monomial"""
        threshold = self.pruning_threshold

        # Parcourir depuis le trunk vers le haut
        node = trunk_node
        while node is not None:
            if node.next_mid is not None:
                # Vérifier les trois nœuds suivants
                next_up = node.next_mid.voisin_up
                next_mid = node.next_mid
                next_down = node.next_mid.voisin_down

                # Si le nœud actuel a une probabilité trop faible, monomial branching
                if next_mid.proba_reach is not None and next_mid.proba_reach < threshold:
                    node.proba_next_up = 0.0
                    node.proba_next_mid = 1.0
                    node.proba_next_down = 0.0
                    self.pruned_nodes_count += 1
                else:
                    self.total_nodes_count += 1

            node = node.voisin_up

        # Parcourir depuis le trunk vers le bas
        node = trunk_node.voisin_down
        while node is not None:
            if node.next_mid is not None:
                next_mid = node.next_mid

                if next_mid.proba_reach is not None and next_mid.proba_reach < threshold:
                    node.proba_next_up = 0.0
                    node.proba_next_mid = 1.0
                    node.proba_next_down = 0.0
                    self.pruned_nodes_count += 1
                else:
                    self.total_nodes_count += 1

            node = node.voisin_down



def find_next_mid(forward, alpha, node):
    threshold_up = (1 + alpha) / 2
    threshold_down = (1 + 1 / alpha) / 2

    while forward > node.v * threshold_up:
        node = node.move_up(alpha)
    while forward <= node.v * threshold_down:
        if forward < 0:
            break
        node = node.move_down(alpha)
    return node



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
    is_american = (op_exercice == "US")

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

            if is_american:
                intrinsic = max((current_node.v - K) * op_multiplicator, 0)
                current_node.v2 = max(val, intrinsic)
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

            if is_american:
                intrinsic = max((current_node.v - K) * op_multiplicator, 0)
                current_node.v2 = max(val, intrinsic)
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


def vega_from_tree(market, contract, n_steps=400, bump=0.01, pruning_threshold=1e-15):
    """
    Calcule le Vega numérique à partir de l'arbre :
    Vega ≈ (V(sigma + bump) - V(sigma - bump)) / (2 * bump)
    """
    sigma0 = market.vol

    market_up = Market(market.stock_price, market.int_rate, sigma0 + bump,
                       market.div_date, market.dividende)
    arbre_up = pricer(Arbre(market_up, contract, n_steps, pruning_threshold))
    price_up = arbre_up.racine.v2

    market_down = Market(market.stock_price, market.int_rate, sigma0 - bump,
                         market.div_date, market.dividende)
    arbre_down = pricer(Arbre(market_down, contract, n_steps, pruning_threshold))
    price_down = arbre_down.racine.v2

    vega_tree = (price_up - price_down) / (2 * bump) / 100
    return vega_tree


# ======================
#  Programme principal
# ======================
if __name__ == "__main__":
    import time

    t0 = datetime(2025, 9, 1)
    div_date = datetime(2026, 4, 21)

    market = Market(100, 0.05, 0.3, div_date, 3.0)
    contract = Contract(102, datetime(2026, 9, 1), t0, "Call", "US")

    print("\n" + "=" * 60)
    print("   PRICING AVEC ARBRE TRINOMIAL + PRUNING")
    print("=" * 60)

    # Avec pruning optimisé
    print("\n🌳 ARBRE AVEC PRUNING (seuil = 1e-9)")
    start_time = time.time()
    arbre_proba = pricer(Arbre(market, contract, 1000, pruning_threshold=1e-9))
    elapsed_time = time.time() - start_time

    price_proba = arbre_proba.racine.v2
    print(f"Prix : {price_proba:.6f}")
    print(f"Nœuds prunés : {arbre_proba.pruned_nodes_count}")
    print(f"Nœuds totaux créés : {arbre_proba.total_nodes_count}")
    print(f"⏱️  Temps d'exécution : {elapsed_time:.4f} secondes")

    # Black-Scholes pour référence
    bs = BS(market.stock_price, contract.strike, contract.maturity,
            market.int_rate, market.vol, contract.op_type, contract.op_exercice)

    print(f"\n📊 BLACK-SCHOLES (référence)")
    print(f"Prix : {bs['price']:.6f}")
    print(f"Écart : {abs(price_proba - bs['price']):.6f} ({abs(price_proba - bs['price']) / bs['price'] * 100:.4f}%)")
    print(f"Note : {bs['note']}")

    # Delta
    delta_tree, (S_up, S_down, V_up, V_down) = delta_from_tree(arbre_proba, step=2)
    print(f"\n📉 DELTA")
    print(f"Delta (arbre avec pruning) : {delta_tree:.6f}")
    print(f"Delta (Black–Scholes) : {bs['delta']:.6f}")
    print(f"Écart : {abs(delta_tree - bs['delta']):.6f}")



    # Vega
    print(f"\n🌪 VEGA")
    start_time_vega = time.time()
    vega_tree = vega_from_tree(market, contract, 400, pruning_threshold=1e-9)
    elapsed_time_vega = time.time() - start_time_vega
    print(f"Vega (arbre avec pruning) : {vega_tree:.6f}")
    print(f"Vega (Black–Scholes) : {bs['vega']:.6f}")
    print(f"Écart : {abs(vega_tree - bs['vega']):.6f}")
    print(f"⏱️  Temps d'exécution Vega : {elapsed_time_vega:.4f} secondes")

    print("\n" + "=" * 60)

