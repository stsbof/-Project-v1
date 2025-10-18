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
        op_type: "Call" ou "Put"
        op_exercice: "EU" ou "US"
        """
        self.strike = strike
        self.maturity = maturity  # en années
        self.pricing_date = pricing_date
        self.op_type = op_type
        self.op_exercice = op_exercice


# ======================
# 🔹 Classe Noeud (Node)
# ======================
class Noeud:
    def __init__(self, v, arbre=None):
        self.v = v                # prix du sous-jacent
        self.v2 = None            # valeur de l'option
        self.arbre = arbre        # référence vers l'arbre
        self.next_mid = None      # mid du step suivant
        self.voisin_up = None     # voisin haut du même step
        self.voisin_down = None   # voisin bas du même step
        self.voisin_behind = None # mid du step précédent
        self.proba_next_up = None
        self.proba_next_mid = None
        self.proba_next_down = None

    def move_up(self, alpha):
        """Créer ou retourner le voisin supérieur."""
        if self.voisin_up is not None:
            return self.voisin_up
        else:
            self.voisin_up = Noeud(self.v * alpha, arbre=self.arbre)
            self.voisin_up.voisin_down = self
            return self.voisin_up

    def move_down(self, alpha):
        """Créer ou retourner le voisin inférieur."""
        if self.voisin_down is not None:
            return self.voisin_down
        else:
            self.voisin_down = Noeud(self.v / alpha, arbre=self.arbre)
            self.voisin_down.voisin_up = self
            return self.voisin_down

    def add_probability(self, D):
        """Calcule les probabilités (avec dividende D) selon les formules exactes."""
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
        """Construit le mid suivant, relie ses voisins, et calcule les proba locales."""
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
        """Construit l'arbre complet comme dans la version Excel, sans Excel."""
        market = self.market
        contract = self.contract

        # Détection de la date du dividende
        if market.div_date is not None:
            T_div_date = (market.div_date - contract.pricing_date).days / 365.0
        else:
            T_div_date = contract.maturity + 1000000  # jamais atteint

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

            # vers le haut
            noeud = noeud_tronc
            while noeud is not None:
                last_next_mid = noeud.good_next_mid(
                    noeud.v * ma.exp(r * dt) - D, last_next_mid, D
                )
                noeud = noeud.voisin_up

            # vers le bas
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
    """Trouve le bon next_mid dont la valeur encadre le forward."""
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
    """Applique le payoff à maturité."""
    current_node = last_node
    while current_node is not None:
        current_node.v2 = max((current_node.v - K) * op_multiplicator, 0)
        current_node = current_node.voisin_up
    current_node = last_node
    while current_node is not None:
        current_node.v2 = max((current_node.v - K) * op_multiplicator, 0)
        current_node = current_node.voisin_down


def pricing(last_node, d_f, op_exercice, K, op_multiplicator):
    """Backward induction par liens entre noeuds."""
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
    """Fonction principale de pricing."""
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
    return arbre.racine.v2


# ======================
# 🔹 Black–Scholes
# ======================
def BS(S, K, T, r, sigma, type_op):
    from scipy.stats import norm

    N = norm.cdf
    d1 = (ma.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * ma.sqrt(T))
    d2 = d1 - sigma * ma.sqrt(T)
    if type_op == "Call":
        return S * N(d1) - K * ma.exp(-r * T) * N(d2)
    else:
        return K * ma.exp(-r * T) * N(-d2) - S * N(-d1)


# ======================
# 🔍 TEST COMPLET
# ======================
if __name__ == "__main__":
    t0 = datetime(2025, 9, 1)
    div_date = datetime(2026, 4, 21)

    market = Market(stock_price=100, int_rate=0.05, vol=0.3,
                    div_date=div_date, dividende=3.0)
    contract = Contract(strike=102, maturity=1.0, pricing_date=t0,
                        op_type="Call", op_exercice="US")

    arbre = Arbre(market, contract, n_steps=400)
    price_tree = pricer(arbre)
    price_bs = BS(market.stock_price, contract.strike, contract.maturity,
                  market.int_rate, market.vol, contract.op_type)

    print(f"\nOption {contract.op_type} {contract.op_exercice}")
    print(f"Prix trinomial (avec dividende discret) : {price_tree:.6f}")
    print(f"Prix Black–Scholes (sans dividende)     : {price_bs:.6f}")
