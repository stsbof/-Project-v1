import math as ma
from datetime import datetime
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLineEdit, QPushButton, QTextEdit, QMessageBox, QComboBox
)
from PySide6.QtCore import Qt
import sys


# =========================================================
# 🔹 Classes financières
# =========================================================
class Market:
    def __init__(self, stock_price, int_rate, vol, div_date=None, dividende=0.0):
        self.stock_price = stock_price
        self.int_rate = int_rate
        self.vol = vol
        self.div_date = div_date
        self.dividende = dividende


class Contract:
    def __init__(self, strike, maturity, pricing_date, op_type="Call", op_exercice="EU"):
        self.strike = strike
        self.pricing_date = pricing_date
        self.op_type = op_type
        self.op_exercice = op_exercice
        if isinstance(maturity, (int, float)):
            self.maturity = float(maturity)
        elif isinstance(maturity, datetime):
            self.maturity = (maturity - pricing_date).days / 365.0
        else:
            raise ValueError("maturity doit être un nombre (années) ou une datetime")


# =========================================================
# 🔹 Arbre trinomial complet
# =========================================================
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
        if self.voisin_up:
            return self.voisin_up
        self.voisin_up = Noeud(self.v * alpha, self.arbre)
        self.voisin_up.voisin_down = self
        return self.voisin_up

    def move_down(self, alpha):
        if self.voisin_down:
            return self.voisin_down
        self.voisin_down = Noeud(self.v / alpha, self.arbre)
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
            (self.next_mid.v ** (-2) * (var + Espe ** 2)) - 1
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
        market, contract = self.market, self.contract
        if market.div_date:
            T_div_date = (market.div_date - contract.pricing_date).days / 365.0
        else:
            T_div_date = contract.maturity + 1000000  # jamais

        S0 = market.stock_price
        r = market.int_rate
        dividende = market.dividende
        D = 0.0
        dt = self.dt
        current_date = 0.0

        noeud_tronc = Noeud(S0, self)
        self.racine = noeud_tronc

        for _ in range(1, N + 1):
            if T_div_date > current_date and T_div_date <= current_date + dt:
                D = dividende

            last_next_mid = Noeud(noeud_tronc.v * ma.exp(r * dt) - D, self)

            # vers le haut
            noeud = noeud_tronc
            while noeud:
                last_next_mid = noeud.good_next_mid(noeud.v * ma.exp(r * dt) - D, last_next_mid, D)
                noeud = noeud.voisin_up

            # vers le bas
            noeud = noeud_tronc
            last_next_mid = noeud_tronc.next_mid
            while noeud:
                last_next_mid = noeud.good_next_mid(noeud.v * ma.exp(r * dt) - D, last_next_mid, D)
                noeud = noeud.voisin_down

            noeud_tronc = noeud_tronc.next_mid
            current_date += dt
            D = 0.0


def find_next_mid(forward, alpha, node):
    while forward > node.v * (1 + alpha) / 2:
        node = node.move_up(alpha)
    while forward <= node.v * (1 + 1 / alpha) / 2 and forward > 0:
        node = node.move_down(alpha)
    return node


def comput_payoff(op_multiplicator, last_node, K):
    current_node = last_node
    while current_node:
        current_node.v2 = max((current_node.v - K) * op_multiplicator, 0)
        current_node = current_node.voisin_up
    current_node = last_node
    while current_node:
        current_node.v2 = max((current_node.v - K) * op_multiplicator, 0)
        current_node = current_node.voisin_down


def pricing(last_node, d_f, op_exercice, K, op_multiplicator):
    while last_node:
        last_node = last_node.voisin_behind
        current_node = last_node

        while current_node:
            u = current_node.proba_next_up
            pm = current_node.proba_next_mid
            d = current_node.proba_next_down
            val = (
                u * current_node.next_mid.voisin_up.v2
                + pm * current_node.next_mid.v2
                + d * current_node.next_mid.voisin_down.v2
            ) * d_f

            if op_exercice == "US":
                current_node.v2 = max(val, max((current_node.v - K) * op_multiplicator, 0))
            else:
                current_node.v2 = val
            current_node = current_node.voisin_up

        current_node = last_node
        while current_node:
            u = current_node.proba_next_up
            pm = current_node.proba_next_mid
            d = current_node.proba_next_down
            val = (
                u * current_node.next_mid.voisin_up.v2
                + pm * current_node.next_mid.v2
                + d * current_node.next_mid.voisin_down.v2
            ) * d_f

            if op_exercice == "US":
                current_node.v2 = max(val, max((current_node.v - K) * op_multiplicator, 0))
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
    op_mult = 1 if op_type == "Call" else -1

    last_node = arbre.racine
    while last_node.next_mid:
        last_node = last_node.next_mid

    comput_payoff(op_mult, last_node, K)
    pricing(last_node, d_f, op_exercice, K, op_mult)
    return arbre.racine.v2


# =========================================================
# 🔹 Greeks numériques via l’arbre
# =========================================================
def compute_greeks(market, contract, n_steps, h=0.1, h_vol=0.01):
    base = pricer(Arbre(market, contract, n_steps))

    up = pricer(Arbre(Market(market.stock_price + h, market.int_rate, market.vol,
                             market.div_date, market.dividende), contract, n_steps))
    dn = pricer(Arbre(Market(market.stock_price - h, market.int_rate, market.vol,
                             market.div_date, market.dividende), contract, n_steps))

    delta_central = (up - dn) / (2 * h)
    gamma_num = (up - 2 * base + dn) / (h ** 2)

    up_vol = pricer(Arbre(Market(market.stock_price, market.int_rate, market.vol + h_vol,
                                 market.div_date, market.dividende), contract, n_steps))
    dn_vol = pricer(Arbre(Market(market.stock_price, market.int_rate, market.vol - h_vol,
                                 market.div_date, market.dividende), contract, n_steps))
    vega = (up_vol - dn_vol) / (2 * h_vol) / 100.0
    vomma = (up_vol - 2 * base + dn_vol) / (h_vol ** 2) / 10000.0

    return {
        "price": base,
        "delta_central": delta_central,
        "gamma": gamma_num,
        "vega": vega,
        "vomma": vomma
    }


# =========================================================
# 🔹 Interface graphique simplifiée (sans graphe)
# =========================================================
class OptionPricerGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("💎 Option Pricer (Modèle Arbre)")
        self.setGeometry(200, 100, 900, 650)

        # --- Styles modernes ---
        self.setStyleSheet("""
            QWidget { background-color: #f2f4f7; font-family: Segoe UI, Arial; font-size: 10.5pt; }
            QLineEdit, QComboBox {
                background: #ffffff; border: 1px solid #cfd6df; border-radius: 4px; padding: 4px;
                max-width: 150px; min-width: 110px; /* champs plus courts */
            }
            QLabel { color: #2c3e50; }
            QPushButton {
                background-color: #3487e2; color: white; border: none; border-radius: 7px;
                padding: 6px 10px; min-height: 28px; font-weight: 600;
            }
            QPushButton:hover { background-color: #2b6fb9; }
            QTextEdit {
                background: #fcfcfc; border: 1px solid #d7dde5; border-radius: 6px;
                font-family: Consolas, Menlo, monospace; font-size: 11pt; color: #1f2937;
            }
        """)

        main_layout = QVBoxLayout(self)
        top_layout = QHBoxLayout()
        left_form = QFormLayout()
        left_form.setFormAlignment(Qt.AlignTop)

        # --- Champs raccourcis
        self.s0 = QLineEdit("100")
        self.k = QLineEdit("102")
        self.r = QLineEdit("5")
        self.sigma = QLineEdit("30")
        self.dividende = QLineEdit("3")
        self.steps = QLineEdit("400")
        self.date_pricing = QLineEdit("2025-9-1")
        self.maturity = QLineEdit("1")
        self.div_date = QLineEdit("2026-4-21")
        self.type_op = QComboBox(); self.type_op.addItems(["Call", "Put"])
        self.exercice = QComboBox(); self.exercice.addItems(["EU", "US"])
        self.qty_options = QLineEdit("100")

        # --- Colonne gauche
        left_form.addRow("Sous-jacent (S0)", self.s0)
        left_form.addRow("Strike (K)", self.k)
        left_form.addRow("Taux sans risque (%)", self.r)
        left_form.addRow("Volatilité (%)", self.sigma)
        left_form.addRow("Dividende (montant)", self.dividende)
        left_form.addRow("Date valorisation", self.date_pricing)
        left_form.addRow("Maturité", self.maturity)
        left_form.addRow("Date dividende", self.div_date)
        left_form.addRow("Steps", self.steps)
        left_form.addRow("Type option", self.type_op)
        left_form.addRow("Exercice", self.exercice)
        left_form.addRow("Qté options", self.qty_options)

        # --- Colonne droite : boutons
        btn_col = QVBoxLayout()
        btn_col.setAlignment(Qt.AlignTop)
        btn_col.setSpacing(6)

        def tiny(b: QPushButton):
            b.setFixedWidth(150)
            b.setFixedHeight(30)
            return b

        self.calc_button = tiny(QPushButton("💰 Calculer"))
        self.calc_button.clicked.connect(self.calculate)
        btn_col.addWidget(self.calc_button)

        self.btns_greeks = {}
        for g in ["Delta", "Gamma", "Vega", "Vomma"]:
            b = tiny(QPushButton(g))
            b.clicked.connect(lambda _, name=g.lower(): self.show_greek(name))
            btn_col.addWidget(b)
            self.btns_greeks[g] = b

        self.btn_buy_option = tiny(QPushButton("🏦 Achat (banque)"))
        self.btn_sell_option = tiny(QPushButton("🏦 Vente (banque)"))
        self.btn_buy_option.clicked.connect(lambda: self.delta_hedge("buy"))
        self.btn_sell_option.clicked.connect(lambda: self.delta_hedge("sell"))
        btn_col.addWidget(self.btn_buy_option)
        btn_col.addWidget(self.btn_sell_option)

        self.btn_clear = tiny(QPushButton("🧹 Effacer sortie"))
        self.btn_clear.clicked.connect(self.clear_output)
        btn_col.addWidget(self.btn_clear)

        # --- Zone de texte résultat
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.output.setMinimumHeight(250)

        # --- Assemblage
        top_layout.addLayout(left_form, stretch=3)
        top_layout.addLayout(btn_col, stretch=1)
        main_layout.addLayout(top_layout)
        main_layout.addWidget(self.output)

        self.greeks = None

    # -----------------------------------------------------
    def clear_output(self):
        self.output.clear()

    # -----------------------------------------------------
    def _read_inputs(self):
        S0 = float(self.s0.text())
        K = float(self.k.text())
        r = float(self.r.text()) / 100.0
        sigma = float(self.sigma.text()) / 100.0
        dividende = float(self.dividende.text())
        n_steps = int(self.steps.text())
        type_op = self.type_op.currentText()
        ex = self.exercice.currentText()

        pricing_date = datetime.strptime(self.date_pricing.text(), "%Y-%m-%d")
        m_str = self.maturity.text().strip()
        try:
            maturity = datetime.strptime(m_str, "%Y-%m-%d")
        except ValueError:
            maturity = float(m_str)

        div_date_str = self.div_date.text().strip()
        div_date = datetime.strptime(div_date_str, "%Y-%m-%d") if div_date_str else None

        market = Market(S0, r, sigma, div_date, dividende)
        contract = Contract(K, maturity, pricing_date, type_op, ex)
        return market, contract, n_steps

    # -----------------------------------------------------
    def calculate(self):
        try:
            market, contract, n_steps = self._read_inputs()
            self.greeks = compute_greeks(market, contract, n_steps)

            self.output.clear()
            self.output.append("📘 <b>Résultats (Modèle Arbre)</b>\n" + "─"*70)
            self.output.append(f"\nPrix (arbre) : {self.greeks['price']:.6f}")
            self.output.append(f"Delta (central, arbre) : {self.greeks['delta_central']:.6f}")

        except Exception as e:
            QMessageBox.critical(self, "Erreur", str(e))

    # -----------------------------------------------------
    def show_greek(self, name):
        if not self.greeks:
            QMessageBox.warning(self, "⚠️ Attention", "Veuillez d'abord calculer.")
            return
        key_map = {"delta": "delta_central", "gamma": "gamma", "vega": "vega", "vomma": "vomma"}
        key = key_map.get(name, name)
        val = self.greeks.get(key, 0.0)
        label = {"delta": "Δ", "gamma": "Γ"}.get(name, name.capitalize())
        self.output.append(f"\n{label} (arbre) : {val:.6f}")

    # -----------------------------------------------------
    def delta_hedge(self, position):
        """Calcule la couverture Delta à partir du modèle arbre"""
        if not self.greeks:
            QMessageBox.warning(self, "⚠️ Attention", "Veuillez d'abord calculer pour obtenir le delta.")
            return
        try:
            delta = self.greeks["delta_central"]
            qty = float(self.qty_options.text())
            sign = 1 if position == "buy" else -1  # +1 = achat d’option, -1 = vente d’option
            hedge = -sign * delta * qty

            if hedge > 0:
                action, nb = "🟢 Acheter", round(hedge)
                color = "green"
            elif hedge < 0:
                action, nb = "🔴 Vendre", round(abs(hedge))
                color = "red"
            else:
                action, nb = "✅ Ne rien faire", 0
                color = "#111"

            self.output.append("\n" + "═" * 70)
            self.output.append("<b>🎯 Couverture Delta (modèle arbre)</b>")
            self.output.append(f"• Δ (central) : {delta:.6f}")
            self.output.append(f"• Qté d’options : {qty:.0f}")
            self.output.append(f"• Position banque : {'Achat' if position == 'buy' else 'Vente'}")

            if nb == 0:
                self.output.append(f"{action}.")
            else:
                self.output.append(
                    f"👉 <font color='{color}'><b>{action.split()[1]}</b> {nb} actions</font> pour être delta-neutre.")
        except Exception as e:
            QMessageBox.critical(self, "Erreur", str(e))


# =========================================================
# 🔹 Lancement de l'application
# =========================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = OptionPricerGUI()
    window.show()
    sys.exit(app.exec())

