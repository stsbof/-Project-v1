from datetime import datetime
from math import exp, sqrt


class Market:
    def __init__(self, underlying: float, rate: float, volatility: float):
        self.underlying = underlying
        self.rate = rate
        self.volatility = volatility


class Models:
    def __init__(self, t0: datetime):
        self.t0 = t0


class TrinomialTree(Models):
    def __init__(self, N_steps: int, maturity: datetime, market: Market, t0: datetime):
        super().__init__(t0)
        self.N_steps = N_steps
        self.maturity = maturity
        self.market = market
        self.dt = (self.maturity - self.t0).days / 365 / self.N_steps
        self.alpha = self.compute_alpha()
        self.root = Node(self.market.underlying, self)

    def compute_alpha(self) -> float:
        return exp(self.market.volatility * sqrt(3 * self.dt))

    def column(self, node: 'Node'):
        tronc = node.forward()
        tronc.Up = Node(tronc.S * self.alpha, self)
        tronc.Down = Node(tronc.S / self.alpha, self)
        node.NextUp = tronc.Up
        node.NextDown = tronc.Down


class Node:
    def __init__(self, price: float, tree: TrinomialTree):
        self.S = price
        self.tree = tree
        self.NextMid = None
        self.NextDown = None
        self.NextUp = None
        self.Up = None
        self.Down = None

    def forward(self):
        price = self.S * exp(self.tree.market.rate * self.tree.dt)
        return Node(price, self.tree)


tree = TrinomialTree(5, datetime(2025, 3, 29), Market(100, 0.03, 0.2), datetime(2025, 1, 29))
tree.column(self.root)

