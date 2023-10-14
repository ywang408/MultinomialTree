import numpy as np
from typing import List, Dict, Tuple, Union
from src.core.process import Process, Heston, GBM
from src.core.option import EuropeanPut, AmericanPut


class Node:
    def __init__(self, stock_value: float):
        self.stock_value = stock_value
        self.option_value: Union[float, None] = None
        self.child_map: Dict['Node', float] = {}

    def add_child(self, tree_constructor: 'TreeConstructor', step: int, index: int, stock_value: float,
                  probability: float) -> None:
        child_node = tree_constructor.get_or_create(step, index, stock_value)
        self.child_map[child_node] = probability

    def set_option_value(self, value: float) -> None:
        self.option_value = value

    def update_option_value(self, discount_factor: float) -> None:
        option_values = [child.option_value for child in self.child_map.keys()]
        probs = np.array(list(self.child_map.values()))
        option_values = np.array(option_values)
        self.option_value = probs @ option_values * discount_factor

    def __repr__(self) -> str:
        return (f"Node(stock_value={self.stock_value}, option_value={self.option_value}, "
                f"{len(self.child_map)} successors)")


class TreeConstructor:
    def __init__(self, process: Process, num_steps: int, tau: float, p: float) -> None:
        self.node_repository: Dict[int, Dict[int, Node]] = {}
        self.volatility_history: List[float] = [process.y0]
        self.process = process
        self.dt = tau / num_steps
        self.root: Node = self.construct_tree(process, num_steps, p)

    def get_or_create(self, step: int, index: int, stock_value: float) -> Node:
        step_nodes = self.node_repository.setdefault(step, {})
        return step_nodes.setdefault(index, Node(stock_value))

    def classify_case(self, node: Node, vol: float, p: float) -> Tuple[int, np.ndarray, np.ndarray]:
        """Identify cases in one-step tree population"""
        sqrt_dt = np.sqrt(self.dt)
        x = node.stock_value
        j = np.ceil(x / (vol * sqrt_dt))  # j in the paper, used as reference
        values = (np.arange(j + 1, j - 3, -1) * vol * sqrt_dt)

        # Determine probabilities based on the value of q
        probs = self.determine_probabilities(x, j, vol, sqrt_dt, p)

        return j, values, probs

    @staticmethod
    def determine_probabilities(x: float, j: int, vol: float, sqrt_dt: float, p: float) -> np.ndarray:
        """Determine transition probabilities based on x, j, vol, sqrt_dt, and p."""
        if np.abs(j * vol * sqrt_dt - x) < np.abs((j - 1) * vol * sqrt_dt - x):
            delta = x - j * vol * sqrt_dt
            q = delta / (vol * sqrt_dt)
            return np.array([
                1 / 2 * (1 + q + q ** 2) - p,
                3 * p - q ** 2,
                1 / 2 * (1 - q + q ** 2) - 3 * p,
                p
            ])
        else:
            delta = x - (j - 1) * vol * sqrt_dt
            q = delta / (vol * sqrt_dt)
            return np.array([
                p,
                1 / 2 * (1 + q + q ** 2) - 3 * p,
                3 * p - q ** 2,
                1 / 2 * (1 - q + q ** 2) - p
            ])

    def construct_tree(self, process: Process, num_steps: int, p: float) -> Node:
        root = self.get_or_create(0, 0, process.x0)
        for t in range(num_steps):
            curr_vol = self.volatility_history[-1]
            next_vol, sig_vol = process.sample_vol(curr_vol, self.dt)
            self.volatility_history.append(next_vol)
            drift = (process.r - sig_vol ** 2 / 2) * self.dt
            for node in self.node_repository[t].values():
                j, stock_values, probabilities = self.classify_case(node, sig_vol, p)
                stock_values += drift
                for i, (s_value, prob) in enumerate(zip(stock_values, probabilities)):
                    node.add_child(self, t + 1, j - i, s_value, prob)
        return root

    def __repr__(self) -> str:
        return f"tree has {len(self.node_repository)} steps"


def price_option(option_type, process, num_steps, tau, p, n):
    discount_factor = np.exp(-process.r * (tau / num_steps))
    prices = []

    for _ in range(n):
        tree = TreeConstructor(process, num_steps, tau, p)
        option_type.backward_computation(tree.node_repository, discount_factor)
        prices.append(tree.root.option_value)
    return np.array(prices).mean()


def print_tree_value(node_repo):
    """
    Helper function to print stock values along the tree
    """
    n = len(node_repo)
    for t in range(n):
        print()
        for node in node_repo[t].values():
            print(np.exp(node.stock_value), end=" ")


def print_tree_option_value(node_repo):
    """
    Helper function to print information along the tree
    """
    n = len(node_repo)
    for t in range(n):
        print()
        for node in node_repo[t].values():
            stock_values = []
            for child in node.child_map.keys():
                stock_values.append(child.stock_value)
            probs = np.array(list(node.child_map.values()))
            expectation = np.array(stock_values) @ probs
            print(f"step {t}, node {node.stock_value} drift {expectation - node.stock_value}\n"
                  f"stock values {stock_values}, probs {probs}")


if __name__ == '__main__':
    r = 0.05
    num_steps = 100
    T = 1
    dt = T / num_steps
    p = 1 / 12
    discount = np.exp(-r * dt)
    gbm = GBM(np.log(100), 0.2, r, 0.2)
    heston = Heston(np.log(1500), 0.04, 0.05, 3, 0.04, 0.1)
    put = EuropeanPut(strike=1500)
    mean_price = price_option(put, heston, num_steps, T, p, 10)
    print(mean_price)
