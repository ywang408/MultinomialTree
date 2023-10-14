from abc import ABC, abstractmethod
import numpy as np


class Option(ABC):
    def __init__(self, strike: float):
        self.strike = strike

    @abstractmethod
    def backward_computation(self, node_repo, discount):
        pass


class EuropeanPut(Option):
    def backward_computation(self, node_repo, discount):
        num_steps = len(node_repo)
        # option value at last step
        for node in node_repo[num_steps - 1].values():
            value = self.strike - np.exp(node.stock_value)
            value = max(0, value)
            node.set_option_value(value)

        # update nodes at previous steps
        for t in range(num_steps - 2, -1, -1):
            for node in node_repo[t].values():
                node.update_option_value(discount)


class AmericanPut(Option):
    def backward_computation(self, node_repo, discount):
        num_steps = len(node_repo)
        # option value at last step
        for node in node_repo[num_steps - 1].values():
            value = self.strike - np.exp(node.stock_value)
            value = max(0, value)
            node.set_option_value(value)

        # update nodes at previous steps
        for t in range(num_steps - 2, -1, -1):
            for node in node_repo[t].values():
                node.update_option_value(discount)
                early_exercise = self.strike - np.exp(node.stock_value)
                if node.option_value < early_exercise:
                    node.set_option_value(early_exercise)
