import random as rnd
from typing import List
import matplotlib.pyplot as plt
import numpy as np

############################################################
#################     MACHINES       #######################
############################################################

class Bandit:
    def __init__(self, prob: float):
        assert 0 < prob < 1
        self.prob = prob

    def play(self, consideration: float = 1.0):
        if rnd.random() < self.prob:
            return consideration*2
        else:
            return 0
        

def init_casino() -> List[Bandit]:
    casino: List[Bandit] = []
    for i in range(10):
        casino.append(Bandit((rnd.random()+0.3+0.4)/3))
    casino.sort(key=lambda x: x.prob)
    return casino
    
############################################################
#################     PLAYERS        #######################
############################################################


class Player:
    def __init__(self, casino: List[Bandit], money: float = 1000.0, ):
        self.money = money
        self.casino = casino
        self.history = [money]
    
    def play(self, machine: Bandit, consideration: float = 1.0):
        self.money -= consideration
        win_amount = machine.play(consideration)
        self.money += win_amount
        self.history.append(self.money)
    
    def simulate(self, rounds: int):
        pass

    def plot(self):
        fig, ax = plt.subplots(figsize=(8, 4))

        # x-axis: round index (0, 1, 2, ...)
        rounds = range(len(self.history))
        ax.plot(rounds, self.history)

        ax.set_xlabel("Round")
        ax.set_ylabel("Money")
        ax.set_title("Player balance over time")
        ax.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.show()


class RandomStrategy(Player):
    def simulate(self, rounds: int):
        for i in range(rounds):
            machine_num: int = rnd.randint(0, len(self.casino)-1)
            self.play(self.casino[machine_num])

class OptimalStrategy(Player):
    def simulate(self, rounds):
        for i in range(rounds):
            self.play(self.casino[len(self.casino)-1])


############################################################
#################     PLOTTING       #######################
############################################################


def simulate(strategies: List[Player], rounds: int):
    """
    Simulates the given number of rounds, and plots the results
    from all simulations in the same plot
    """
    # run each strategy's simulation
    for strategy in strategies:
        strategy.simulate(rounds)

    # combined plot
    fig, ax = plt.subplots(figsize=(8, 4))

    for strategy in strategies:
        x = range(len(strategy.history))
        label = type(strategy).__name__
        ax.plot(x, strategy.history, label=label)

    ax.set_xlabel("Round")
    ax.set_ylabel("Money")
    ax.set_title(f"Player balance over time ({rounds} rounds)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    casino = init_casino()
    for i in casino:
        print(i.prob)
    print(f"Average: {np.mean([i.prob for i in casino])}")




    r = RandomStrategy(casino)
    o = OptimalStrategy(casino)

    simulate(strategies=[r, o], rounds=10_000)