import random as rnd
from typing import List
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

## Seed
rnd.seed(42)


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
    for i in range(100):
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
        self.machine_history: List[int] = [] 
    
    def play(self, machine: Bandit, consideration: float = 1.0):
        self.money -= consideration
        win_amount = machine.play(consideration)
        self.money += win_amount
        self.history.append(self.money)
        return win_amount
    
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
            self.machine_history.append(machine_num)
            self.play(self.casino[machine_num])

class OptimalStrategy(Player):
    def simulate(self, rounds):
        best_index = len(self.casino) - 1
        for i in range(rounds):
            self.machine_history.append(best_index)
            self.play(self.casino[best_index])

class UCB1(Player):
    def __init__(self, casino, money = 1000):
        super().__init__(casino, money)
        self.total_rewards = {k: 0 for k in range(len(self.casino))}
        self.total_machine_plays = {k: 0 for k in range(len(self.casino))}
        self.total_plays = 0

    def play(self, machine_index, consideration = 1):
        reward = super().play(self.casino[machine_index], consideration)
        self.machine_history.append(machine_index)
        self.total_rewards[machine_index] += reward
        self.total_machine_plays[machine_index] += 1
        self.total_plays += 1

    def machine_value(self, machine_index):
        x_avg = self.total_rewards[machine_index] / self.total_machine_plays[machine_index]
        return x_avg + np.sqrt(2*np.log(self.total_plays)/self.total_machine_plays[machine_index])
    
    def max_machine_value(self):
        max = -np.inf
        max_mach = None
        for i in range(len(self.casino)):
            val = self.machine_value(i)
            if val>max:
                max = val
                max_mach = i
        return max_mach

    def simulate(self, rounds):
        init_rounds = len(self.casino)
        assert rounds >= init_rounds, "Must complete more rounds than number of machines"
        leftover = rounds - init_rounds

        for i in range(init_rounds):
            self.play(i)
        for i in range(leftover):
            max_machine = self.max_machine_value()
            self.play(machine_index=max_machine)

    


############################################################
#################     PLOTTING       #######################
############################################################


def simulate(strategies: List[Player], rounds: int, colour_optimal: bool = False):
    """
    Simulates the given number of rounds, and plots the results
    from all simulations in the same plot.
    If colour_optimal is True, colours the line segments green on
    rounds where the optimal bandit was played.
    """
    # run each strategy's simulation
    for strategy in strategies:
        strategy.simulate(rounds)

    # combined plot
    fig, ax = plt.subplots(figsize=(8, 4))

    for strategy in strategies:
        x = np.arange(len(strategy.history))
        label = type(strategy).__name__
        history = np.asarray(strategy.history)

        # base line in its own colour
        (line,) = ax.plot(x, history, label=label)

        if colour_optimal and strategy.machine_history:
            # find optimal machine for this strategy's casino
            optimal_index = max(
                range(len(strategy.casino)),
                key=lambda i: strategy.casino[i].prob
            )

            machines = np.asarray(strategy.machine_history)
            # indices t where optimal machine was played (rounds are 1..N)
            opt_rounds = np.nonzero(machines == optimal_index)[0] + 1

            if opt_rounds.size > 0:
                # build segments [((t-1, y[t-1]), (t, y[t]))] for all optimal rounds
                segs = np.stack([
                    np.column_stack((opt_rounds - 1, history[opt_rounds - 1])),
                    np.column_stack((opt_rounds,     history[opt_rounds]))
                ], axis=1)
                # segs.shape = (num_segments, 2, 2)

                lc = LineCollection(
                    segs,
                    colors="green",
                    linewidths=line.get_linewidth()
                )
                ax.add_collection(lc)

    ax.set_xlabel("Round")
    ax.set_ylabel("Money")
    ax.set_title(f"Player balance over time ({rounds} rounds)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left")

    plt.tight_layout()
    plt.show()





if __name__ == "__main__":
    casino = init_casino()
    for i in casino:
        print(i.prob)
    print(f"Average: {np.mean([i.prob for i in casino])}")




    #r = RandomStrategy(casino)
    o = OptimalStrategy(casino)
    u = UCB1(casino)
    simulate(strategies=[o, u], rounds=1_000_000, colour_optimal=True)