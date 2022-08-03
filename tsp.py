import itertools
import random
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class Instance:
    def __init__(self, n):
        self.coordinates = [(random.random(), random.random()) for i in range(n)]

    def distance(self, v, u):
        return np.linalg.norm([self.coordinates[v][i] - self.coordinates[u][i] for i in range(2)])

    def get_n(self):
        return len(self.coordinates)

    def get_coordinates(self, v):
        return self.coordinates[v]


class Solution:
    def __init__(self, instance, sequence):
        self.instance = instance
        self.sequence = sequence

    def calculate_objective(self):
        prev = self.sequence[-1]
        result = 0
        for v in self.sequence:
            result += self.instance.distance(prev, v)
            prev = v

        return result

    # Removes edges (sequence[i], sequence[i+1]) and (sequence[j], sequence[j+1])
    # and inserts edges (sequence[i], sequence[j]) and (sequence[i+1], sequence[j+1])
    def two_opt(self, i, j):
        if i > j:
            i, j = j, i

        # if j == i + 1 or j == 0 and i == self.instance.get_n():
        #    return

        self.sequence[i+1:j+1] = reversed(self.sequence[i+1:j+1])

    def swap(self, i, j):
        self.sequence[i], self.sequence[j] = self.sequence[j], self.sequence[i]


def random_construction(instance):
    sequence = list(range(instance.get_n()))
    random.shuffle(sequence)
    return Solution(instance, sequence)


def nearest_neighbour_construction(instance):
    prev = random.randint(0, instance.get_n() - 1)
    sequence = [prev]
    unvisited = list(range(instance.get_n()))
    unvisited.remove(prev)

    while len(unvisited) > 0:
        i = np.argmin([instance.distance(prev, u) for u in unvisited])
        prev = unvisited.pop(i)
        sequence.append(prev)

    return Solution(instance, sequence)


def local_search(solution, move):
    objective = solution.calculate_objective()
    for i, j in itertools.combinations(range(solution.instance.get_n()), 2):
        move(solution, i, j)
        new_objective = solution.calculate_objective()
        if new_objective < objective:
            return True

        # This reverses the move
        move(solution, i, j)

    return False


def plot_solution(solution, ax):
    ax.plot(*zip(*[solution.instance.get_coordinates(v) for v in solution.sequence + solution.sequence[0:1]]), marker='o')


def test_construction(instance, construction):
    solution = construction(instance)
    plot_solution(solution, plt)
    plt.title(f'Objective value: {solution.calculate_objective():.2f}')
    plt.show()


def test_local_search(instance, construction, move):
    solution = construction(instance)
    while local_search(solution, move):
        print(f'{solution.calculate_objective()}')

    plot_solution(solution, plt.axes())
    plt.title(f'Objective value: {solution.calculate_objective():.2f}')
    plt.gcf().canvas.manager.set_window_title(f'{construction.__name__} {move.__name__}')
    plt.show()


def animate_local_search(instance, construction, move):
    solution = construction(instance)

    fig, axs = plt.subplots(1, 2)
    fig.canvas.manager.set_window_title(f'{construction.__name__} {move.__name__}')

    obj = [solution.calculate_objective()]

    def animation_frame(i):
        if animation_frame.running and local_search(solution, move):
            obj.append(solution.calculate_objective())
        else:
            animation_frame.running = False

        axs[0].cla()
        axs[1].cla()
        axs[1].set_xlim([0, (len(obj) // 50 + 1) * 50])
        axs[1].set_ylim([0, obj[0]])
        plot_solution(solution, axs[0])
        axs[1].plot(range(len(obj)), obj)

        status = f'iteration {len(obj)}' if animation_frame.running else f'stopped after {len(obj)} iterations'
        axs[0].set_title(f'Objective value: {solution.calculate_objective():.2f}; {status}')

    animation_frame.running = True
    animation = FuncAnimation(fig, animation_frame, blit=False, interval=100)
    plt.show()


def animate_simulated_annealing(instance, construction, move):
    solution = construction(instance)

    fig, axs = plt.subplots(1, 2)
    fig.canvas.manager.set_window_title(f'{construction.__name__} {move.__name__}')

    obj = [solution.calculate_objective()]

    def animation_frame(_):
        global iteration

        for _ in range(100):
            i, j = random.sample(range(instance.get_n()), k=2)
            cur_obj = solution.calculate_objective()
            move(solution, i, j)
            new_obj = solution.calculate_objective()

            if new_obj < cur_obj:
                accept = True
            else:
                accept = random.random() < np.exp(-(new_obj - cur_obj) / animation_frame.temperature)

            if not accept:
                move(solution, i, j)

            obj.append(cur_obj)
            animation_frame.temperature *= 0.9999
            if new_obj < animation_frame.best_found:
                animation_frame.best_found = new_obj

        axs[0].cla()
        axs[1].cla()
        axs[1].set_xlim([0, len(obj)])
        axs[1].set_ylim([0, np.max(obj) * 1.1])
        plot_solution(solution, axs[0])
        axs[1].plot(range(len(obj)), obj)

        axs[0].set_title(f'Objective value: {solution.calculate_objective():.2f}; temperature {animation_frame.temperature:.5f}')
        axs[1].set_title(f'Best found: {animation_frame.best_found:.2f}')

    animation_frame.temperature = 0.1
    animation_frame.best_found = sys.float_info.max

    animation = FuncAnimation(fig, animation_frame, blit=False, interval=100)
    plt.show()


def main():
    random.seed(123)
    instance = Instance(30) # 30 cities

    # try different ways for constructing a solution to the traveling salesman problem
    #test_construction(instance, random_construction)
    test_construction(instance, nearest_neighbour_construction)
    test_construction(instance, local_search)

    # test_local_search(instance, random_construction, Solution.two_opt)

    #animate_local_search(instance, random_construction, Solution.two_opt)
    #animate_simulated_annealing(instance, random_construction, Solution.two_opt)


main()
