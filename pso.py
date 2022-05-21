from functions import funcs, bounds, names, optimums
from random import uniform

swarm_size = 20
max_speed = 1
alpha = 0.5
beta = 0.5
gamma = 0.4
delta = 1


def distance(a, b):
    return sum([(a[i] - b[i]) ** 2 for i in range(len(a))]) ** 0.5


class Individual:
    def __init__(self, dimensions, b):
        self.dimensions = dimensions
        self.position = [uniform(b[i].L, b[i].U) for i in range(dimensions)]
        self.velocity = [uniform(-max_speed, max_speed) for _ in range(dimensions)]
        self.best_position = self.position

    def update(self, global_best, f):
        # Update position.
        # print("Prev position:", self.position)
        self.position = [self.position[i] + self.velocity[i] for i in range(self.dimensions)]
        # print("New position:", self.position)
        # input()

        # Calculate the new velocity.
        v = []
        for j in range(self.dimensions):
            b = uniform(0, beta)
            c = uniform(0, gamma)
            d = uniform(0, delta)
            vj = alpha * self.velocity[j] + b * distance(self.best_position, self.position) + \
                 c * distance(global_best, self.position) + \
                 d * distance(self.position, self.best_position)
            v.append(vj)
        self.velocity = v

        # Save the best position.
        if f(self.position) < f(self.best_position):
            self.best_position = self.position


def pso(name):
    f = funcs[name]
    dim = len(bounds[name])
    population = []

    # Initialize individuals
    for i in range(swarm_size):
        population.append(Individual(dim, bounds[name]))

    best = None
    global_best = None

    while True:  # TODO: condition
        # Update best individual and the global best.
        for i in range(swarm_size):
            ind = population[i]
            if best is None or f(ind.position) > f(best):
                best = ind.position

            if global_best is None or f(ind.position) > f(global_best):
                global_best = ind.position

        for i in range(swarm_size):
            # print(ind.position)
            ind = population[i]
            ind.update(global_best, f)
            # print(ind.position)

        print(f(best), optimums[name])
        input("next iter?")

    return best

pso(names[0])
