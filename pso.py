from functions import funcs, bounds, names, optimums
from random import uniform, seed, random
from datetime import datetime
import warnings

# Disable warnings for overflows since those results won't ever be used as the best anyway.
warnings.filterwarnings("ignore")

# Seed for reproducibility.
seed(0)

# Algorithm settings for separate funcitons.
PARAMS = {
    'Schaffer1': [40, 0.99, 0.04, 0.01, 20, 1000, 0.02],
    'Schaffer2': [40, 0.99, 0.04, 0.01, 20, 1000, 0.001],
    'Salomon': [20, 0.98, 0.08, 0.08, 20, 1000, 0.02],
    'Griewank': [20, 0.98, 0.08, 0.08, 20, 1000, 0.02],
    'PriceTransistor': [20, 0.98, 0.08, 0.08, 20, 1000, 0.02],
    'Expo': [20, 0.998, 0.08, 0.08, 20, 1000, 0.02],
    'Modlangerman': [40, 0.98, 0.05, 0.1, 20, 1000, 0.5],
    'EMichalewicz': [20, 0.98, 0.08, 0.08, 20, 1000, 0.02],
    'Shekelfox5': [20, 0.98, 0.08, 0.08, 20, 1000, 0.02],
    'Schwefel': [20, 0.98, 0.08, 0.08, 20, 1000, 0.02],
}


# Individual class to hold all the data about a single instance.
class Individual:

    # Initializes the individual with a random position and velocity.
    def __init__(self, dimensions, b, max_speed, alpha, beta, gamma):
        # Save settings.
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.bounds = b
        self.dimensions = dimensions

        # Generate a new random position and velocity.
        self.position = [uniform(b[i].L, b[i].U) for i in range(dimensions)]
        self.velocity = []
        for i in range(dimensions):
            # max_speed = (b[i].U - b[i].L)
            self.velocity.append(uniform(-max_speed, max_speed))
        self.best_position = self.position

    # Moves the individual by the velocity, calculates the next velocity and saves the best position.
    def update(self, global_best, f):
        # Update position.
        new_position = []
        for i in range(self.dimensions):
            x = self.position[i] + self.velocity[i]

            # Bounce against the bounds.
            if x < self.bounds[i].L or x > self.bounds[i].U:
                self.velocity[i] *= -1

            new_position.append(x)
        self.position = new_position

        # new_position = [self.position[i] + self.velocity[i] for i in range(self.dimensions)]

        # Calculate the new velocity.
        v = []
        for j in range(self.dimensions):
            r1 = random()
            r2 = random()
            vj = self.alpha * self.velocity[j] + self.beta * r1 * (self.best_position[j] - self.position[j]) + \
                 self.gamma * r2 * (global_best[j] - self.position[j])
            v.append(vj)
        self.velocity = v
        self.position = new_position

        # Save the best position.
        if self.valid() and f(self.position) < f(self.best_position):
            self.best_position = self.position

    # Check if the individual is within the bounds.
    def valid(self):
        for i in range(self.dimensions):
            if self.position[i] < self.bounds[i].L or self.position[i] > self.bounds[i].U:
                return False
        return True


# Particle Swarm Optimization algorithm.
def pso(name):
    # Start time for stopping the algorithm.
    start_time = datetime.now()

    # Get the algorithm parameters.
    swarm_size, alpha, beta, gamma, max_time, max_iters, max_speed = PARAMS[name]

    # Get the function to optimize.
    f = funcs[name]
    dim = len(bounds[name])
    population = []
    iters = 0

    # Initialize individuals
    for i in range(swarm_size):
        population.append(Individual(dim, bounds[name], max_speed, alpha, beta, gamma))

    best = None
    global_best = None

    while iters < max_iters and (datetime.now() - start_time).seconds < max_time:

        # Update best individual and the global best.
        for i in range(len(population)):
            ind = population[i]
            if best is None or f(ind.position) < f(best) and ind.valid():
                best = ind.position

            if global_best is None or f(ind.position) < f(global_best) and ind.valid():
                global_best = ind.position

        # Update the velocity and position of each individual.
        for i in range(len(population)):
            ind = population[i]
            ind.update(global_best, f)

        iters += 1

    return best


def main():
    # Test all the functions.
    for name in names:
        best = pso(name)
        f = funcs[name]

        print(name, ':', f(best), ':', abs(f(best) - optimums[name]))


if __name__ == '__main__':
    main()
