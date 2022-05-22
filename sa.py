#!/usr/bin/python

from functions import funcs, bounds, names, optimums
import numpy as np
from scipy import optimize
import sympy as sp
from gmpy2 import mpfr, exp


def expo(t, step, curr_t):
    return t * pow(0.95, step)


def fsa(t, step, curr_t):
    return t / (step + 1)


def boltz(t, step, curr_t):
    return t / np.log(step + 2)


def geom(t, step, curr_t):
    return 0.95 * curr_t


schedules = {
    "exp": expo,
    "fsa": fsa,
    "boltz": boltz,
    "geom": geom
}

scheduleNames = {
    "exp",
    "fsa",
    "boltz",
    "geom"
}


class Parameters:
    def __init__(self, temp, schedule, step_size_coeff):
        self.temp = temp
        self.schedule = schedule
        self.step_size_coeff = step_size_coeff

controls = {
    'Schaffer1': Parameters(100, 'fsa', 20),
    'Schaffer2': Parameters(100, 'boltz', 20),
    'Salomon': Parameters(10, 'exp', 20),
    'Griewank': Parameters(10, 'exp', 5),
    'PriceTransistor': Parameters(1000, 'geom', 3),
    'Expo': Parameters(100, 'boltz', 20),
    'Modlangerman': Parameters(1000, 'geom', 3),
    'EMichalewicz': Parameters(10, 'boltz', 3),
    'Shekelfox5': Parameters(10, 'geom', 3),
    'Schwefel': Parameters(10, 'boltz', 3)
}


def insideBounds(b, candidate):
    return all(bd.L <= d <= bd.U for bd, d in zip(b, candidate))

def sa(fn, b, params, name):
    # Generate initial solution
    best_val = np.inf
    for i in range(100):
        candidate = [np.random.uniform(bd.L, bd.U) for bd in b]
        if fn(candidate) < best_val:
            best = candidate
            best_val = fn(candidate)

    # Dimensions
    dims = len(b)

    # Evaluate the initial solution
    best_val = fn(best)

    # Initialize temperature and annealing schedule
    initial_t = mpfr(params.temp)
    t = mpfr(params.temp)

    schedule = schedules[params.schedule]

    # Initialize current result
    curr, curr_val = best, best_val
    n_iterations = 10000

    # Initialize step size
    step_size = np.abs(b[0].L - b[0].U) / params.step_size_coeff

    for i in range(n_iterations):
        # Generate candidate
        candidate = [np.random.uniform(-1, 1) * step_size for bd in b]
        while not insideBounds(b, candidate):
            candidate = [np.random.uniform(-1, 1) * step_size for bd in b]

        # Evaluate candidate
        candidate_val = fn(candidate)

        # Check if better candidate and store best
        if candidate_val < best_val:
            best, best_val = candidate, candidate_val

        # Difference between candidate and current point objective function value
        diff = candidate_val - curr_val

        # Adjust temperature
        t = mpfr(schedule(initial_t, i, t))

        # Adjust step size
        step_size = step_size * 0.99

        # Calculate acceptance probability
        prob = exp(-diff / t)

        # Move to new point (or not)
        if diff < 0 or np.random.rand() < prob:
            curr, curr_val = candidate, candidate_val

    # After getting best result, perform local optimisation
    # best = optimize.fmin_cg(fn, best)
    return best, best_val


def main():
    for fn in names:
        best, best_val = sa(funcs[fn], bounds[fn], controls[fn], fn)
        print(fn, "opt: ", optimums.get(fn), " result: ", best_val, " diff: ",
              np.abs(np.abs(optimums.get(fn)) - np.abs(best_val)))


if __name__ == "__main__":
    main()
