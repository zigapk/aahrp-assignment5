#!/usr/bin/python

from functions import funcs, bounds, names, optimums
import numpy as np
from scipy import optimize
import itertools
import sympy as sp
from gmpy2 import mpfr, exp
import matplotlib.pyplot as plt


def expo(t, step):
    return t * mpfr(pow(0.95, step))


def fsa(t, step):
    return mpfr(t / 0.5 * mpfr(step + 1))


def boltz(t, step):
    return mpfr(t / mpfr(np.log(step + 2)))


schedules = {
    "exp": expo,
    "fsa": fsa,
    "boltz": boltz
}

scheduleNames = {
    "exp",
    "fsa",
    "boltz"
}

temps = {
    10,
    100,
    1000
}


class Parameters:
    def __init__(self, temp, schedule, step_size):
        self.temp = temp
        self.schedule = schedule
        self.step_size = step_size

controls = {
    'Schaffer1': Parameters(100, 'fsa', 2),  # BEST: -1.871351376442343e-05
    'Schaffer2': Parameters(100, 'boltz', 2),  # BEST: -0.11495008887399101
    'Salomon': Parameters(10, 'exp', 2),  # BEST: -0.09987334850340979
    'Griewank': Parameters(10, 'exp', 10),  # BEST: -0.11374005622802086
    'PriceTransistor': Parameters(100, 'boltz', 2),  # BEST: -11.882357455049325
    'Expo': Parameters(100, 'boltz', 2),  # BEST: 1.1798516886252834
    'Modlangerman': Parameters(1000, 'boltz', 2),  # BEST: 0.9925941343417226
    'EMichalewicz': Parameters(10, 'boltz', 2),  # BEST: 1.2213803104203915
    'Shekelfox5': Parameters(10, 'fsa', 2),  # BEST: 7.253687654404963
    'Schwefel': Parameters(1000, 'fsa', 2)  # BEST: 1480.6897282414966
}


def insideBounds(b, candidate):
    return all(bd.L <= d <= bd.U for bd, d in zip(b, candidate))


def gradient_descent(gradient, x0):
    n_iter = 200 * len(x0)
    x = x0
    for i in range(n_iter):
        diff = -0.1 * gradient(x)
        x += diff
    return x


def genSA(fn, b, params):
    # Generate initial solution
    best = [np.random.uniform(bd.L, bd.U) for bd in b]

    # Dimensions
    dims = len(b)

    # Evaluate the initial solution
    best_eval = fn(best)

    # Initialize temperature and annealing schedule
    t = mpfr(params.temp)
    schedule = schedules[params.schedule]

    # Initialize current result
    curr, curr_eval = best, best_eval
    n_iterations = 10000

    # Initialize step size
    step_size = params.step_size

    for i in range(n_iterations):
        # take a step
        candidate = curr + np.random.randn(dims) * step_size
        if not insideBounds(b, candidate):
            continue
        # evaluate candidate point
        candidate_eval = fn(candidate)
        # check for new best solution
        if candidate_eval < best_eval:
            # store new best point
            best, best_eval = candidate, candidate_eval
        # difference between candidate and current point evaluation
        diff = candidate_eval - curr_eval
        # calculate temperature for current epoch
        t = mpfr(schedule(t, i))
        # calculate metropolis acceptance criterion
        metropolis = exp(-diff / t)
        # check if we should keep the new point
        if diff < 0 or np.random.rand() < metropolis:
            # store the new current point
            curr, curr_eval = candidate, candidate_eval

    # After getting best result, perform local optimisation
    #best = optimize.fmin_cg(fn, best)
    #best = gradient_descent(np.gradient(fn), best)
    return fn(best)


def main():
    for fn in names:
        best = genSA(funcs[fn], bounds[fn], controls[fn])
        print(fn, ": opt: ", optimums.get(fn), " result: ", best, " diff: ", np.abs(np.abs(optimums.get(fn)) - np.abs(best)))


if __name__ == "__main__":
    main()
