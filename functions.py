import numpy as np


class Bound:
    def __init__(self, bounds):
        self.L = bounds[0]
        self.U = bounds[1]


bounds = {
    "Schaffer1": [Bound([-120, 100]), Bound([-120, 100])],
    "Schaffer2": [Bound([-120, 100]), Bound([-120, 100])],
    "Salomon": [Bound([-120, 100]), Bound([-120, 100]), Bound([-120, 100]), Bound([-120, 100]), Bound([-120, 100])],
    "Griewank": [Bound([-550, 500]), Bound([-550, 500]), Bound([-550, 500]), Bound([-550, 500]), Bound([-550, 500]),
                 Bound([-550, 500]), Bound([-550, 500]), Bound([-550, 500]), Bound([-550, 500]), Bound([-550, 500])],
    "PriceTransistor": [Bound([0, 10]), Bound([0, 10]), Bound([0, 10]), Bound([0, 10]), Bound([0, 10]), Bound([0, 10]),
                        Bound([0, 10]), Bound([0, 10]), Bound([0, 10])],
    "Expo": [Bound([-12, 10]), Bound([-12, 10]), Bound([-12, 10]), Bound([-12, 10]), Bound([-12, 10]), Bound([-12, 10]),
             Bound([-12, 10]), Bound([-12, 10]), Bound([-12, 10]), Bound([-12, 10])],
    "Modlangerman": [Bound([0, 10]), Bound([0, 10]), Bound([0, 10]), Bound([0, 10]), Bound([0, 10]), Bound([0, 10]),
                     Bound([0, 10]), Bound([0, 10]), Bound([0, 10]), Bound([0, 10])],
    "EMichalewicz": [Bound([0, np.pi]), Bound([0, np.pi]), Bound([0, np.pi]), Bound([0, np.pi]), Bound([0, np.pi])],
    "Shekelfox5": [Bound([0, 10]), Bound([0, 10]), Bound([0, 10]), Bound([0, 10]), Bound([0, 10])],
    "Schwefel": [Bound([-500, 500]), Bound([-500, 500]), Bound([-500, 500]), Bound([-500, 500]), Bound([-500, 500]),
                 Bound([-500, 500]), Bound([-500, 500]), Bound([-500, 500]), Bound([-500, 500]), Bound([-500, 500])]
}

optimums = {
    "Schaffer1": 0.0,
    "Schaffer2": 0.0,
    "Salomon": 0.0,
    "Griewank": 0.0,
    "PriceTransistor": 0.0,
    "Expo": -1.0,
    "Modlangerman": -0.965,
    "EMichalewicz": -4.6877,
    "Shekelfox5": -10.4056,
    "Schwefel": -4189.829
}


def Schaffer2(par):
    if len(par) != 2:
        print("WARNING: Schaffer2 works on 2d")
    x = par[0]
    y = par[1]
    prod1 = np.power(x * x + y * y, 0.25)
    prod2 = np.power(50 * (x * x + y * y), 0.1)
    return prod1 * (np.sin(np.sin(prod2)) + 1)


def Salomon(par):
    if len(par) != 5:
        print("WARNING:   Parameter vector should be length 5")
    sum = np.sqrt(np.dot(par, par))
    sum = -np.cos(2 * np.pi * sum) + 0.1 * sum + 1
    return sum


def PriceTransistor(par):
    if len(par) != 9:
        print("WARNING:   Parameter vector should be length 9")
    sumsqr = 0.0
    g = np.array([[0.485, 0.752, 0.869, 0.982],
                  [0.369, 1.254, 0.703, 1.455],
                  [5.2095, 10.0677, 22.9274, 20.2153],
                  [23.3037, 101.779, 111.461, 191.267],
                  [28.5132, 111.8467, 134.3884, 211.4823]])
    for k in range(4):
        alpha = (1.0 - par[0] * par[1]) * par[2] * (
                np.exp(par[4] * (g[0][k] - 0.001 * g[2][k] * par[6] - 0.001 * par[7] * g[4][k])) - 1.0) - g[4][k] + \
                g[3][k] * par[1]
        beta = (1.0 - par[0] * par[1]) * par[3] * (
                np.exp(par[5] * (g[0][k] - g[1][k] - 0.001 * g[2][k] * par[6] + g[3][k] * 0.001 * par[8])) - 1.0) - \
               g[4][k] * par[0] + g[3][k]
        sumsqr += alpha * alpha + beta * beta
    sum = par[0] * par[2] - par[1] * par[3]
    sum *= sum
    return sum + sumsqr


def Modlangerman(par):
    if len(par) != 10:
        print("WARNING:   Parameter vector should be length 9")
    a = np.array([[9.681, 0.667, 4.783, 9.095, 3.517, 9.325, 6.544, 0.211, 5.122, 2.020],
                  [9.400, 2.041, 3.788, 7.931, 2.882, 2.672, 3.568, 1.284, 7.033, 7.374],
                  [8.025, 9.152, 5.114, 7.621, 4.564, 4.711, 2.996, 6.126, 0.734, 4.982],
                  [2.196, 0.415, 5.649, 6.979, 9.510, 9.166, 6.304, 6.054, 9.377, 1.426],
                  [8.074, 8.777, 3.467, 1.867, 6.708, 6.349, 4.534, 0.276, 7.633, 1.567]])

    c = np.array([0.806, 0.517, 0.1, 0.908, 0.965])
    sum = 0.0
    for i in range(5):
        dist = 0.0
        for j, x in enumerate(par):
            dx = x - a[i][j]
            dist += dx * dx

        sum -= c[i] * (np.exp(-dist / np.pi) * np.cos(np.pi * dist))

    return sum


def Shekelfox5(par):
    if len(par) != 5:
        print("WARNING:   Parameter vector should be length 5")
    a = np.array([
        [9.681, 0.667, 4.783, 9.095, 3.517, 9.325, 6.544, 0.211, 5.122, 2.020],
        [9.400, 2.041, 3.788, 7.931, 2.882, 2.672, 3.568, 1.284, 7.033, 7.374],
        [8.025, 9.152, 5.114, 7.621, 4.564, 4.711, 2.996, 6.126, 0.734, 4.982],
        [2.196, 0.415, 5.649, 6.979, 9.510, 9.166, 6.304, 6.054, 9.377, 1.426],
        [8.074, 8.777, 3.467, 1.863, 6.708, 6.349, 4.534, 0.276, 7.633, 1.567],
        [7.650, 5.658, 0.720, 2.764, 3.278, 5.283, 7.474, 6.274, 1.409, 8.208],
        [1.256, 3.605, 8.623, 6.905, 4.584, 8.133, 6.071, 6.888, 4.187, 5.448],
        [8.314, 2.261, 4.224, 1.781, 4.124, 0.932, 8.129, 8.658, 1.208, 5.762],
        [0.226, 8.858, 1.420, 0.945, 1.622, 4.698, 6.228, 9.096, 0.972, 7.637],
        [7.305, 2.228, 1.242, 5.928, 9.133, 1.826, 4.060, 5.204, 8.713, 8.247],
        [0.652, 7.027, 0.508, 4.876, 8.807, 4.632, 5.808, 6.937, 3.291, 7.016],
        [2.699, 3.516, 5.874, 4.119, 4.461, 7.496, 8.817, 0.690, 6.593, 9.789],
        [8.327, 3.897, 2.017, 9.570, 9.825, 1.150, 1.395, 3.885, 6.354, 0.109],
        [2.132, 7.006, 7.136, 2.641, 1.882, 5.943, 7.273, 7.691, 2.880, 0.564],
        [4.707, 5.579, 4.080, 0.581, 9.698, 8.542, 8.077, 8.515, 9.231, 4.670],
        [8.304, 7.559, 8.567, 0.322, 7.128, 8.392, 1.472, 8.524, 2.277, 7.826],
        [8.632, 4.409, 4.832, 5.768, 7.050, 6.715, 1.711, 4.323, 4.405, 4.591],
        [4.887, 9.112, 0.170, 8.967, 9.693, 9.867, 7.508, 7.770, 8.382, 6.740],
        [2.440, 6.686, 4.299, 1.007, 7.008, 1.427, 9.398, 8.480, 9.950, 1.675],
        [6.306, 8.583, 6.084, 1.138, 4.350, 3.134, 7.853, 6.061, 7.457, 2.258],
        [0.652, 2.343, 1.370, 0.821, 1.310, 1.063, 0.689, 8.819, 8.833, 9.070],
        [5.558, 1.272, 5.756, 9.857, 2.279, 2.764, 1.284, 1.677, 1.244, 1.234],
        [3.352, 7.549, 9.817, 9.437, 8.687, 4.167, 2.570, 6.540, 0.228, 0.027],
        [8.798, 0.880, 2.370, 0.168, 1.701, 3.680, 1.231, 2.390, 2.499, 0.064],
        [1.460, 8.057, 1.336, 7.217, 7.914, 3.615, 9.981, 9.198, 5.292, 1.224],
        [0.432, 8.645, 8.774, 0.249, 8.081, 7.461, 4.416, 0.652, 4.002, 4.644],
        [0.679, 2.800, 5.523, 3.049, 2.968, 7.225, 6.730, 4.199, 9.614, 9.229],
        [4.263, 1.074, 7.286, 5.599, 8.291, 5.200, 9.214, 8.272, 4.398, 4.506],
        [9.496, 4.830, 3.150, 8.270, 5.079, 1.231, 5.731, 9.494, 1.883, 9.732],
        [4.138, 2.562, 2.532, 9.661, 5.611, 5.500, 6.886, 2.341, 9.699, 6.500]])
    c = np.array([0.806, 0.517, 0.1, 0.908, 0.965, 0.669, 0.524, 0.902, 0.531, 0.876,
                  0.462, 0.491, 0.463, 0.714, 0.352, 0.869, 0.813, 0.811, 0.828, 0.964,
                  0.789, 0.360, 0.369, 0.992, 0.332, 0.817, 0.632, 0.883, 0.608, 0.326])

    sum = 0.0
    for j in range(30):
        sp = 0.0
        for i, x in enumerate(par):
            h = x - a[j][i]
            sp += h * h

        sum -= 1 / (sp + c[j])
    return sum


def Schaffer1(par):
    num = np.power((np.sin(np.power(par[0] * par[0] + par[1] * par[1], 0.5))), 2) - 0.5
    den = np.power((1 + .001 * (par[0] * par[0] + par[1] * par[1])), 2)
    return 0.5 + num / den


def Griewank(par):
    prod = 1.0
    sum = 0.0
    for i, x in enumerate(par):
        sum += x * x
        prod *= np.cos(x / np.power(i + 1, 0.5))
    sum = sum / 4000 - prod + 1
    return sum


def Expo(par):
    sum = 0
    for i in range(10):
        sum += par[i] * par[i]
    sum = -np.exp(-0.5 * sum)
    return sum


def EMichalewicz(par):
    y = np.zeros(10)
    cost = np.cos(np.pi / 6.0)
    sint = np.sin(np.pi / 6.0)
    i = 0
    for i in range(0, len(par) - 1, 2):
        y[i] = par[i] * cost - par[i + 1] * sint
        y[i + 1] = par[i] * sint + par[i + 1] * cost
    if i == len(par) - 1:
        y[i] = par[i]
    sum = 0
    for i in range(len(par)):
        sum -= np.sin(y[i]) * np.power(np.sin((i + 1) * y[i] * y[i] / np.pi), 20)
    return sum


def Schwefel(par):
    sum = 0
    for i in range(10):
        sum -= par[i] * np.sin(np.power(abs(par[i]), 1 / 2))
    return sum


funcs = {
    'Schaffer1': Schaffer1,
    'Schaffer2': Schaffer2,
    'Salomon': Salomon,
    'Griewank': Griewank,
    'PriceTransistor': PriceTransistor,
    'Expo': Expo,
    'Modlangerman': Modlangerman,
    'EMichalewicz': EMichalewicz,
    'Shekelfox5': Shekelfox5,
    'Schwefel': Schwefel
}

names = [
    'Schaffer1',
    'Schaffer2',
    'Salomon',
    'Griewank',
    'PriceTransistor',
    'Expo',
    'Modlangerman',
    'EMichalewicz',
    'Shekelfox5',
    'Schwefel',
]
