import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt


def f(x):
    return 5 * np.sin(x) - x + 1


def f1(x):
    return 5 * np.cos(x) - 1


def dichotomy_method(a, b, eps):
    if f(a) * f(b) >= 0:
        return
    n = 1
    x = (a + b) / 2
    while abs(f(x)) > eps:
        x = (a + b) / 2
        n += 1
        if f(a) * f(x) < 0:
            b = x
        else:
            a = x
    return x, n


def chord_method(a, b, eps):
    if f(a) * f(b) >= 0:
        return
    n = 1
    x0 = a
    xn = b - (f(b) / (f(b) - f(x0))) * (b - x0)
    while abs(xn - x0) > eps:
        x0 = xn
        xn = b - (f(b) / (f(b) - f(x0))) * (b - x0)
        n += 1
    return xn, n


def chord_tangent_method(a, b, eps):
    if f(a) * f(b) >= 0:
        return
    n = 1
    a0 = a
    b0 = b
    xn1 = b0 - (f(b0) / (f(b0) - f(a0))) * (b0 - a0)
    xn2 = b0 - f(b0) / f1(a0)
    while abs(xn1 - xn2) > 2 * eps:
        a0 = xn1
        b0 = xn2
        xn1 = b0 - (f(b0) / (f(b0) - f(a0))) * (b0 - a0)
        xn2 = b0 - f(b0) / f1(a0)
        n += 1
    return xn1, n


A = 2.0
B = 4.0
e = 10e-4
X = np.arange(A, B, 0.01)

res1, count1 = dichotomy_method(A, B, e)
res2, count2 = chord_method(A, B, e)
res3, count3 = chord_tangent_method(A, B, e)
rest = opt.fsolve(f, A)[0]
print("Half-division method: x = %0.4f" % res1)
print("Number of iterations = %d\n" % count1)
print("Chord method: x = %0.4f" % res2)
print("Number of iterations = %d\n" % count2)
print("Chord and tangent method: x = %0.4f" % res3)
print("Number of iterations = %d\n" % count3)
print("Check: x = %0.4f" % rest)

plt.grid(True)
plt.plot(X, f(X), lw=2, color="green")
plt.show()
