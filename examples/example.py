#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal callable example for Algorithm 1.

Run from the repository root with

    sage -python example.py
"""

import time

from sage.all import PolynomialRing, QQ

from algorithm1 import main


# ---------------------------------------------------------------------------
# Define the polynomial ring and input polynomials
# ---------------------------------------------------------------------------

pr = PolynomialRing(QQ, names=("x1", "x2", "x3"))
x1, x2, x3 = pr.gens()

f12 = x1 * x2 + x1 + x2 + 1
f13 = x1 * x3 + x1 + x3 + 1
f23 = x2 * x3 + x2 + x3 + 1
polys = [f12, f13, f23]


# ---------------------------------------------------------------------------
# Define P = {x : A*x <= b}
# ---------------------------------------------------------------------------

A = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [-1, 0, 0],
    [0, -1, 0],
    [0, 0, -1],
    [1, 1, 1],
]
b = [1, 1, 1, 0, 0, 0, 3]


# ---------------------------------------------------------------------------
# Call Algorithm 1
# ---------------------------------------------------------------------------

start = time.time()
result = main(
    pr,
    polys,
    (A, b),
    order="deglex",
    # The order is written from the smallest variable to the largest,
    # following the notation of the paper.
    variable_order=(x1, x2, x3),
    # f12, f13, f23 are all equations modulo M.
    e=(1, 1, 1),
)
end = time.time()

print("\nAdditional callable outputs")
print("---------------------------")
print("phi_P =", result.phi_string())
print("sigma =", result.sigma_variables)
print("sigma_0 =", result.sigma_0)
print("symmetric exponent =", result.symmetric_exponent())
print("time =", end - start, "seconds")
