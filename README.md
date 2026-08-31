# CopperLP

Code for the paper **“Computing Asymptotic Bounds for the Automated Coppersmith Method via Linear Programming.”**

The current repository contains a callable SageMath implementation of **Algorithm 1**. Given input polynomials, a monomial order, modulus-power exponents, and a full-dimensional rational polytope, it computes

- the matrix `Delta`;
- the dual feasible polyhedron `Y`;
- the piecewise-affine function `phi_P`;
- the asymptotic coefficients `sigma_1, ..., sigma_k, sigma_0`;
- the resulting asymptotic small-root bound.

## Requirements

- SageMath 10.8

Create the environment with

```bash
conda env create -f environment.yml
conda activate copper-lp
```

## Running the example

```bash
sage -python example.py
```

The included three-sample MIHNP example should return the symmetric exponent

```text
11/24
```

## Callable use

The public entry point follows the simple style

```python
from algorithm1 import main
```

A complete example is:

```python
import time
from sage.all import PolynomialRing, QQ
from algorithm1 import main

# Define the polynomial ring and polynomials
pr = PolynomialRing(QQ, names=("x1", "x2", "x3"))
x1, x2, x3 = pr.gens()

f12 = x1*x2 + x1 + x2 + 1
f13 = x1*x3 + x1 + x3 + 1
f23 = x2*x3 + x2 + x3 + 1
polys = [f12, f13, f23]

# Define P = {x : A*x <= b}
A = [
    [ 1,  0,  0],
    [ 0,  1,  0],
    [ 0,  0,  1],
    [-1,  0,  0],
    [ 0, -1,  0],
    [ 0,  0, -1],
    [ 1,  1,  1],
]
b = [1, 1, 1, 0, 0, 0, 3]

start = time.time()
result = main(
    pr,
    polys,
    (A, b),
    order="deglex",
    variable_order=(x1, x2, x3),
    e=(1, 1, 1),
)
end = time.time()

print("symmetric exponent =", result.symmetric_exponent())
print("time =", end - start)
```

The third argument can also be a Sage `Polyhedron` instead of the pair `(A,b)`.

## Public API

### `main(...)`

The easiest entry point. It accepts compact string specifications for the monomial order and prints a summary by default.

```python
result = main(
    pr,
    polys,
    polytope,
    order="lex",                 # or "deglex", "weighted_deglex"
    variable_order=(x2, x1, x3), # smallest to largest
    e=(1, 1),
)
```

For weighted degree-lexicographic order, additionally supply

```python
weights=(w1, w2, w3)
```

where the weights follow the canonical generator order `pr.gens()`.

### Structured result

`main(...)` returns an `AsymptoticBoundResult`. Useful fields and methods include

```python
result.Delta
result.dual_vertices
result.affine_pieces
result.cells
result.sigma_variables
result.sigma_0

result.phi_string()
result.phi_latex()
result.bound_string()
result.bound_latex()
result.symmetric_exponent()
result.one_parameter_exponent(scales)
result.summary()
```

Set `verbose=False` when the caller only needs the returned object:

```python
result = main(..., verbose=False)
```

## Files

- `algorithm1.py`: callable implementation of Algorithm 1;
- `example.py`: minimal three-sample MIHNP example;
- `environment.yml`: reproducible SageMath environment;
- `.gitignore`: ignored local files.

## Current scope

The current version treats a **fixed** rational polytope. Parameterized-polytope optimization and the remaining paper examples will be added separately.
