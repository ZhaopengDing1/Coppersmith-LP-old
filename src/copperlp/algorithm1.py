#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""algorithm1.py

Exact SageMath implementation of Algorithm 1 in

    "Computing Asymptotic Bounds for the Automated Coppersmith Method
     via Linear Programming".

The module treats a *fixed*, full-dimensional rational polytope P and computes

    phi_P(x) = max <e, tau>
               s.t. Delta tau <= b - A x, tau >= 0,

through the vertices of the dual polyhedron

    Y = {z >= 0 : Delta^T z >= e}.

It then constructs the polyhedral cells on which phi_P is affine and computes
exactly

    sigma_j = integral_P x_j dx,
    sigma_0 = integral_P phi_P(x) dx,

so that the asymptotic Coppersmith condition is

    product_j X_j^(sigma_j) < M^(sigma_0 - epsilon).

Use it as a callable module with ``from algorithm1 import main``, or run
the built-in regression example with SageMath:

    sage -python algorithm1.py

The code uses exact rational arithmetic throughout.  It does not enumerate the
monomial sets mP cap Z^k and it does not use interpolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from sage.all import Polyhedron, PolynomialRing, QQ, ZZ, latex, matrix, vector
except ImportError as exc:  # pragma: no cover - gives a useful message outside Sage
    raise ImportError(
        "algorithm1.py must be imported or executed inside SageMath. "
        "Use: sage -python algorithm1.py"
    ) from exc


Exponent = Tuple[int, ...]
RationalTuple = Tuple[Any, ...]


__all__ = (
    "AffineFunction",
    "AsymptoticBoundResult",
    "CellData",
    "MonomialOrder",
    "PolynomialData",
    "compute_asymptotic_bound",
    "demo_mihnp_three_samples",
    "main",
    "polytope_from_leq",
)


# ---------------------------------------------------------------------------
# Exact conversion and formatting helpers
# ---------------------------------------------------------------------------


def _as_qq(value: Any, label: str = "value") -> Any:
    """Convert ``value`` to QQ and raise a precise error if this is impossible."""

    try:
        return QQ(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label}={value!r} is not an exact rational number") from exc


def _as_positive_integer(value: Any, label: str) -> Any:
    q = _as_qq(value, label)
    if q.denominator() != 1 or q <= 0:
        raise ValueError(f"{label} must be a positive integer, got {value!r}")
    return ZZ(q)


def _normalize_variable_names(variables: Sequence[Any]) -> Tuple[str, ...]:
    names = tuple(str(v) for v in variables)
    if not names:
        raise ValueError("At least one polynomial variable is required")
    if len(set(names)) != len(names):
        raise ValueError(f"Variable names must be pairwise distinct, got {names}")
    return names


def _monomial_text(exponent: Exponent, variables: Sequence[str]) -> str:
    factors: List[str] = []
    for name, power in zip(variables, exponent):
        if power == 0:
            continue
        factors.append(name if power == 1 else f"{name}^{power}")
    return "1" if not factors else "*".join(factors)


def _polytope_volume(polytope: Any, engine: str) -> Any:
    return _as_qq(
        polytope.volume(measure="ambient", engine=engine),
        "polytope volume",
    )


def _polytope_centroid(polytope: Any, engine: str) -> RationalTuple:
    # Sage's centroid() passes keyword arguments to its triangulation engine.
    return tuple(
        _as_qq(c, "centroid coordinate")
        for c in polytope.centroid(engine=engine)
    )


# ---------------------------------------------------------------------------
# Monomial orders in the notation of Definition 1 of the paper
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MonomialOrder:
    """A monomial order used only to determine the leading exponents.

    ``variable_order`` is written from the *smallest* variable to the largest,
    exactly as in the paper.  For example,

        MonomialOrder.lex(("x2", "x1", "x3"))

    represents ``x2 < x1 < x3``.

    Supported kinds are ``lex``, ``deglex``, and ``weighted_deglex``.  In the
    weighted case, ``weights`` are aligned with the canonical variable order
    supplied to :func:`compute_asymptotic_bound`, not with
    ``variable_order``.
    """

    kind: str
    variable_order: Tuple[str, ...]
    weights: Optional[Tuple[Any, ...]] = None

    def __post_init__(self) -> None:
        aliases = {
            "lex": "lex",
            "lexicographic": "lex",
            "deglex": "deglex",
            "degree_lex": "deglex",
            "degree-lex": "deglex",
            "weighted_deglex": "weighted_deglex",
            "weighted-deglex": "weighted_deglex",
            "wdeglex": "weighted_deglex",
        }
        raw_kind = str(self.kind).lower().strip()
        if raw_kind not in aliases:
            raise ValueError(
                "Unsupported monomial order. Choose lex, deglex, or "
                "weighted_deglex"
            )
        object.__setattr__(self, "kind", aliases[raw_kind])
        object.__setattr__(
            self,
            "variable_order",
            _normalize_variable_names(self.variable_order),
        )

        if self.kind == "weighted_deglex":
            if self.weights is None:
                raise ValueError("weighted_deglex requires a positive weight vector")
            rational_weights = tuple(
                _as_qq(w, f"weight[{i}]") for i, w in enumerate(self.weights)
            )
            if any(w <= 0 for w in rational_weights):
                raise ValueError("All weighted-deglex weights must be positive")
            object.__setattr__(self, "weights", rational_weights)
        elif self.weights is not None:
            raise ValueError(f"Weights are not used by the {self.kind} order")

    @classmethod
    def lex(cls, variable_order: Sequence[Any]) -> "MonomialOrder":
        return cls("lex", tuple(str(v) for v in variable_order))

    @classmethod
    def deglex(cls, variable_order: Sequence[Any]) -> "MonomialOrder":
        return cls("deglex", tuple(str(v) for v in variable_order))

    @classmethod
    def weighted_deglex(
        cls,
        variable_order: Sequence[Any],
        weights: Sequence[Any],
    ) -> "MonomialOrder":
        return cls(
            "weighted_deglex",
            tuple(str(v) for v in variable_order),
            tuple(weights),
        )

    def validate_for(self, canonical_variables: Sequence[str]) -> None:
        canonical = tuple(canonical_variables)
        if set(self.variable_order) != set(canonical):
            raise ValueError(
                "variable_order must be a permutation of the polynomial "
                f"variables {canonical}; got {self.variable_order}"
            )
        if len(self.variable_order) != len(canonical):
            raise ValueError("variable_order has the wrong length")
        if self.kind == "weighted_deglex" and len(self.weights or ()) != len(
            canonical
        ):
            raise ValueError(
                "The weighted-deglex weight vector must have one entry for "
                "each canonical polynomial variable"
            )

    def key(
        self,
        exponent: Sequence[Any],
        canonical_variables: Sequence[str],
    ) -> Tuple[Any, ...]:
        """Return an exact Python/Sage comparison key for an exponent vector."""

        canonical = tuple(canonical_variables)
        self.validate_for(canonical)
        exponent_tuple = tuple(ZZ(a) for a in exponent)
        if len(exponent_tuple) != len(canonical):
            raise ValueError("Exponent vector has the wrong dimension")

        index = {name: i for i, name in enumerate(canonical)}
        # The paper compares at the largest position in
        # x_{pi(1)} < ... < x_{pi(k)}, hence the reversed variable order.
        lex_key = tuple(
            exponent_tuple[index[name]] for name in reversed(self.variable_order)
        )

        if self.kind == "lex":
            return lex_key
        if self.kind == "deglex":
            return (sum(exponent_tuple),) + lex_key

        assert self.weights is not None
        weighted_degree = sum(
            self.weights[i] * exponent_tuple[i]
            for i in range(len(exponent_tuple))
        )
        return (weighted_degree,) + lex_key

    def leading_exponent(
        self,
        support: Iterable[Exponent],
        canonical_variables: Sequence[str],
    ) -> Exponent:
        support_tuple = tuple(support)
        if not support_tuple:
            raise ValueError("The zero polynomial has no leading monomial")
        return max(support_tuple, key=lambda u: self.key(u, canonical_variables))

    def description(self) -> str:
        chain = " < ".join(self.variable_order)
        if self.kind == "weighted_deglex":
            return f"weighted deglex ({chain}), weights={self.weights}"
        return f"{self.kind} ({chain})"


# ---------------------------------------------------------------------------
# Affine pieces, polynomial metadata, cells, and result object
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AffineFunction:
    """An exact affine function ``constant + linear dot x`` over QQ."""

    constant: Any
    linear: RationalTuple

    def __post_init__(self) -> None:
        object.__setattr__(self, "constant", _as_qq(self.constant, "constant"))
        object.__setattr__(
            self,
            "linear",
            tuple(_as_qq(c, "linear coefficient") for c in self.linear),
        )

    def __call__(self, point: Sequence[Any]) -> Any:
        if len(point) != len(self.linear):
            raise ValueError("Point has the wrong dimension")
        x = tuple(_as_qq(c, "point coordinate") for c in point)
        return self.constant + sum(c * xi for c, xi in zip(self.linear, x))

    def __sub__(self, other: "AffineFunction") -> "AffineFunction":
        if len(self.linear) != len(other.linear):
            raise ValueError("Affine functions have different dimensions")
        return AffineFunction(
            self.constant - other.constant,
            tuple(a - b for a, b in zip(self.linear, other.linear)),
        )

    def signature(self) -> Tuple[Any, ...]:
        return (self.constant,) + self.linear

    def polynomial(self, variables: Sequence[str]) -> Any:
        if len(variables) != len(self.linear):
            raise ValueError("Variable list has the wrong dimension")
        ring = PolynomialRing(QQ, names=tuple(variables))
        gens = ring.gens()
        return ring(self.constant) + sum(
            ring(self.linear[i]) * gens[i] for i in range(len(gens))
        )

    def text(self, variables: Sequence[str]) -> str:
        return str(self.polynomial(variables))

    def latex(self, variables: Sequence[str]) -> str:
        return str(latex(self.polynomial(variables)))


@dataclass(frozen=True)
class PolynomialData:
    index: int
    polynomial: Any
    support: Tuple[Exponent, ...]
    leading_exponent: Exponent
    leading_coefficient: Any
    modulus_power: Any

    def leading_monomial_text(self, variables: Sequence[str]) -> str:
        return _monomial_text(self.leading_exponent, variables)


@dataclass(frozen=True)
class CellData:
    piece_index: int
    affine_piece: AffineFunction
    polyhedron: Any
    dimension: int
    volume: Any
    centroid: Optional[RationalTuple]
    integral: Any

    @property
    def full_dimensional(self) -> bool:
        return self.centroid is not None


@dataclass
class AsymptoticBoundResult:
    """Structured output of Algorithm 1."""

    variables: Tuple[str, ...]
    monomial_order: MonomialOrder
    polytope: Any
    polynomial_data: Tuple[PolynomialData, ...]
    A: Any
    b: Any
    Delta: Any
    e: Any
    dual_polyhedron: Any
    dual_vertices: Tuple[RationalTuple, ...]
    affine_pieces: Tuple[AffineFunction, ...]
    cells: Tuple[CellData, ...]
    polytope_volume: Any
    polytope_centroid: RationalTuple
    sigma_variables: RationalTuple
    sigma_0: Any

    @property
    def dimension(self) -> int:
        return len(self.variables)

    @property
    def full_dimensional_cells(self) -> Tuple[CellData, ...]:
        return tuple(cell for cell in self.cells if cell.full_dimensional)

    def phi(self, point: Sequence[Any], check_membership: bool = True) -> Any:
        """Evaluate ``phi_P`` from the dual affine minimum."""

        x = tuple(_as_qq(c, "point coordinate") for c in point)
        if len(x) != self.dimension:
            raise ValueError("Point has the wrong dimension")
        if check_membership and not self.polytope.contains(vector(QQ, x)):
            raise ValueError(f"The point {x} does not belong to P")
        return min(piece(x) for piece in self.affine_pieces)

    def primal_phi(self, point: Sequence[Any]) -> Any:
        """Evaluate ``phi_P`` by directly solving the primal LP at ``point``.

        This method is mainly a verification tool.  The main algorithm uses the
        dual vertices once, rather than solving a new LP for every point.
        """

        x = vector(QQ, tuple(_as_qq(c, "point coordinate") for c in point))
        if len(x) != self.dimension:
            raise ValueError("Point has the wrong dimension")
        if not self.polytope.contains(x):
            raise ValueError(f"The point {tuple(x)} does not belong to P")

        n = len(self.polynomial_data)
        inequalities: List[List[Any]] = []

        # tau_j >= 0
        for j in range(n):
            row = [QQ(0)] + [QQ(0)] * n
            row[j + 1] = QQ(1)
            inequalities.append(row)

        # b_t - <a_t,x> - sum_j Delta[t,j] tau_j >= 0
        for t in range(self.A.nrows()):
            rhs = self.b[t] - self.A.row(t).dot_product(x)
            inequalities.append(
                [rhs] + [-self.Delta[t, j] for j in range(n)]
            )

        feasible = Polyhedron(ieqs=inequalities, base_ring=QQ)
        vertices = tuple(tuple(v) for v in feasible.vertices_list())
        if not vertices:
            raise RuntimeError("The primal feasible region unexpectedly has no vertex")
        return max(
            sum(self.e[j] * v[j] for j in range(n)) for v in vertices
        )

    def verify_phi(self, points: Iterable[Sequence[Any]]) -> bool:
        """Check dual and primal evaluations at the supplied rational points."""

        for point in points:
            if self.phi(point) != self.primal_phi(point):
                return False
        return True

    def symmetric_exponent(self) -> Any:
        """Return delta for ``X_1 = ... = X_k = M^delta``."""

        denominator = sum(self.sigma_variables)
        if denominator <= 0:
            raise ZeroDivisionError("The sum of variable coefficients is not positive")
        return self.sigma_0 / denominator

    def one_parameter_exponent(self, scales: Sequence[Any]) -> Any:
        """Return maximal delta for ``X_j = M^(scales[j] * delta)``."""

        if len(scales) != self.dimension:
            raise ValueError("scales has the wrong length")
        q_scales = tuple(_as_qq(s, "scale") for s in scales)
        denominator = sum(
            self.sigma_variables[j] * q_scales[j]
            for j in range(self.dimension)
        )
        if denominator <= 0:
            raise ValueError("sum_j sigma_j * scales_j must be positive")
        return self.sigma_0 / denominator

    def is_admissible_exponent_vector(
        self,
        exponents: Sequence[Any],
        strict: bool = True,
    ) -> bool:
        """Test ``sum_j sigma_j delta_j <(=) sigma_0``."""

        if len(exponents) != self.dimension:
            raise ValueError("Exponent vector has the wrong length")
        delta = tuple(_as_qq(d, "root exponent") for d in exponents)
        lhs = sum(
            self.sigma_variables[j] * delta[j]
            for j in range(self.dimension)
        )
        return lhs < self.sigma_0 if strict else lhs <= self.sigma_0

    def phi_string(self) -> str:
        body = ", ".join(piece.text(self.variables) for piece in self.affine_pieces)
        return f"phi_P({', '.join(self.variables)}) = min{{{body}}}"

    def phi_latex(self) -> str:
        body = r",\; ".join(
            piece.latex(self.variables) for piece in self.affine_pieces
        )
        args = ",".join(self.variables)
        return rf"\varphi_P({args})=\min\left\{{{body}\right\}}"

    def exponent_region_string(self, strict: bool = True) -> str:
        relation = "<" if strict else "<="
        lhs = " + ".join(
            f"({self.sigma_variables[j]})*delta{j + 1}"
            for j in range(self.dimension)
        )
        return f"{lhs} {relation} {self.sigma_0}"

    def exponent_region_latex(self, strict: bool = True) -> str:
        relation = "<" if strict else r"\le"
        lhs = "+".join(
            rf"{latex(self.sigma_variables[j])}\delta_{{{j + 1}}}"
            for j in range(self.dimension)
        )
        return rf"{lhs}{relation}{latex(self.sigma_0)}"

    def bound_string(self, include_epsilon: bool = True) -> str:
        left = " * ".join(
            f"X{j + 1}^({self.sigma_variables[j]})"
            for j in range(self.dimension)
        )
        exponent = f"{self.sigma_0} - epsilon" if include_epsilon else str(self.sigma_0)
        return f"{left} < M^({exponent})"

    def bound_latex(self, include_epsilon: bool = True) -> str:
        left = " ".join(
            rf"X_{{{j + 1}}}^{{{latex(self.sigma_variables[j])}}}"
            for j in range(self.dimension)
        )
        exponent = str(latex(self.sigma_0))
        if include_epsilon:
            exponent += r"-\varepsilon"
        return rf"{left}<M^{{{exponent}}}"

    def asymptotic_function_strings(self) -> Tuple[str, ...]:
        """Leading asymptotics of p_j(m), p_0(m), |M_m|, and t_m."""

        k = self.dimension
        lines = [
            f"p_{j + 1}(m) = ({self.sigma_variables[j]})*m^{k + 1} + O(m^{k})"
            for j in range(k)
        ]
        lines.append(f"p_0(m) = ({self.sigma_0})*m^{k + 1} + O(m^{k})")
        lines.append(
            f"|M_m| = ({self.polytope_volume})*m^{k} + O(m^{k - 1})"
        )
        lines.append("t_m = O(m)")
        return tuple(lines)

    def summary(self) -> str:
        lines: List[str] = []
        lines.append("Algorithm 1: exact asymptotic-bound computation")
        lines.append("=" * 57)
        lines.append(f"Variables: {self.variables}")
        lines.append(f"Monomial order: {self.monomial_order.description()}")
        lines.append(
            f"Polytope: dim={self.polytope.dim()}, facets={self.A.nrows()}, "
            f"volume={self.polytope_volume}, centroid={self.polytope_centroid}"
        )
        lines.append("")
        lines.append("Input polynomial data:")
        for data in self.polynomial_data:
            lines.append(
                f"  f_{data.index}: LM={data.leading_monomial_text(self.variables)}, "
                f"alpha={data.leading_exponent}, e={data.modulus_power}, "
                f"|A(f)|={len(data.support)}"
            )
        lines.append("")
        lines.append("Delta matrix (rows = inequalities of P, columns = polynomials):")
        lines.append(str(self.Delta))
        lines.append("")
        lines.append(
            f"Dual polyhedron Y: {len(self.dual_vertices)} vertices, "
            f"{len(self.dual_polyhedron.rays_list())} rays"
        )
        lines.append(self.phi_string())
        lines.append(
            f"Full-dimensional cells: {len(self.full_dimensional_cells)}"
        )
        lines.append("")
        for j, sigma in enumerate(self.sigma_variables, start=1):
            lines.append(f"sigma_{j} = integral_P x_{j} dx = {sigma}")
        lines.append(f"sigma_0 = integral_P phi_P(x) dx = {self.sigma_0}")
        lines.append("")
        lines.append("Asymptotic bound:")
        lines.append(f"  {self.bound_string()}")
        lines.append(f"  Root-exponent region: {self.exponent_region_string()}")
        lines.append(
            "Symmetric specialization X_1=...=X_k=M^delta: "
            f"delta < {self.symmetric_exponent()}"
        )
        lines.append("")
        lines.append("Leading asymptotic functions:")
        lines.extend(f"  {line}" for line in self.asymptotic_function_strings())
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Polytope construction and input validation
# ---------------------------------------------------------------------------


def polytope_from_leq(
    A: Sequence[Sequence[Any]],
    b: Sequence[Any],
    *,
    add_nonnegative_constraints: bool = False,
) -> Any:
    """Construct ``P = {x : A*x <= b}`` as an exact Sage rational polytope.

    Sage stores an inequality as ``c + d*x >= 0``.  Therefore a paper-style
    row ``a*x <= beta`` is passed to Sage as ``[beta, -a]``.

    Set ``add_nonnegative_constraints=True`` to append ``x_i >= 0`` for every
    coordinate.
    """

    rows = [tuple(row) for row in A]
    rhs = tuple(b)
    if len(rows) != len(rhs):
        raise ValueError("A and b have incompatible numbers of rows")
    if not rows:
        raise ValueError("At least one inequality is required")
    k = len(rows[0])
    if k == 0:
        raise ValueError("The ambient dimension must be positive")
    if any(len(row) != k for row in rows):
        raise ValueError("All rows of A must have the same length")

    inequalities: List[List[Any]] = []
    for t, row in enumerate(rows):
        beta = _as_qq(rhs[t], f"b[{t}]")
        a = [_as_qq(value, f"A[{t},{i}]") for i, value in enumerate(row)]
        inequalities.append([beta] + [-value for value in a])

    if add_nonnegative_constraints:
        for i in range(k):
            row = [QQ(0)] + [QQ(0)] * k
            row[i + 1] = QQ(1)  # x_i >= 0
            inequalities.append(row)

    return Polyhedron(ieqs=inequalities, base_ring=QQ)


def _coerce_rational_polytope(polytope: Any) -> Any:
    """Copy a Sage polyhedron to an exact QQ backend."""

    required = (
        "inequalities_list",
        "equations_list",
        "ambient_dim",
        "dim",
        "is_empty",
        "is_compact",
    )
    if any(not hasattr(polytope, name) for name in required):
        raise TypeError(
            "polytope must be a Sage Polyhedron; use polytope_from_leq(A,b) "
            "for an A*x <= b description"
        )

    try:
        inequalities = [
            [_as_qq(c, "polytope inequality coefficient") for c in row]
            for row in polytope.inequalities_list()
        ]
        equations = [
            [_as_qq(c, "polytope equation coefficient") for c in row]
            for row in polytope.equations_list()
        ]
    except Exception as exc:
        raise TypeError("P must be a rational polyhedron") from exc

    return Polyhedron(
        ieqs=inequalities,
        eqns=equations,
        base_ring=QQ,
        ambient_dim=polytope.ambient_dim(),
    )


def _validate_polytope(
    polytope: Any,
    dimension: int,
    require_nonnegative: bool,
) -> None:
    if polytope.ambient_dim() != dimension:
        raise ValueError(
            f"P lies in ambient dimension {polytope.ambient_dim()}, but the "
            f"polynomial system has {dimension} variables"
        )
    if polytope.is_empty():
        raise ValueError("P must be nonempty")
    if not polytope.is_compact():
        raise ValueError("P must be bounded")
    if polytope.dim() != dimension:
        raise ValueError(
            f"P must be full-dimensional in QQ^{dimension}; got dim(P)="
            f"{polytope.dim()}"
        )
    if require_nonnegative:
        negative_vertices = [
            tuple(v)
            for v in polytope.vertices_list()
            if any(coordinate < 0 for coordinate in v)
        ]
        if negative_vertices:
            raise ValueError(
                "The theorem assumes P is contained in the nonnegative "
                f"orthant; negative vertices include {negative_vertices[:3]}"
            )


def _extract_polynomial_data(
    polynomials: Sequence[Any],
    canonical_variables: Tuple[str, ...],
    monomial_order: MonomialOrder,
    modulus_powers: Tuple[Any, ...],
    require_monic: bool,
    require_constant_term: bool,
) -> Tuple[PolynomialData, ...]:
    data: List[PolynomialData] = []
    k = len(canonical_variables)

    for j, polynomial in enumerate(polynomials):
        if not hasattr(polynomial, "parent") or not hasattr(polynomial, "dict"):
            raise TypeError(f"f_{j + 1} is not a Sage polynomial")
        if polynomial == 0:
            raise ValueError(f"f_{j + 1} is the zero polynomial")

        parent = polynomial.parent()
        if not hasattr(parent, "gens"):
            raise TypeError(f"The parent of f_{j + 1} is not a polynomial ring")
        parent_variables = tuple(str(g) for g in parent.gens())
        if len(parent_variables) != k or set(parent_variables) != set(
            canonical_variables
        ):
            raise ValueError(
                f"f_{j + 1} uses variables {parent_variables}, expected a "
                f"polynomial ring on {canonical_variables}"
            )
        canonical_positions = tuple(
            parent_variables.index(name) for name in canonical_variables
        )

        coefficient_by_exponent: Dict[Exponent, Any] = {}
        for raw_exponent, coefficient in polynomial.dict().items():
            if coefficient == 0:
                continue
            # Multivariate Sage backends use ETuple keys; univariate rings use
            # integer keys.  ETuple is iterable but is not assumed to be a
            # Python tuple here.
            try:
                parent_exponent = tuple(ZZ(a) for a in raw_exponent)
            except TypeError:
                parent_exponent = (ZZ(raw_exponent),)
            if len(parent_exponent) != k:
                raise ValueError(
                    f"Unexpected exponent dimension in f_{j + 1}: "
                    f"{parent_exponent}"
                )
            exponent = tuple(parent_exponent[i] for i in canonical_positions)
            coefficient_by_exponent[exponent] = coefficient

        support = tuple(sorted(coefficient_by_exponent))
        if not support:
            raise ValueError(f"f_{j + 1} is the zero polynomial")
        zero = tuple(ZZ(0) for _ in range(k))
        if require_constant_term and zero not in coefficient_by_exponent:
            raise ValueError(
                f"f_{j + 1} does not satisfy 0 in A(f_{j + 1}); "
                "a nonzero constant term is required by the theorem"
            )
        if all(all(a == 0 for a in exponent) for exponent in support):
            raise ValueError(f"f_{j + 1} must be nonconstant")

        alpha = monomial_order.leading_exponent(support, canonical_variables)
        leading_coefficient = coefficient_by_exponent[alpha]
        if require_monic and leading_coefficient != 1:
            raise ValueError(
                f"f_{j + 1} is not monic for the supplied order: "
                f"LC={leading_coefficient}, LM="
                f"{_monomial_text(alpha, canonical_variables)}. "
                "Normalize it first or set require_monic=False if only the "
                "support computation is intended."
            )

        data.append(
            PolynomialData(
                index=j + 1,
                polynomial=polynomial,
                support=support,
                leading_exponent=alpha,
                leading_coefficient=leading_coefficient,
                modulus_power=modulus_powers[j],
            )
        )

    return tuple(data)


# ---------------------------------------------------------------------------
# Algorithm 1
# ---------------------------------------------------------------------------


def compute_asymptotic_bound(
    polynomials: Sequence[Any],
    monomial_order: MonomialOrder,
    polytope: Any,
    *,
    modulus_powers: Optional[Sequence[Any]] = None,
    variables: Optional[Sequence[Any]] = None,
    require_monic: bool = True,
    require_constant_term: bool = True,
    require_nonnegative_polytope: bool = True,
    integration_engine: str = "internal",
    verify_cell_decomposition: bool = True,
) -> AsymptoticBoundResult:
    """Compute Algorithm 1 exactly for a fixed rational polytope.

    Parameters
    ----------
    polynomials:
        Sage multivariate polynomials ``f_1,...,f_n``.  Coefficients affect
        only the monicity check; Algorithm 1 itself uses their supports and
        leading monomials.
    monomial_order:
        An instance of :class:`MonomialOrder`.
    polytope:
        A full-dimensional bounded rational Sage ``Polyhedron`` in the
        nonnegative orthant.  The helper :func:`polytope_from_leq` accepts the
        paper's ``A*x <= b`` convention.
    modulus_powers:
        Positive integers ``e_j`` for equations modulo ``M^(e_j)``.  Defaults
        to one for every polynomial.
    variables:
        Canonical coordinate order ``(x_1,...,x_k)``.  Defaults to the order of
        generators in the parent ring of the first polynomial.
    require_monic, require_constant_term, require_nonnegative_polytope:
        Enforce the hypotheses used in the paper.
    integration_engine:
        Sage triangulation/volume engine.  ``"internal"`` requires no optional
        external package and keeps rational inputs exact.
    verify_cell_decomposition:
        Check exactly that the full-dimensional cells have total volume
        ``vol(P)``.

    Returns
    -------
    :class:`AsymptoticBoundResult`
        Contains ``phi_P``, all affine pieces and cells, ``Delta``, dual
        vertices, ``sigma_1,...,sigma_k,sigma_0``, and convenience methods for
        specializations of the root bounds.
    """

    polynomial_tuple = tuple(polynomials)
    if not polynomial_tuple:
        raise ValueError("At least one polynomial is required")
    if not isinstance(monomial_order, MonomialOrder):
        raise TypeError("monomial_order must be a MonomialOrder instance")

    if variables is None:
        first_parent = polynomial_tuple[0].parent()
        canonical_variables = _normalize_variable_names(first_parent.gens())
    else:
        canonical_variables = _normalize_variable_names(variables)
    k = len(canonical_variables)
    monomial_order.validate_for(canonical_variables)

    if modulus_powers is None:
        e_tuple = tuple(ZZ(1) for _ in polynomial_tuple)
    else:
        if len(modulus_powers) != len(polynomial_tuple):
            raise ValueError(
                "modulus_powers must contain one exponent for every polynomial"
            )
        e_tuple = tuple(
            _as_positive_integer(value, f"e_{j + 1}")
            for j, value in enumerate(modulus_powers)
        )

    P = _coerce_rational_polytope(polytope)
    _validate_polytope(P, k, require_nonnegative_polytope)

    polynomial_data = _extract_polynomial_data(
        polynomial_tuple,
        canonical_variables,
        monomial_order,
        e_tuple,
        require_monic,
        require_constant_term,
    )

    # Convert Sage's H-representation b_sage + c*x >= 0 to the paper's
    # <a_t,x> <= b_t by setting a_t=-c and b_t=b_sage.
    paper_normals: List[RationalTuple] = []
    paper_rhs: List[Any] = []
    for inequality in P.inequality_generator():
        c = tuple(_as_qq(v, "facet coefficient") for v in inequality.A())
        b_t = _as_qq(inequality.b(), "facet right-hand side")
        paper_normals.append(tuple(-v for v in c))
        paper_rhs.append(b_t)

    if not paper_normals:
        raise RuntimeError("A positive-dimensional bounded polytope has no facets")

    A = matrix(QQ, paper_normals)
    b = vector(QQ, paper_rhs)
    T = A.nrows()
    n = len(polynomial_data)

    # Delta_t(f_j) = max_{u in A(f_j)} <a_t, u-alpha_j>.
    delta_entries: List[List[Any]] = []
    for t in range(T):
        row: List[Any] = []
        a_t = A.row(t)
        for data in polynomial_data:
            alpha = vector(QQ, data.leading_exponent)
            values = [
                a_t.dot_product(vector(QQ, exponent) - alpha)
                for exponent in data.support
            ]
            delta = max(values)
            if delta < 0:
                raise RuntimeError(
                    "Internal error: Delta_t(f_j) must be nonnegative because "
                    "the leading exponent belongs to A(f_j)"
                )
            row.append(delta)
        delta_entries.append(row)
    Delta = matrix(QQ, delta_entries)
    e = vector(QQ, e_tuple)

    # Boundedness of P and nonconstancy of each f_j imply that each Delta
    # column contains a positive entry.  Detect malformed inputs explicitly.
    for j in range(n):
        if all(Delta[t, j] == 0 for t in range(T)):
            raise ValueError(
                f"All Delta_t(f_{j + 1}) vanish.  Check that P is bounded, "
                "f_j is nonconstant, and the leading monomial/order are correct."
            )

    # Y = {z in R^T : z>=0, Delta^T z >= e}.
    dual_inequalities: List[List[Any]] = []
    for t in range(T):
        row = [QQ(0)] + [QQ(0)] * T
        row[t + 1] = QQ(1)
        dual_inequalities.append(row)
    for j in range(n):
        dual_inequalities.append(
            [-e[j]] + [Delta[t, j] for t in range(T)]
        )

    Y = Polyhedron(ieqs=dual_inequalities, base_ring=QQ)
    if Y.is_empty():
        raise RuntimeError("The dual feasible polyhedron Y is empty")
    dual_vertices = tuple(
        sorted(
            (
                tuple(_as_qq(c, "dual vertex coordinate") for c in vertex)
                for vertex in Y.vertices_list()
            ),
            key=lambda vertex: tuple(vertex),
        )
    )
    if not dual_vertices:
        raise RuntimeError("The dual feasible polyhedron Y has no vertices")

    # L_i(x) = (b-Ax)^T z^(i) = b^T z^(i) - (A^T z^(i))^T x.
    unique_pieces: Dict[Tuple[Any, ...], AffineFunction] = {}
    for vertex in dual_vertices:
        z = vector(QQ, vertex)
        piece = AffineFunction(
            b.dot_product(z),
            tuple(-(A.transpose() * z)[i] for i in range(k)),
        )
        unique_pieces[piece.signature()] = piece
    affine_pieces = tuple(
        unique_pieces[key] for key in sorted(unique_pieces, key=lambda item: tuple(item))
    )

    # Q_i = P intersect {phi_i <= phi_h for every h}.
    cells: List[CellData] = []
    sigma_0 = QQ(0)
    total_full_dimensional_volume = QQ(0)
    base_inequalities = [list(row) for row in P.inequalities_list()]
    base_equations = [list(row) for row in P.equations_list()]

    for i, piece_i in enumerate(affine_pieces):
        cell_inequalities = list(base_inequalities)
        for h, piece_h in enumerate(affine_pieces):
            if h == i:
                continue
            # piece_i <= piece_h  <=>  piece_h-piece_i >= 0.
            difference = piece_h - piece_i
            cell_inequalities.append(
                [difference.constant] + list(difference.linear)
            )

        cell = Polyhedron(
            ieqs=cell_inequalities,
            eqns=base_equations,
            base_ring=QQ,
            ambient_dim=k,
        )
        if cell.is_empty():
            cells.append(
                CellData(i, piece_i, cell, -1, QQ(0), None, QQ(0))
            )
            continue

        cell_dimension = int(cell.dim())
        if cell_dimension != k:
            cells.append(
                CellData(
                    i,
                    piece_i,
                    cell,
                    cell_dimension,
                    QQ(0),
                    None,
                    QQ(0),
                )
            )
            continue

        cell_volume = _polytope_volume(cell, integration_engine)
        cell_centroid = _polytope_centroid(cell, integration_engine)
        cell_integral = cell_volume * piece_i(cell_centroid)
        total_full_dimensional_volume += cell_volume
        sigma_0 += cell_integral
        cells.append(
            CellData(
                i,
                piece_i,
                cell,
                cell_dimension,
                cell_volume,
                cell_centroid,
                cell_integral,
            )
        )

    P_volume = _polytope_volume(P, integration_engine)
    P_centroid = _polytope_centroid(P, integration_engine)
    if verify_cell_decomposition and total_full_dimensional_volume != P_volume:
        raise RuntimeError(
            "The exact cell-volume check failed: "
            f"sum_i vol(Q_i)={total_full_dimensional_volume}, vol(P)={P_volume}. "
            "This usually indicates a polyhedral backend or input issue."
        )

    # For an affine coordinate function, integral_P x_j dx = vol(P)*centroid_j.
    sigma_variables = tuple(P_volume * P_centroid[j] for j in range(k))

    return AsymptoticBoundResult(
        variables=canonical_variables,
        monomial_order=monomial_order,
        polytope=P,
        polynomial_data=polynomial_data,
        A=A,
        b=b,
        Delta=Delta,
        e=e,
        dual_polyhedron=Y,
        dual_vertices=dual_vertices,
        affine_pieces=affine_pieces,
        cells=tuple(cells),
        polytope_volume=P_volume,
        polytope_centroid=P_centroid,
        sigma_variables=sigma_variables,
        sigma_0=sigma_0,
    )


# ---------------------------------------------------------------------------
# Simple public entry point, mirroring ``from algorithm1 import main``
# ---------------------------------------------------------------------------


def _resolve_public_order(
    order: Any,
    variable_order: Optional[Sequence[Any]],
    weights: Optional[Sequence[Any]],
    canonical_variables: Sequence[Any],
) -> MonomialOrder:
    """Convert the compact public order specification to ``MonomialOrder``."""

    if isinstance(order, MonomialOrder):
        if variable_order is not None or weights is not None:
            raise ValueError(
                "When order is already a MonomialOrder, do not also pass "
                "variable_order or weights"
            )
        return order

    if order is None:
        order = "deglex"
    if variable_order is None:
        variable_order = canonical_variables

    kind = str(order).lower().strip()
    if kind in {"lex", "lexicographic"}:
        if weights is not None:
            raise ValueError("weights are only used with weighted_deglex")
        return MonomialOrder.lex(variable_order)
    if kind in {"deglex", "degree_lex", "degree-lex"}:
        if weights is not None:
            raise ValueError("weights are only used with weighted_deglex")
        return MonomialOrder.deglex(variable_order)
    if kind in {"weighted_deglex", "weighted-deglex", "wdeglex"}:
        if weights is None:
            raise ValueError("weighted_deglex requires weights=(w_1,...,w_k)")
        return MonomialOrder.weighted_deglex(variable_order, weights)

    raise ValueError(
        "Unsupported order. Choose 'lex', 'deglex', or 'weighted_deglex'"
    )


def main(
    pr: Any,
    polys: Sequence[Any],
    polytope: Any,
    *,
    order: Any = "deglex",
    variable_order: Optional[Sequence[Any]] = None,
    weights: Optional[Sequence[Any]] = None,
    e: Optional[Sequence[Any]] = None,
    add_nonnegative_constraints: bool = False,
    verbose: bool = True,
    require_monic: bool = True,
    require_constant_term: bool = True,
    require_nonnegative_polytope: bool = True,
    integration_engine: str = "internal",
    verify_cell_decomposition: bool = True,
) -> AsymptoticBoundResult:
    """Run Algorithm 1 through a compact, repository-style callable API.

    This is the recommended entry point for examples and external scripts::

        from algorithm1 import main

        result = main(
            pr,
            polys,
            (A, b),
            order="deglex",
            variable_order=(x1, x2, x3),
            e=(1, 1, 1),
        )

    Parameters
    ----------
    pr:
        Sage polynomial ring containing ``polys``.
    polys:
        Input polynomials ``f_1,...,f_n``.
    polytope:
        Either a Sage ``Polyhedron`` or a pair ``(A,b)`` representing
        ``P={x : A*x <= b}`` in the notation of the paper.
    order:
        ``"lex"``, ``"deglex"``, ``"weighted_deglex"``, or an already
        constructed :class:`MonomialOrder`.
    variable_order:
        Variables listed from smallest to largest, exactly as in the paper.
        It defaults to ``pr.gens()``.
    weights:
        Positive weights in the canonical generator order ``pr.gens()`` when
        ``order="weighted_deglex"``.
    e:
        Modulus-power exponents ``(e_1,...,e_n)``.  It defaults to all ones.
    verbose:
        Print a readable summary when ``True``.  The structured result is
        returned in either case.

    Returns
    -------
    AsymptoticBoundResult
        The LP function, all asymptotic coefficients, the final bound, and
        intermediate polyhedral data.
    """

    if not hasattr(pr, "gens"):
        raise TypeError("pr must be a Sage polynomial ring")
    canonical_variables = tuple(pr.gens())
    if not canonical_variables:
        raise ValueError("pr must contain at least one variable")

    polynomial_tuple = tuple(polys)
    if not polynomial_tuple:
        raise ValueError("polys must contain at least one polynomial")
    for j, polynomial in enumerate(polynomial_tuple, start=1):
        if not hasattr(polynomial, "parent"):
            raise TypeError(f"polys[{j - 1}] is not a Sage polynomial")
        if polynomial.parent() is not pr and tuple(
            str(g) for g in polynomial.parent().gens()
        ) != tuple(str(g) for g in canonical_variables):
            raise ValueError(
                f"polys[{j - 1}] is not defined over the supplied ring pr"
            )

    monomial_order = _resolve_public_order(
        order,
        variable_order,
        weights,
        canonical_variables,
    )

    # Allow users to pass the paper's H-representation directly as ``(A,b)``.
    if isinstance(polytope, (tuple, list)) and len(polytope) == 2 and not hasattr(
        polytope, "inequalities_list"
    ):
        P = polytope_from_leq(
            polytope[0],
            polytope[1],
            add_nonnegative_constraints=add_nonnegative_constraints,
        )
    else:
        if add_nonnegative_constraints:
            raise ValueError(
                "add_nonnegative_constraints is only available when polytope "
                "is supplied as the pair (A,b)"
            )
        P = polytope

    result = compute_asymptotic_bound(
        polynomial_tuple,
        monomial_order,
        P,
        modulus_powers=e,
        variables=canonical_variables,
        require_monic=require_monic,
        require_constant_term=require_constant_term,
        require_nonnegative_polytope=require_nonnegative_polytope,
        integration_engine=integration_engine,
        verify_cell_decomposition=verify_cell_decomposition,
    )

    if verbose:
        print(result.summary())
    return result


# ---------------------------------------------------------------------------
# Regression example: MIHNP with three samples (Section 4.2 of the paper)
# ---------------------------------------------------------------------------


def demo_mihnp_three_samples() -> AsymptoticBoundResult:
    """Reproduce the paper's three-sample MIHNP value ``delta = 11/24``.

    At the near-optimal parameter theta=3, the polytope

        0 <= x_i <= 1,  x_1+x_2+x_3 <= 3

    is the unit cube.  Generic nonzero coefficients do not affect Algorithm 1,
    so support representatives with all coefficients equal to one are used.
    """

    ring = PolynomialRing(QQ, names=("x1", "x2", "x3"))
    x1, x2, x3 = ring.gens()

    f12 = x1 * x2 + x1 + x2 + 1
    f13 = x1 * x3 + x1 + x3 + 1
    f23 = x2 * x3 + x2 + x3 + 1

    # Paper convention A*x <= b.  Include the redundant sum <= 3 inequality
    # to mirror P(theta) at theta=3; Sage may remove it from the minimal H-form.
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
    P = polytope_from_leq(A, b)

    result = compute_asymptotic_bound(
        [f12, f13, f23],
        MonomialOrder.deglex(("x1", "x2", "x3")),
        P,
        modulus_powers=(1, 1, 1),
    )

    expected = QQ(11) / QQ(24)
    if result.symmetric_exponent() != expected:
        raise AssertionError(
            "MIHNP regression failed: expected 11/24, got "
            f"{result.symmetric_exponent()}"
        )

    verification_points = list(P.vertices_list()) + [
        (QQ(1) / 2, QQ(1) / 2, QQ(1) / 2),
        (QQ(1) / 4, QQ(1) / 2, QQ(3) / 4),
    ]
    if not result.verify_phi(verification_points):
        raise AssertionError("MIHNP regression failed: primal/dual phi mismatch")
    return result


if __name__ == "__main__":
    demo_result = demo_mihnp_three_samples()
    print(demo_result.summary())
    print("\nLaTeX output")
    print("------------")
    print(demo_result.phi_latex())
    print(demo_result.bound_latex())
