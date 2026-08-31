from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np

StopMode = Literal["iterations", "function_evaluations"]


@dataclass(frozen=True)
class _RunControl:
    """Resolved stopping criterion used internally by every optimizer."""

    mode: StopMode
    requested_iterations: int | None
    max_function_evaluations: int | None
    nominal_iterations: int


class _ObjectiveEvaluator:
    """Count objective calls, enforce an optional cap, and trace improvements."""

    def __init__(
        self,
        function: Callable[[np.ndarray], float],
        max_function_evaluations: int | None,
    ) -> None:
        self.function = function
        self.max_function_evaluations = max_function_evaluations
        self.count = 0

        self.best_value = np.inf
        self.best_position: np.ndarray | None = None

        # Sparse exact-FE trace: a point is stored whenever the incumbent
        # improves. A terminal point is appended when the run finishes.
        self.function_evaluations_history: list[int] = []
        self.best_fitness_history: list[float] = []

    @property
    def remaining(self) -> int | None:
        if self.max_function_evaluations is None:
            return None
        return self.max_function_evaluations - self.count

    @property
    def exhausted(self) -> bool:
        remaining = self.remaining
        return remaining is not None and remaining <= 0

    def evaluate_one(self, position: np.ndarray) -> float | None:
        if self.exhausted:
            return None

        value = float(self.function(position))
        self.count += 1

        if value < self.best_value:
            self.best_value = value
            self.best_position = np.asarray(position, dtype=float).copy()
            self.function_evaluations_history.append(self.count)
            self.best_fitness_history.append(value)

        return value

    def evaluate_batch(
        self,
        positions: np.ndarray,
    ) -> tuple[np.ndarray, int]:
        """Evaluate all positions or the allowed prefix of the final batch."""

        n_requested = int(positions.shape[0])
        remaining = self.remaining
        n_to_evaluate = (
            n_requested
            if remaining is None
            else min(n_requested, max(remaining, 0))
        )

        values = np.empty(n_to_evaluate, dtype=float)
        for index in range(n_to_evaluate):
            value = self.evaluate_one(positions[index])
            if value is None:  # Defensive; n_to_evaluate already respects cap.
                return values[:index], index
            values[index] = value

        return values, n_to_evaluate

    def finalize_trace(self) -> None:
        if self.best_position is None:
            raise ValueError(
                "A função objetivo não produziu nenhum valor finito melhor "
                "que infinito."
            )

        if (
            not self.function_evaluations_history
            or self.function_evaluations_history[-1] != self.count
        ):
            self.function_evaluations_history.append(self.count)
            self.best_fitness_history.append(float(self.best_value))


def _resolve_run_control(
    *,
    n_particles: int,
    n_iterations: int | None,
    max_function_evaluations: int | None,
) -> _RunControl:
    if (n_iterations is None) == (max_function_evaluations is None):
        raise ValueError(
            "Informe exatamente um critério de parada: n_iterations ou "
            "max_function_evaluations."
        )

    if n_iterations is not None:
        if n_iterations < 1:
            raise ValueError("n_iterations deve ser pelo menos 1.")
        return _RunControl(
            mode="iterations",
            requested_iterations=n_iterations,
            max_function_evaluations=None,
            nominal_iterations=n_iterations,
        )

    assert max_function_evaluations is not None
    if max_function_evaluations < n_particles:
        raise ValueError(
            "max_function_evaluations deve permitir ao menos a avaliação "
            f"inicial das {n_particles} partículas."
        )

    remaining_after_initialization = max_function_evaluations - n_particles
    nominal_iterations = (
        remaining_after_initialization + n_particles - 1
    ) // n_particles

    return _RunControl(
        mode="function_evaluations",
        requested_iterations=None,
        max_function_evaluations=max_function_evaluations,
        nominal_iterations=nominal_iterations,
    )


def _new_history() -> dict[str, list]:
    return {
        "fitness": [],
        "w": [],
        "c1": [],
        "c2": [],
        "iterations": [],
        "iteration_function_evaluations": [],
        "function_evaluations": [],
        "evaluation_fitness": [],
    }


def _append_iteration_history(
    history: dict[str, list],
    *,
    completed_iterations: int,
    function_evaluations: int,
    fitness: float,
    w: float,
    c1: float,
    c2: float,
) -> None:
    history["fitness"].append(float(fitness))
    history["w"].append(float(w))
    history["c1"].append(float(c1))
    history["c2"].append(float(c2))
    history["iterations"].append(completed_iterations)
    history["iteration_function_evaluations"].append(function_evaluations)


def _finalize_run(
    optimizer: object,
    history: dict[str, list],
    evaluator: _ObjectiveEvaluator,
    control: _RunControl,
    completed_iterations: int,
) -> None:
    evaluator.finalize_trace()
    history["function_evaluations"] = list(
        evaluator.function_evaluations_history
    )
    history["evaluation_fitness"] = list(evaluator.best_fitness_history)

    optimizer.stop_mode = control.mode
    optimizer.requested_n_iterations = control.requested_iterations
    optimizer.requested_max_function_evaluations = (
        control.max_function_evaluations
    )
    optimizer.completed_iterations = completed_iterations
    optimizer.n_function_evaluations = evaluator.count
    optimizer.termination_reason = (
        "max_iterations"
        if control.mode == "iterations"
        else "max_function_evaluations"
    )


class PSOVectorized:
    def __init__(
        self,
        n_particles: int,
        low: float,
        high: float,
        w: float,
        c1: float,
        c2: float,
        dim: int,
        seed: int | None = None,
    ):
        self.n_particles = n_particles

        self.w = w
        self.c1 = c1
        self.c2 = c2

        self.dim = dim
        self.low = low
        self.high = high

        self.v_max = 0.2 * (high - low)

        self.rng = np.random.default_rng(seed)

        self.particles_positions = self.rng.uniform(
            low, high, (n_particles, dim)
        )
        self.particles_velocities = np.zeros((n_particles, dim))

        self.pbest_positions = self.particles_positions.copy()
        self.pbest_values = np.full(n_particles, np.inf)

        self.gbest_position = np.zeros(dim)
        self.gbest_value = np.inf

        self.current_fitness = np.full(n_particles, np.inf)

    def optimize(
        self,
        function: Callable,
        n_iterations: int | None = None,
        *,
        max_function_evaluations: int | None = None,
    ) -> tuple[
        float,
        np.ndarray,
        dict[str, list],
    ]:
        control = _resolve_run_control(
            n_particles=self.n_particles,
            n_iterations=n_iterations,
            max_function_evaluations=max_function_evaluations,
        )
        evaluator = _ObjectiveEvaluator(
            function,
            control.max_function_evaluations,
        )
        self.history = _new_history()

        initial_values, n_evaluated = evaluator.evaluate_batch(
            self.particles_positions
        )
        if n_evaluated != self.n_particles:
            raise RuntimeError("A avaliação inicial do enxame foi incompleta.")
        self.current_fitness = initial_values

        self.pbest_values = self.current_fitness.copy()
        self.pbest_positions = self.particles_positions.copy()

        min_idx = np.argmin(self.pbest_values)

        self.gbest_value = float(self.pbest_values[min_idx])
        self.gbest_position = self.pbest_positions[min_idx].copy()

        completed_iterations = 0

        for _iteration in range(control.nominal_iterations):
            if evaluator.exhausted:
                break

            r1 = self.rng.random(size=(self.n_particles, self.dim))
            r2 = self.rng.random(size=(self.n_particles, self.dim))

            inertia = self.w * self.particles_velocities
            cognitive = (
                self.c1 * r1 * (self.pbest_positions - self.particles_positions)
            )
            social = (
                self.c2 * r2 * (self.gbest_position - self.particles_positions)
            )

            self.particles_velocities = inertia + cognitive + social
            self.particles_velocities = np.clip(
                self.particles_velocities, -self.v_max, self.v_max
            )

            self.particles_positions += self.particles_velocities

            self.particles_positions = np.clip(
                self.particles_positions, self.low, self.high
            )

            evaluated_fitness, n_evaluated = evaluator.evaluate_batch(
                self.particles_positions
            )

            if n_evaluated == 0:
                break

            self.current_fitness[:n_evaluated] = evaluated_fitness
            mask_better = evaluated_fitness < self.pbest_values[:n_evaluated]

            self.pbest_values[:n_evaluated][mask_better] = evaluated_fitness[
                mask_better
            ]
            self.pbest_positions[:n_evaluated][mask_better] = (
                self.particles_positions[:n_evaluated][mask_better]
            )

            min_idx = int(np.argmin(self.pbest_values))

            if self.pbest_values[min_idx] < self.gbest_value:
                self.gbest_value = float(self.pbest_values[min_idx])
                self.gbest_position = self.pbest_positions[min_idx].copy()

            if n_evaluated < self.n_particles:
                break

            completed_iterations += 1
            _append_iteration_history(
                self.history,
                completed_iterations=completed_iterations,
                function_evaluations=evaluator.count,
                fitness=self.gbest_value,
                w=self.w,
                c1=self.c1,
                c2=self.c2,
            )

        _finalize_run(
            self,
            self.history,
            evaluator,
            control,
            completed_iterations,
        )

        return self.gbest_value, self.gbest_position.copy(), self.history

    def __repr__(self):
        return f"PSOVectorized(n_particles={self.n_particles}, w={self.w}, c1={self.c1}, c2={self.c2})"


class PSOLVIW:
    def __init__(
        self,
        n_particles: int,
        low: float,
        high: float,
        w_max: float,
        w_min: float,
        c1: float,
        c2: float,
        dim: int,
        seed: int | None = None,
    ):
        self.n_particles = n_particles

        self.w_max = w_max
        self.w_min = w_min

        self.w = w_max
        self.c1 = c1
        self.c2 = c2

        self.dim = dim
        self.low = low
        self.high = high

        self.v_max = 0.2 * (high - low)

        self.rng = np.random.default_rng(seed)

        self.particles_positions = self.rng.uniform(
            low, high, (n_particles, dim)
        )
        self.particles_velocities = np.zeros((n_particles, dim))

        self.pbest_positions = self.particles_positions.copy()
        self.pbest_values = np.full(n_particles, np.inf)

        self.gbest_position = np.zeros(dim)
        self.gbest_value = np.inf

        self.current_fitness = np.full(n_particles, np.inf)

    def optimize(
        self,
        function: Callable,
        n_iterations: int | None = None,
        *,
        max_function_evaluations: int | None = None,
    ) -> tuple[
        float,
        np.ndarray,
        dict[str, list],
    ]:
        control = _resolve_run_control(
            n_particles=self.n_particles,
            n_iterations=n_iterations,
            max_function_evaluations=max_function_evaluations,
        )
        evaluator = _ObjectiveEvaluator(
            function,
            control.max_function_evaluations,
        )
        self.history = _new_history()

        initial_values, n_evaluated = evaluator.evaluate_batch(
            self.particles_positions
        )
        if n_evaluated != self.n_particles:
            raise RuntimeError("A avaliação inicial do enxame foi incompleta.")
        self.current_fitness = initial_values

        self.pbest_values = self.current_fitness.copy()
        self.pbest_positions = self.particles_positions.copy()

        min_idx = np.argmin(self.pbest_values)

        self.gbest_value = float(self.pbest_values[min_idx])
        self.gbest_position = self.pbest_positions[min_idx].copy()

        denominator = max(control.nominal_iterations - 1, 1)
        completed_iterations = 0

        for iteration in range(control.nominal_iterations):
            if evaluator.exhausted:
                break

            self.w = (
                self.w_max - (self.w_max - self.w_min) * iteration / denominator
            )
            r1 = self.rng.random(size=(self.n_particles, self.dim))
            r2 = self.rng.random(size=(self.n_particles, self.dim))

            inertia = self.w * self.particles_velocities
            cognitive = (
                self.c1 * r1 * (self.pbest_positions - self.particles_positions)
            )
            social = (
                self.c2 * r2 * (self.gbest_position - self.particles_positions)
            )

            self.particles_velocities = inertia + cognitive + social
            self.particles_velocities = np.clip(
                self.particles_velocities, -self.v_max, self.v_max
            )

            self.particles_positions += self.particles_velocities

            self.particles_positions = np.clip(
                self.particles_positions, self.low, self.high
            )

            evaluated_fitness, n_evaluated = evaluator.evaluate_batch(
                self.particles_positions
            )
            if n_evaluated == 0:
                break

            self.current_fitness[:n_evaluated] = evaluated_fitness
            mask_better = evaluated_fitness < self.pbest_values[:n_evaluated]

            self.pbest_values[:n_evaluated][mask_better] = evaluated_fitness[
                mask_better
            ]
            self.pbest_positions[:n_evaluated][mask_better] = (
                self.particles_positions[:n_evaluated][mask_better]
            )

            min_idx = int(np.argmin(self.pbest_values))

            if self.pbest_values[min_idx] < self.gbest_value:
                self.gbest_value = float(self.pbest_values[min_idx])
                self.gbest_position = self.pbest_positions[min_idx].copy()

            if n_evaluated < self.n_particles:
                break

            completed_iterations += 1
            _append_iteration_history(
                self.history,
                completed_iterations=completed_iterations,
                function_evaluations=evaluator.count,
                fitness=self.gbest_value,
                w=self.w,
                c1=self.c1,
                c2=self.c2,
            )

        _finalize_run(
            self,
            self.history,
            evaluator,
            control,
            completed_iterations,
        )

        return self.gbest_value, self.gbest_position.copy(), self.history

    def __repr__(self):
        return f"PSOLVIW(n_particles={self.n_particles}, w={self.w}, c1={self.c1}, c2={self.c2})"


class PSOTVAC:
    def __init__(
        self,
        n_particles: int,
        low: float,
        high: float,
        w_min: float,
        w_max: float,
        c1_min: float,
        c1_max: float,
        c2_min: float,
        c2_max: float,
        dim: int,
        seed: int | None = None,
    ):
        self.n_particles = n_particles

        self.w_min = w_min
        self.w_max = w_max

        self.c1_min = c1_min
        self.c1_max = c1_max

        self.c2_min = c2_min
        self.c2_max = c2_max

        self.dim = dim
        self.low = low
        self.high = high

        self.v_max = 0.2 * (high - low)

        self.rng = np.random.default_rng(seed)

        self.particles_positions = self.rng.uniform(
            low, high, (n_particles, dim)
        )
        self.particles_velocities = np.zeros((n_particles, dim))

        self.pbest_positions = self.particles_positions.copy()
        self.pbest_values = np.full(n_particles, np.inf)

        self.gbest_position = np.zeros(dim)
        self.gbest_value = np.inf

        self.current_fitness = np.full(n_particles, np.inf)

    def optimize(
        self,
        function: Callable,
        n_iterations: int | None = None,
        *,
        max_function_evaluations: int | None = None,
    ) -> tuple[
        float,
        np.ndarray,
        dict[str, list],
    ]:
        control = _resolve_run_control(
            n_particles=self.n_particles,
            n_iterations=n_iterations,
            max_function_evaluations=max_function_evaluations,
        )
        evaluator = _ObjectiveEvaluator(
            function,
            control.max_function_evaluations,
        )
        self.history = _new_history()

        initial_values, n_evaluated = evaluator.evaluate_batch(
            self.particles_positions
        )
        if n_evaluated != self.n_particles:
            raise RuntimeError("A avaliação inicial do enxame foi incompleta.")
        self.current_fitness = initial_values

        self.pbest_values = self.current_fitness.copy()
        self.pbest_positions = self.particles_positions.copy()

        min_idx = int(np.argmin(self.pbest_values))

        self.gbest_value = float(self.pbest_values[min_idx])
        self.gbest_position = self.pbest_positions[min_idx].copy()

        denominator = max(control.nominal_iterations - 1, 1)
        completed_iterations = 0

        for iteration in range(control.nominal_iterations):
            if evaluator.exhausted:
                break

            self.c1 = (self.c1_min - self.c1_max) * (
                iteration / denominator
            ) + self.c1_max
            self.c2 = (self.c2_max - self.c2_min) * (
                iteration / denominator
            ) + self.c2_min
            self.w = (self.w_max - self.w_min) * (
                (denominator - iteration) / denominator
            ) + self.w_min

            r1 = self.rng.random(size=(self.n_particles, self.dim))
            r2 = self.rng.random(size=(self.n_particles, self.dim))

            inertia = self.w * self.particles_velocities
            cognitive = (
                self.c1 * r1 * (self.pbest_positions - self.particles_positions)
            )
            social = (
                self.c2 * r2 * (self.gbest_position - self.particles_positions)
            )

            self.particles_velocities = inertia + cognitive + social
            self.particles_velocities = np.clip(
                self.particles_velocities, -self.v_max, self.v_max
            )

            self.particles_positions += self.particles_velocities
            self.particles_positions = np.clip(
                self.particles_positions, self.low, self.high
            )

            evaluated_fitness, n_evaluated = evaluator.evaluate_batch(
                self.particles_positions
            )
            if n_evaluated == 0:
                break

            self.current_fitness[:n_evaluated] = evaluated_fitness
            mask_better = evaluated_fitness < self.pbest_values[:n_evaluated]

            self.pbest_values[:n_evaluated][mask_better] = evaluated_fitness[
                mask_better
            ]
            self.pbest_positions[:n_evaluated][mask_better] = (
                self.particles_positions[:n_evaluated][mask_better]
            )

            min_idx = int(np.argmin(self.pbest_values))

            if self.pbest_values[min_idx] < self.gbest_value:
                self.gbest_value = float(self.pbest_values[min_idx])
                self.gbest_position = self.pbest_positions[min_idx].copy()

            if n_evaluated < self.n_particles:
                break

            completed_iterations += 1
            _append_iteration_history(
                self.history,
                completed_iterations=completed_iterations,
                function_evaluations=evaluator.count,
                fitness=self.gbest_value,
                w=self.w,
                c1=self.c1,
                c2=self.c2,
            )

        _finalize_run(
            self,
            self.history,
            evaluator,
            control,
            completed_iterations,
        )

        return self.gbest_value, self.gbest_position.copy(), self.history

    def __repr__(self):
        return f"PSOTVAC(n_particles={self.n_particles}, w={self.w}, c1={self.c1}, c2={self.c2})"


class APSOVI:
    def __init__(
        self,
        n_particles: int,
        low: float,
        high: float,
        w_min: float,
        w_max: float,
        step_size,
        c1: float,
        c2: float,
        dim: int,
        seed: int | None = None,
    ):
        self.n_particles = n_particles

        self.w_max = w_max
        self.w_min = w_min
        self.w = w_max
        self.step_size = step_size
        self.c1 = c1
        self.c2 = c2

        self.dim = dim
        self.low = low
        self.high = high

        self.v_max = 0.2 * (high - low)
        self.v_inicial = (high - low) / 2.0

        self.rng = np.random.default_rng(seed)

        self.particles_positions = self.rng.uniform(
            low, high, (n_particles, dim)
        )
        self.particles_velocities = np.zeros((n_particles, dim))

        self.pbest_positions = self.particles_positions.copy()
        self.pbest_values = np.full(n_particles, np.inf)

        self.gbest_position = np.zeros(dim)
        self.gbest_value = np.inf

        self.current_fitness = np.full(n_particles, np.inf)

    def optimize(
        self,
        function: Callable,
        n_iterations: int | None = None,
        *,
        max_function_evaluations: int | None = None,
    ) -> tuple[
        float,
        np.ndarray,
        dict[str, list],
    ]:
        control = _resolve_run_control(
            n_particles=self.n_particles,
            n_iterations=n_iterations,
            max_function_evaluations=max_function_evaluations,
        )
        evaluator = _ObjectiveEvaluator(
            function,
            control.max_function_evaluations,
        )
        self.history = _new_history()

        self.w = self.w_max

        initial_values, n_evaluated = evaluator.evaluate_batch(
            self.particles_positions
        )
        if n_evaluated != self.n_particles:
            raise RuntimeError("A avaliação inicial do enxame foi incompleta.")
        self.current_fitness = initial_values

        self.pbest_values = self.current_fitness.copy()
        self.pbest_positions = self.particles_positions.copy()

        min_idx = int(np.argmin(self.pbest_values))

        self.gbest_value = float(self.pbest_values[min_idx])
        self.gbest_position = self.pbest_positions[min_idx].copy()

        t_end = 0.95 * control.nominal_iterations
        completed_iterations = 0

        for iteration in range(control.nominal_iterations):
            if evaluator.exhausted:
                break

            v_ave = np.sum(np.absolute(self.particles_velocities)) / (
                self.n_particles * self.dim
            )

            next_iteration = iteration + 1

            if next_iteration >= t_end:
                v_ideal_next = 0.0
            else:
                v_ideal_next = (
                    self.v_inicial
                    * (1.0 + np.cos((iteration + 1.0) * np.pi / t_end))
                    / 2.0
                )

            if v_ave >= v_ideal_next:
                self.w = np.max([self.w - self.step_size, self.w_min])
            else:
                self.w = np.min([self.w + self.step_size, self.w_max])

            r1 = self.rng.random(size=(self.n_particles, self.dim))
            r2 = self.rng.random(size=(self.n_particles, self.dim))

            inertia = self.w * self.particles_velocities
            cognitive = (
                self.c1 * r1 * (self.pbest_positions - self.particles_positions)
            )
            social = (
                self.c2 * r2 * (self.gbest_position - self.particles_positions)
            )

            self.particles_velocities = inertia + cognitive + social
            self.particles_velocities = np.clip(
                self.particles_velocities, -self.v_max, self.v_max
            )

            self.particles_positions += self.particles_velocities

            self.particles_positions = np.clip(
                self.particles_positions, self.low, self.high
            )

            evaluated_fitness, n_evaluated = evaluator.evaluate_batch(
                self.particles_positions
            )
            if n_evaluated == 0:
                break

            self.current_fitness[:n_evaluated] = evaluated_fitness
            mask_better = evaluated_fitness < self.pbest_values[:n_evaluated]

            self.pbest_values[:n_evaluated][mask_better] = evaluated_fitness[
                mask_better
            ]
            self.pbest_positions[:n_evaluated][mask_better] = (
                self.particles_positions[:n_evaluated][mask_better]
            )

            min_idx = int(np.argmin(self.pbest_values))

            if self.pbest_values[min_idx] < self.gbest_value:
                self.gbest_value = float(self.pbest_values[min_idx])
                self.gbest_position = self.pbest_positions[min_idx].copy()

            if n_evaluated < self.n_particles:
                break

            completed_iterations += 1
            _append_iteration_history(
                self.history,
                completed_iterations=completed_iterations,
                function_evaluations=evaluator.count,
                fitness=self.gbest_value,
                w=self.w,
                c1=self.c1,
                c2=self.c2,
            )

        _finalize_run(
            self,
            self.history,
            evaluator,
            control,
            completed_iterations,
        )

        return self.gbest_value, self.gbest_position.copy(), self.history

    def __repr__(self):
        return f"APSOVI(n_particles={self.n_particles}, w={self.w}, c1={self.c1}, c2={self.c2})"


class APSO:
    def __init__(
        self,
        n_particles: int,
        low: float,
        high: float,
        w: float,
        c1: float,
        c2: float,
        dim: int,
        seed: int | None = None,
    ):
        if n_particles < 2:
            raise ValueError("n_particles deve ser pelo menos 2.")

        if dim < 1:
            raise ValueError("dim deve ser pelo menos 1.")

        if low >= high:
            raise ValueError("low deve ser menor que high.")

        self.n_particles = n_particles
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.dim = dim
        self.low = low
        self.high = high

        # O paper utiliza 20% do intervalo de busca.
        self.v_max = 0.2 * (high - low)

        self.rng = np.random.default_rng(seed)

        self.particles_positions = self.rng.uniform(
            low,
            high,
            size=(n_particles, dim),
        )

        self.particles_velocities = np.zeros((n_particles, dim))

        self.pbest_positions = self.particles_positions.copy()
        self.pbest_values = np.full(n_particles, np.inf)

        self.gbest_position = np.zeros(dim)
        self.gbest_value = np.inf

        # Mantém explicitamente o índice da partícula que contém o gBest.
        self.gbest_index: int | None = None

        self.current_fitness = np.full(n_particles, np.inf)

        # O estado inicial é considerado exploration.
        self.current_state = "exploration"
        self.previous_state: str | None = None

        self.n_function_evaluations = 0

        self.history: dict[str, list] = {}

    def _evaluate_swarm(
        self,
        evaluator: _ObjectiveEvaluator,
    ) -> int:
        """
        Avalia a população permitida pelo orçamento e atualiza pBest e gBest.
        """
        evaluated_fitness, n_evaluated = evaluator.evaluate_batch(
            self.particles_positions
        )
        self.n_function_evaluations = evaluator.count

        if n_evaluated == 0:
            return 0

        if np.any(np.isnan(evaluated_fitness)):
            raise ValueError(
                "A função objetivo retornou NaN para pelo menos uma partícula."
            )

        self.current_fitness[:n_evaluated] = evaluated_fitness
        mask_better = evaluated_fitness < self.pbest_values[:n_evaluated]

        self.pbest_values[:n_evaluated][mask_better] = evaluated_fitness[
            mask_better
        ]

        self.pbest_positions[:n_evaluated][mask_better] = (
            self.particles_positions[:n_evaluated][mask_better]
        )

        best_idx = int(np.argmin(self.pbest_values))

        if self.pbest_values[best_idx] < self.gbest_value:
            self.gbest_index = best_idx
            self.gbest_value = float(self.pbest_values[best_idx])
            self.gbest_position = self.pbest_positions[best_idx].copy()

        return n_evaluated

    def _calculate_evolutionary_factor(self) -> float:
        """
        Calcula o fator evolutivo f das equações (7) e (8).
        """
        diff = (
            self.particles_positions[:, np.newaxis, :]
            - self.particles_positions[np.newaxis, :, :]
        )

        pairwise_distances = np.linalg.norm(diff, axis=2)

        # A diagonal é zero. A divisão por N - 1 implementa
        # exatamente a média definida na equação (7).
        mean_distances = pairwise_distances.sum(axis=1) / (self.n_particles - 1)

        if self.gbest_index is None:
            raise RuntimeError(
                "O enxame deve ser avaliado antes do cálculo de f."
            )

        d_g = float(mean_distances[self.gbest_index])
        d_min = float(np.min(mean_distances))
        d_max = float(np.max(mean_distances))

        denominator = d_max - d_min

        if np.isclose(denominator, 0.0):
            # Caso todas as partículas estejam na mesma posição,
            # o enxame está efetivamente convergido.
            if np.allclose(pairwise_distances, 0.0):
                return 0.0

            # Caso degenerado, como uma configuração perfeitamente
            # simétrica. O paper não especifica esse tratamento.
            return 0.5

        f = (d_g - d_min) / denominator

        return float(np.clip(f, 0.0, 1.0))

    @staticmethod
    def _calculate_memberships(
        f: float,
    ) -> dict[str, float]:
        # S1: Exploration
        if 0.0 <= f <= 0.4:
            state_1 = 0.0
        elif f <= 0.6:
            state_1 = 5.0 * f - 2.0
        elif f <= 0.7:
            state_1 = 1.0
        elif f <= 0.8:
            state_1 = -10.0 * f + 8.0
        else:
            state_1 = 0.0

        # S2: Exploitation
        if 0.0 <= f <= 0.2:
            state_2 = 0.0
        elif f <= 0.3:
            state_2 = 10.0 * f - 2.0
        elif f <= 0.4:
            state_2 = 1.0
        elif f <= 0.6:
            state_2 = -5.0 * f + 3.0
        else:
            state_2 = 0.0

        # S3: Convergence
        if 0.0 <= f <= 0.1:
            state_3 = 1.0
        elif f <= 0.3:
            state_3 = -5.0 * f + 1.5
        else:
            state_3 = 0.0

        # S4: Jumping out
        if 0.0 <= f <= 0.7:
            state_4 = 0.0
        elif f <= 0.9:
            state_4 = 5.0 * f - 3.5
        else:
            state_4 = 1.0

        return {
            "exploration": float(np.clip(state_1, 0.0, 1.0)),
            "exploitation": float(np.clip(state_2, 0.0, 1.0)),
            "convergence": float(np.clip(state_3, 0.0, 1.0)),
            "jumping_out": float(np.clip(state_4, 0.0, 1.0)),
        }

    def _get_state(self, f: float) -> str:
        """
        Classifica o estado evolutivo usando as pertinências fuzzy
        e a sequência:

        exploration -> exploitation -> convergence
        -> jumping_out -> exploration
        """
        scores = self._calculate_memberships(f)

        active_states = {
            state for state, membership in scores.items() if membership > 0.0
        }

        previous_state = self.current_state

        # Em uma região de sobreposição, mantém o estado anterior
        # quando ele ainda possui pertinência positiva. Isso evita
        # trocas excessivas de estado.
        if previous_state in active_states:
            return previous_state

        next_state = {
            "exploration": "exploitation",
            "exploitation": "convergence",
            "convergence": "jumping_out",
            "jumping_out": "exploration",
        }[previous_state]

        # Se a próxima etapa da sequência estiver ativa,
        # favorece essa transição.
        if next_state in active_states:
            return next_state

        # Fora das regiões de sobreposição, utiliza singleton:
        # escolhe a maior pertinência.
        return max(scores, key=scores.get)

    def _adapt_parameters(
        self,
        state: str,
        f: float,
    ) -> None:
        """
        Adapta w, c1 e c2 de acordo com as equações (10)-(12)
        e a Tabela II do paper.
        """

        self.w = float(
            np.clip(
                1.0 / (1.0 + 1.5 * np.exp(-2.6 * f)),
                0.4,
                0.9,
            )
        )

        delta = float(self.rng.uniform(0.05, 0.1))

        if state == "exploration":
            self.c1 += delta
            self.c2 -= delta

        elif state == "exploitation":
            self.c1 += 0.5 * delta
            self.c2 -= 0.5 * delta

        elif state == "convergence":
            self.c1 += 0.5 * delta
            self.c2 += 0.5 * delta

        elif state == "jumping_out":
            self.c1 -= delta
            self.c2 += delta

        else:
            raise ValueError(f"Estado evolutivo desconhecido: {state!r}")

        self.c1 = float(np.clip(self.c1, 1.5, 2.5))
        self.c2 = float(np.clip(self.c2, 1.5, 2.5))

        sum_c = self.c1 + self.c2

        if sum_c > 4.0:
            self.c1 = (self.c1 / sum_c) * 4.0
            self.c2 = (self.c2 / sum_c) * 4.0

    def _elitist_learning(
        self,
        evaluator: _ObjectiveEvaluator,
        current_iteration: int,
        n_iterations: int,
    ) -> bool:
        """
        Executa a Elitist Learning Strategy das equações (13) e (14).
        """
        if self.gbest_index is None:
            raise RuntimeError(
                "O enxame deve ser avaliado antes da execução do ELS."
            )

        candidate_position = self.gbest_position.copy()

        sigma_max = 1.0
        sigma_min = 0.1

        sigma = sigma_max - (sigma_max - sigma_min) * (
            current_iteration / n_iterations
        )

        dimension_idx = int(self.rng.integers(0, self.dim))

        gaussian_noise = float(
            self.rng.normal(
                loc=0.0,
                scale=sigma,
            )
        )

        candidate_position[dimension_idx] += (
            self.high - self.low
        ) * gaussian_noise

        candidate_position[dimension_idx] = np.clip(
            candidate_position[dimension_idx],
            self.low,
            self.high,
        )

        candidate_value = evaluator.evaluate_one(candidate_position)
        self.n_function_evaluations = evaluator.count
        if candidate_value is None:
            return False

        if np.isnan(candidate_value):
            return True

        if candidate_value < self.gbest_value:
            # O gBest deve continuar sendo o pBest de alguma partícula.
            # Portanto, atualizamos o pBest da partícula líder.
            leader_idx = self.gbest_index

            self.pbest_positions[leader_idx] = candidate_position.copy()
            self.pbest_values[leader_idx] = candidate_value

            self.gbest_position = candidate_position.copy()
            self.gbest_value = candidate_value

        else:
            # O paper determina que o candidato que não melhora
            # o gBest substitua a pior partícula do enxame.
            worst_idx = int(np.argmax(self.current_fitness))

            self.particles_positions[worst_idx] = candidate_position.copy()
            self.current_fitness[worst_idx] = candidate_value

            # Mantém a consistência do pBest da partícula substituída.
            if candidate_value < self.pbest_values[worst_idx]:
                self.pbest_positions[worst_idx] = candidate_position.copy()
                self.pbest_values[worst_idx] = candidate_value

        return True

    def optimize(
        self,
        function: Callable[[np.ndarray], float],
        n_iterations: int | None = None,
        *,
        max_function_evaluations: int | None = None,
    ) -> tuple[
        float,
        np.ndarray,
        dict[str, list],
    ]:
        control = _resolve_run_control(
            n_particles=self.n_particles,
            n_iterations=n_iterations,
            max_function_evaluations=max_function_evaluations,
        )
        evaluator = _ObjectiveEvaluator(
            function,
            control.max_function_evaluations,
        )

        self.history = _new_history()
        self.history.update({"f": [], "state": []})

        self.n_function_evaluations = 0

        # Avaliação da população inicial.
        n_evaluated = self._evaluate_swarm(evaluator)
        if n_evaluated != self.n_particles:
            raise RuntimeError("A avaliação inicial do enxame foi incompleta.")

        completed_iterations = 0

        for iteration in range(control.nominal_iterations):
            if evaluator.exhausted:
                break

            # ESE utiliza a população corrente e o gBest conhecido.
            f = self._calculate_evolutionary_factor()

            state = self._get_state(f)

            # Agora previous_state recebe realmente o estado
            # imediatamente anterior, e não o estado de duas
            # gerações atrás.
            self.previous_state = self.current_state
            self.current_state = state

            self._adapt_parameters(
                state=state,
                f=f,
            )

            if state == "convergence":
                els_evaluated = self._elitist_learning(
                    evaluator=evaluator,
                    current_iteration=iteration,
                    n_iterations=control.nominal_iterations,
                )
                if not els_evaluated or evaluator.exhausted:
                    break

            r1 = self.rng.random(size=(self.n_particles, self.dim))
            r2 = self.rng.random(size=(self.n_particles, self.dim))

            inertia = self.w * self.particles_velocities

            cognitive = (
                self.c1 * r1 * (self.pbest_positions - self.particles_positions)
            )

            social = (
                self.c2 * r2 * (self.gbest_position - self.particles_positions)
            )

            self.particles_velocities = inertia + cognitive + social

            self.particles_velocities = np.clip(
                self.particles_velocities,
                -self.v_max,
                self.v_max,
            )

            self.particles_positions += self.particles_velocities

            # Estratégia de tratamento de fronteiras mantida
            # conforme sua implementação original.
            self.particles_positions = np.clip(
                self.particles_positions,
                self.low,
                self.high,
            )

            # Avalia as posições produzidas pela geração atual.
            # Assim, a última população não é ignorada.
            n_evaluated = self._evaluate_swarm(evaluator)
            if n_evaluated == 0:
                break

            if n_evaluated < self.n_particles:
                break

            completed_iterations += 1
            _append_iteration_history(
                self.history,
                completed_iterations=completed_iterations,
                function_evaluations=evaluator.count,
                fitness=self.gbest_value,
                w=self.w,
                c1=self.c1,
                c2=self.c2,
            )
            self.history["f"].append(f)
            self.history["state"].append(state)

        _finalize_run(
            self,
            self.history,
            evaluator,
            control,
            completed_iterations,
        )

        return (
            self.gbest_value,
            self.gbest_position.copy(),
            self.history,
        )

    def __repr__(self) -> str:
        return (
            f"APSO("
            f"n_particles={self.n_particles}, "
            f"w={self.w:.4f}, "
            f"c1={self.c1:.4f}, "
            f"c2={self.c2:.4f}, "
            f"function_evaluations="
            f"{self.n_function_evaluations}"
            f")"
        )


class UAPSO:
    def __init__(
        self,
        n_particles: int,
        low: float,
        high: float,
        w_min: float = 0.0,
        w_max: float = 1.0,
        c1_min: float = 0.0,
        c1_max: float = 4.0,
        c2_min: float = 0.0,
        c2_max: float = 4.0,
        threshold: float = 0.5,
        learning_rate: float = 0.01,
        dim: int = 30,
        seed: int | None = None,
    ):
        self.n_particles = n_particles
        self.dim = dim
        self.low = low
        self.high = high
        self.threshold = threshold
        self.learning_rate = learning_rate

        self.la_w_actions = np.linspace(w_min, w_max, 20)
        self.la_c1_actions = np.linspace(c1_min, c1_max, 10)
        self.la_c2_actions = np.linspace(c2_min, c2_max, 10)

        self.probs_la_w = np.full(20, 1.0 / 20)
        self.probs_la_c1 = np.full(10, 1.0 / 10)
        self.probs_la_c2 = np.full(10, 1.0 / 10)

        self.v_max = 0.2 * (high - low)
        self.rng = np.random.default_rng(seed)
        self.particles_positions = self.rng.uniform(
            low, high, (n_particles, dim)
        )

        self.particles_velocities = np.zeros((n_particles, dim))

        self.pbest_positions = self.particles_positions.copy()
        self.pbest_values = np.full(n_particles, np.inf)

        self.gbest_position = np.zeros(dim)
        self.gbest_value = np.inf

        self.current_fitness = np.full(n_particles, np.inf)
        self.history = {"fitness": [], "w": [], "c1": [], "c2": []}

    def _update_probs(
        self, probs: np.ndarray, chosen_idx: int, is_success: bool
    ) -> np.ndarray:
        r = len(probs)
        a = self.learning_rate
        b = self.learning_rate

        new_probs = probs.copy()

        if is_success:
            new_probs[chosen_idx] += a * (1 - new_probs[chosen_idx])

            mask = np.arange(r) != chosen_idx
            new_probs[mask] *= 1 - a
        else:
            new_probs[chosen_idx] *= 1 - b

            mask = np.arange(r) != chosen_idx
            dist_term = b / (r - 1)
            new_probs[mask] = dist_term + (1 - b) * new_probs[mask]

        return new_probs / new_probs.sum()

    def optimize(
        self,
        function: Callable,
        n_iterations: int | None = None,
        *,
        max_function_evaluations: int | None = None,
    ) -> tuple[
        float,
        np.ndarray,
        dict[str, list],
    ]:
        control = _resolve_run_control(
            n_particles=self.n_particles,
            n_iterations=n_iterations,
            max_function_evaluations=max_function_evaluations,
        )
        evaluator = _ObjectiveEvaluator(
            function,
            control.max_function_evaluations,
        )
        self.history = _new_history()

        initial_values, n_evaluated = evaluator.evaluate_batch(
            self.particles_positions
        )
        if n_evaluated != self.n_particles:
            raise RuntimeError("A avaliação inicial do enxame foi incompleta.")
        self.current_fitness = initial_values

        self.pbest_values = self.current_fitness.copy()
        self.pbest_positions = self.particles_positions.copy()

        min_idx = int(np.argmin(self.current_fitness))
        if self.current_fitness[min_idx] < self.gbest_value:
            self.gbest_value = float(self.current_fitness[min_idx])
            self.gbest_position = self.particles_positions[min_idx].copy()

        completed_iterations = 0

        for _iteration in range(control.nominal_iterations):
            if evaluator.exhausted:
                break

            previous_fitness = self.current_fitness.copy()

            idx_w = self.rng.choice(len(self.la_w_actions), p=self.probs_la_w)
            idx_c1 = self.rng.choice(
                len(self.la_c1_actions), p=self.probs_la_c1
            )
            idx_c2 = self.rng.choice(
                len(self.la_c2_actions), p=self.probs_la_c2
            )

            w = self.la_w_actions[idx_w]
            c1 = self.la_c1_actions[idx_c1]
            c2 = self.la_c2_actions[idx_c2]

            r1 = self.rng.random(size=(self.n_particles, self.dim))
            r2 = self.rng.random(size=(self.n_particles, self.dim))

            inertia = w * self.particles_velocities
            cognitive = (
                c1 * r1 * (self.pbest_positions - self.particles_positions)
            )
            social = c2 * r2 * (self.gbest_position - self.particles_positions)

            self.particles_velocities = inertia + cognitive + social

            self.particles_velocities = np.clip(
                self.particles_velocities, -self.v_max, self.v_max
            )

            self.particles_positions += self.particles_velocities
            self.particles_positions = np.clip(
                self.particles_positions, self.low, self.high
            )

            evaluated_fitness, n_evaluated = evaluator.evaluate_batch(
                self.particles_positions
            )
            if n_evaluated == 0:
                break

            self.current_fitness[:n_evaluated] = evaluated_fitness
            mask_better = evaluated_fitness < self.pbest_values[:n_evaluated]
            self.pbest_values[:n_evaluated][mask_better] = evaluated_fitness[
                mask_better
            ]
            self.pbest_positions[:n_evaluated][mask_better] = (
                self.particles_positions[:n_evaluated][mask_better]
            )

            min_idx = int(np.argmin(self.pbest_values))
            if self.pbest_values[min_idx] < self.gbest_value:
                self.gbest_value = float(self.pbest_values[min_idx])
                self.gbest_position = self.pbest_positions[min_idx].copy()

            # A recompensa do autômato depende da população completa.
            if n_evaluated < self.n_particles:
                break

            n_improved = np.sum(self.current_fitness < previous_fitness)
            ratio = n_improved / self.n_particles

            is_successful = ratio >= self.threshold

            self.probs_la_w = self._update_probs(
                self.probs_la_w, idx_w, is_successful
            )
            self.probs_la_c1 = self._update_probs(
                self.probs_la_c1, idx_c1, is_successful
            )
            self.probs_la_c2 = self._update_probs(
                self.probs_la_c2, idx_c2, is_successful
            )

            completed_iterations += 1
            _append_iteration_history(
                self.history,
                completed_iterations=completed_iterations,
                function_evaluations=evaluator.count,
                fitness=self.gbest_value,
                w=float(w),
                c1=float(c1),
                c2=float(c2),
            )

        _finalize_run(
            self,
            self.history,
            evaluator,
            control,
            completed_iterations,
        )

        return self.gbest_value, self.gbest_position.copy(), self.history

    def __repr__(self):
        return f"UAPSO(n_particles={self.n_particles}, dim={self.dim}, current_gbest={self.gbest_value:.4f})"
