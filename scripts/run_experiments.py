from __future__ import annotations

import os
import time
import traceback
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from configs.benchmarks import BenchmarksCEC2017Config
from configs.hyperparameters import Algorithms, HyperparametersConfig
from src.pso import APSO, APSOVI, PSOLVIW, PSOTVAC, UAPSO, PSOVectorized

ObjectiveFunction = Callable[[np.ndarray], float]
StopMode = Literal["iterations", "function_evaluations"]


@dataclass
class ExperimentsConfig:
    algorithms: Algorithms | None = None
    config_hyperparameters: HyperparametersConfig | None = None
    config_benchmarks: BenchmarksCEC2017Config | None = None

    output_dir: str = "results"
    n_runs: int = 1
    dim: list[int] = field(default_factory=lambda: [30, 50, 100])
    stop_mode: StopMode = "iterations"

    # (10_000 * dimensão) // 50.
    n_iterations_by_dim: dict[int, int] = field(
        default_factory=lambda: {
            30: 6_000,
            50: 10_000,
            100: 20_000,
        }
    )

    max_function_evaluations_by_dim: dict[int, int] = field(
        default_factory=lambda: {
            30: 300_000,
            50: 500_000,
            100: 1_000_000,
        }
    )

    n_jobs: int = 6
    seed: int = 42
    measure_memory: bool = True
    fail_on_error: bool = True


def _validate_config(config: ExperimentsConfig) -> None:
    if config.algorithms is None:
        raise ValueError("config.algorithms não pode ser None.")

    if config.config_hyperparameters is None:
        raise ValueError("config.config_hyperparameters não pode ser None.")

    if config.config_benchmarks is None:
        raise ValueError("config.config_benchmarks não pode ser None.")

    if config.n_runs < 1:
        raise ValueError("n_runs deve ser pelo menos 1.")

    if config.n_jobs == 0:
        raise ValueError("n_jobs não pode ser zero.")

    if not config.dim:
        raise ValueError("A lista de dimensões não pode estar vazia.")

    if config.stop_mode not in {"iterations", "function_evaluations"}:
        raise ValueError(
            "stop_mode deve ser 'iterations' ou 'function_evaluations'."
        )

    budget_by_dim = (
        config.n_iterations_by_dim
        if config.stop_mode == "iterations"
        else config.max_function_evaluations_by_dim
    )

    missing_dimensions = [dim for dim in config.dim if dim not in budget_by_dim]
    if missing_dimensions:
        raise ValueError(
            f"Não há orçamento configurado para as dimensões: {missing_dimensions}."
        )

    invalid_budgets = {
        dim: budget_by_dim[dim] for dim in config.dim if budget_by_dim[dim] < 1
    }
    if invalid_budgets:
        raise ValueError(f"O orçamento deve ser positivo: {invalid_budgets}.")

    if config.stop_mode == "function_evaluations":
        n_particles = config.config_hyperparameters.n_particles
        insufficient_budgets = {
            dim: budget_by_dim[dim]
            for dim in config.dim
            if budget_by_dim[dim] < n_particles
        }
        if insufficient_budgets:
            raise ValueError(
                "O orçamento de FEs deve permitir a avaliação inicial de "
                f"{n_particles} partículas: {insufficient_budgets}."
            )


def _algorithm_mapping(algorithms: Algorithms) -> dict[str, type]:
    return {
        item.name: getattr(algorithms, item.name) for item in fields(algorithms)
    }


def _hyperparameter_mapping(
    config: HyperparametersConfig,
) -> dict[str, dict[str, Any]]:
    return {
        name: value
        for name, value in vars(config).items()
        if isinstance(value, dict)
    }


def _run_single(
    algo_class: type,
    algo_hyperparams: dict[str, Any],
    function: ObjectiveFunction,
    stop_mode: StopMode,
    budget_value: int,
    run_id: int,
    base_seed: int,
    measure_memory: bool,
) -> dict[str, Any]:
    current_seed = base_seed + run_id

    run_params = algo_hyperparams.copy()
    run_params["seed"] = current_seed

    if measure_memory:
        tracemalloc.start()

    try:
        initialization_start = time.perf_counter()
        optimizer = algo_class(**run_params)
        initialization_end = time.perf_counter()

        optimization_start = time.perf_counter()
        optimize_kwargs = (
            {"n_iterations": budget_value}
            if stop_mode == "iterations"
            else {"max_function_evaluations": budget_value}
        )
        best_fitness, best_position, history = optimizer.optimize(
            function,
            **optimize_kwargs,
        )
        optimization_end = time.perf_counter()

        if measure_memory:
            current_memory, peak_memory = tracemalloc.get_traced_memory()
            current_memory_mb = current_memory / (1024**2)
            peak_memory_mb = peak_memory / (1024**2)
        else:
            current_memory_mb = np.nan
            peak_memory_mb = np.nan

        known_history_keys = {
            "fitness",
            "w",
            "c1",
            "c2",
            "iterations",
            "iteration_function_evaluations",
            "function_evaluations",
            "evaluation_fitness",
        }
        extra_history = {
            key: value
            for key, value in history.items()
            if key not in known_history_keys
        }

        return {
            "success": True,
            "run_id": run_id,
            "seed": current_seed,
            "best_fitness": float(best_fitness),
            "best_position": np.asarray(best_position, dtype=float).copy(),
            "fitness_history": list(history.get("fitness", [])),
            "iteration_history": list(history.get("iterations", [])),
            "iteration_function_evaluations_history": list(
                history.get("iteration_function_evaluations", [])
            ),
            "function_evaluations_history": list(
                history.get("function_evaluations", [])
            ),
            "evaluation_fitness_history": list(
                history.get("evaluation_fitness", [])
            ),
            "w_history": list(history.get("w", [])),
            "c1_history": list(history.get("c1", [])),
            "c2_history": list(history.get("c2", [])),
            "extra_history": extra_history,
            "stop_mode": stop_mode,
            "budget_value": budget_value,
            "requested_n_iterations": getattr(
                optimizer, "requested_n_iterations", None
            ),
            "requested_max_function_evaluations": getattr(
                optimizer,
                "requested_max_function_evaluations",
                None,
            ),
            "completed_iterations": int(optimizer.completed_iterations),
            "n_function_evaluations": int(optimizer.n_function_evaluations),
            "unused_function_evaluations": (
                budget_value - int(optimizer.n_function_evaluations)
                if stop_mode == "function_evaluations"
                else np.nan
            ),
            "termination_reason": optimizer.termination_reason,
            "initialization_time_s": initialization_end - initialization_start,
            "execution_time_s": optimization_end - optimization_start,
            "current_python_memory_mb": current_memory_mb,
            "peak_python_memory_mb": peak_memory_mb,
            "error_type": None,
            "error_message": None,
            "traceback": None,
        }

    except Exception as error:
        return {
            "success": False,
            "run_id": run_id,
            "seed": current_seed,
            "best_fitness": np.nan,
            "best_position": None,
            "fitness_history": [],
            "iteration_history": [],
            "iteration_function_evaluations_history": [],
            "function_evaluations_history": [],
            "evaluation_fitness_history": [],
            "w_history": [],
            "c1_history": [],
            "c2_history": [],
            "extra_history": {},
            "stop_mode": stop_mode,
            "budget_value": budget_value,
            "requested_n_iterations": (
                budget_value if stop_mode == "iterations" else None
            ),
            "requested_max_function_evaluations": (
                budget_value if stop_mode == "function_evaluations" else None
            ),
            "completed_iterations": np.nan,
            "n_function_evaluations": np.nan,
            "unused_function_evaluations": np.nan,
            "termination_reason": "error",
            "initialization_time_s": np.nan,
            "execution_time_s": np.nan,
            "current_python_memory_mb": np.nan,
            "peak_python_memory_mb": np.nan,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
        }

    finally:
        if measure_memory and tracemalloc.is_tracing():
            tracemalloc.stop()


def _sample_std(
    values: np.ndarray,
    axis: int | None = None,
) -> np.ndarray | float:
    sample_size = values.shape[0] if axis is not None else values.size
    ddof = 1 if sample_size > 1 else 0
    return np.std(values, axis=axis, ddof=ddof)


def _stack_histories(
    results: list[dict[str, Any]],
    key: str,
    *,
    allow_empty: bool,
    truncate_to_common_length: bool = False,
) -> np.ndarray | None:
    histories = [result[key] for result in results]

    if allow_empty:
        histories = [history for history in histories if len(history) > 0]
        if not histories:
            return None

    lengths = {len(history) for history in histories}
    if len(lengths) != 1:
        if not truncate_to_common_length:
            raise ValueError(
                f"Os históricos de {key!r} possuem tamanhos diferentes: "
                f"{sorted(lengths)}."
            )

        common_length = min(lengths)
        if common_length == 0:
            return None
        histories = [history[:common_length] for history in histories]

    return np.asarray(histories, dtype=float)


def _step_resample(
    x_values: list[int] | np.ndarray,
    y_values: list[float] | np.ndarray,
    grid: np.ndarray,
    *,
    context: str,
) -> np.ndarray:
    x = np.asarray(x_values, dtype=int)
    y = np.asarray(y_values, dtype=float)

    if x.ndim != 1 or y.ndim != 1 or x.size != y.size or x.size == 0:
        raise ValueError(
            f"Histórico inválido para {context}: x e y devem ser vetores "
            "não vazios de mesmo tamanho."
        )
    if np.any(np.diff(x) <= 0):
        raise ValueError(
            f"O eixo de {context} deve ser estritamente crescente."
        )

    indices = np.searchsorted(x, grid, side="right") - 1
    if np.any(indices < 0):
        raise ValueError(
            f"A grade de {context} começa antes do primeiro ponto observado."
        )
    return y[indices]


def _build_regular_grid(start: int, end: int, step: int) -> np.ndarray:
    if end < start:
        return np.asarray([], dtype=int)

    grid = np.arange(start, end + 1, step, dtype=int)
    if grid.size == 0 or grid[-1] != end:
        grid = np.append(grid, end)
    return grid


def _aggregate_step_histories(
    results: list[dict[str, Any]],
    *,
    x_key: str,
    y_key: str,
    grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    if grid.size == 0:
        return None

    curves = [
        _step_resample(
            result[x_key],
            result[y_key],
            grid,
            context=f"{x_key}/{y_key}, run={result['run_id']}",
        )
        for result in results
    ]
    matrix = np.vstack(curves)
    return np.mean(matrix, axis=0), _sample_std(matrix, axis=0)


def _aggregate_results(
    results: list[dict[str, Any]],
    benchmark: str,
    dimension: int,
    algorithm: str,
    stop_mode: StopMode,
    budget_value: int,
    n_particles: int,
    expected_runs: int,
) -> dict[str, Any]:
    best_fitness = np.asarray(
        [result["best_fitness"] for result in results],
        dtype=float,
    )
    execution_times = np.asarray(
        [result["execution_time_s"] for result in results],
        dtype=float,
    )
    initialization_times = np.asarray(
        [result["initialization_time_s"] for result in results],
        dtype=float,
    )
    peak_memory = np.asarray(
        [result["peak_python_memory_mb"] for result in results],
        dtype=float,
    )
    current_memory = np.asarray(
        [result["current_python_memory_mb"] for result in results],
        dtype=float,
    )
    completed_iterations = np.asarray(
        [result["completed_iterations"] for result in results],
        dtype=float,
    )
    function_evaluations = np.asarray(
        [result["n_function_evaluations"] for result in results],
        dtype=float,
    )

    fitness_history = _stack_histories(
        results,
        "fitness_history",
        allow_empty=True,
        truncate_to_common_length=True,
    )

    result_entry: dict[str, Any] = {
        "benchmark": benchmark,
        "dim": dimension,
        "algorithm": algorithm,
        "stop_mode": stop_mode,
        "budget_value": budget_value,
        "requested_n_iterations": (
            budget_value if stop_mode == "iterations" else np.nan
        ),
        "requested_max_function_evaluations": (
            budget_value if stop_mode == "function_evaluations" else np.nan
        ),
        "n_runs": expected_runs,
        "n_successful_runs": len(results),
        "mean_completed_iterations": float(np.mean(completed_iterations)),
        "std_completed_iterations": float(_sample_std(completed_iterations)),
        "min_completed_iterations": int(np.min(completed_iterations)),
        "max_completed_iterations": int(np.max(completed_iterations)),
        "mean_function_evaluations": float(np.mean(function_evaluations)),
        "std_function_evaluations": float(_sample_std(function_evaluations)),
        "min_function_evaluations": int(np.min(function_evaluations)),
        "max_function_evaluations": int(np.max(function_evaluations)),
        "best_fitness_history": best_fitness.copy(),
        "final_best_fitness_runs": best_fitness.copy(),
        "mean_best_fitness": float(np.mean(best_fitness)),
        "std_best_fitness": float(_sample_std(best_fitness)),
        "median_best_fitness": float(np.median(best_fitness)),
        "q1_best_fitness": float(np.quantile(best_fitness, 0.25)),
        "q3_best_fitness": float(np.quantile(best_fitness, 0.75)),
        "min_best_fitness": float(np.min(best_fitness)),
        "max_best_fitness": float(np.max(best_fitness)),
        "mean_execution_time_s": float(np.mean(execution_times)),
        "std_execution_time_s": float(_sample_std(execution_times)),
        "median_execution_time_s": float(np.median(execution_times)),
        "mean_initialization_time_s": float(np.mean(initialization_times)),
        "std_initialization_time_s": float(_sample_std(initialization_times)),
    }

    if fitness_history is not None:
        result_entry["iteration_grid"] = np.arange(
            1,
            fitness_history.shape[1] + 1,
            dtype=int,
        )
        result_entry["mean_fitness_history"] = np.mean(
            fitness_history,
            axis=0,
        )
        result_entry["std_fitness_history"] = _sample_std(
            fitness_history,
            axis=0,
        )

    evaluation_end = (
        budget_value
        if stop_mode == "function_evaluations"
        else int(np.min(function_evaluations))
    )
    evaluation_grid = _build_regular_grid(
        start=n_particles,
        end=evaluation_end,
        step=n_particles,
    )
    evaluation_curve = _aggregate_step_histories(
        results,
        x_key="function_evaluations_history",
        y_key="evaluation_fitness_history",
        grid=evaluation_grid,
    )
    if evaluation_curve is not None:
        mean_curve, std_curve = evaluation_curve
        result_entry["function_evaluations_grid"] = evaluation_grid
        result_entry["mean_evaluation_fitness_history"] = mean_curve
        result_entry["std_evaluation_fitness_history"] = std_curve

    if not np.all(np.isnan(peak_memory)):
        result_entry.update(
            {
                "mean_peak_memory_mb": float(np.nanmean(peak_memory)),
                "std_peak_memory_mb": float(
                    _sample_std(peak_memory[~np.isnan(peak_memory)])
                ),
                "median_peak_memory_mb": float(np.nanmedian(peak_memory)),
                # Mantido para compatibilidade com os resultados anteriores.
                "max_peak_memory_mb": float(np.nanmax(peak_memory)),
                "mean_current_memory_mb": float(np.nanmean(current_memory)),
            }
        )

    for history_key, output_name in (
        ("w_history", "w"),
        ("c1_history", "c1"),
        ("c2_history", "c2"),
    ):
        history_matrix = _stack_histories(
            results,
            history_key,
            allow_empty=True,
            truncate_to_common_length=True,
        )
        if history_matrix is not None:
            result_entry[f"mean_{output_name}_history"] = np.mean(
                history_matrix,
                axis=0,
            )
            result_entry[f"std_{output_name}_history"] = _sample_std(
                history_matrix,
                axis=0,
            )

    parameter_axes = [
        result["iteration_function_evaluations_history"] for result in results
    ]
    if parameter_axes and all(len(axis) > 0 for axis in parameter_axes):
        parameter_grid = _build_regular_grid(
            start=max(int(axis[0]) for axis in parameter_axes),
            end=min(int(axis[-1]) for axis in parameter_axes),
            step=n_particles,
        )
        if parameter_grid.size > 0:
            result_entry["parameter_function_evaluations_grid"] = parameter_grid
            for history_key, output_name in (
                ("w_history", "w"),
                ("c1_history", "c1"),
                ("c2_history", "c2"),
            ):
                curve = _aggregate_step_histories(
                    results,
                    x_key="iteration_function_evaluations_history",
                    y_key=history_key,
                    grid=parameter_grid,
                )
                if curve is not None:
                    mean_curve, std_curve = curve
                    result_entry[f"mean_{output_name}_evaluation_history"] = (
                        mean_curve
                    )
                    result_entry[f"std_{output_name}_evaluation_history"] = (
                        std_curve
                    )

    return result_entry


def _atomic_to_pickle(dataframe: pd.DataFrame, path: Path) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    dataframe.to_pickle(temporary_path)
    os.replace(temporary_path, path)


def _atomic_to_csv(dataframe: pd.DataFrame, path: Path) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    dataframe.to_csv(temporary_path, index=False)
    os.replace(temporary_path, path)


def _save_dimension_checkpoint(
    output_dir: Path,
    dimension: int,
    aggregate_entries: list[dict[str, Any]],
    raw_entries: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    aggregate_dataframe = pd.DataFrame(aggregate_entries)
    raw_dataframe = pd.DataFrame(raw_entries)

    aggregate_pickle = output_dir / f"benchmarks_{dimension}dim_results.pkl"
    aggregate_csv = output_dir / (
        f"benchmarks_{dimension}dim_results_summary.csv"
    )
    raw_pickle = output_dir / f"benchmarks_{dimension}dim_runs.pkl"
    raw_csv = output_dir / f"benchmarks_{dimension}dim_runs.csv"
    failures_csv = output_dir / f"benchmarks_{dimension}dim_failures.csv"

    _atomic_to_pickle(aggregate_dataframe, aggregate_pickle)
    _atomic_to_pickle(raw_dataframe, raw_pickle)

    aggregate_array_columns = [
        "best_fitness_history",
        "final_best_fitness_runs",
        "iteration_grid",
        "mean_fitness_history",
        "std_fitness_history",
        "function_evaluations_grid",
        "mean_evaluation_fitness_history",
        "std_evaluation_fitness_history",
        "parameter_function_evaluations_grid",
        "mean_w_history",
        "std_w_history",
        "mean_w_evaluation_history",
        "std_w_evaluation_history",
        "mean_c1_history",
        "std_c1_history",
        "mean_c1_evaluation_history",
        "std_c1_evaluation_history",
        "mean_c2_history",
        "std_c2_history",
        "mean_c2_evaluation_history",
        "std_c2_evaluation_history",
    ]
    aggregate_summary = aggregate_dataframe.drop(
        columns=aggregate_array_columns,
        errors="ignore",
    )
    _atomic_to_csv(aggregate_summary, aggregate_csv)

    raw_object_columns = [
        "best_position",
        "fitness_history",
        "iteration_history",
        "iteration_function_evaluations_history",
        "function_evaluations_history",
        "evaluation_fitness_history",
        "w_history",
        "c1_history",
        "c2_history",
        "extra_history",
        "traceback",
    ]
    raw_summary = raw_dataframe.drop(
        columns=raw_object_columns,
        errors="ignore",
    )
    _atomic_to_csv(raw_summary, raw_csv)

    if not raw_dataframe.empty and "success" in raw_dataframe.columns:
        failures = raw_dataframe.loc[~raw_dataframe["success"]].drop(
            columns=raw_object_columns,
            errors="ignore",
        )
        if not failures.empty:
            _atomic_to_csv(failures, failures_csv)
        elif failures_csv.exists():
            failures_csv.unlink()


def run_experiments(config: ExperimentsConfig) -> None:
    _validate_config(config)

    assert config.algorithms is not None
    assert config.config_hyperparameters is not None
    assert config.config_benchmarks is not None

    # Partitioning prevents one protocol from overwriting or being mixed with
    # the other when the same base output directory is reused.
    output_dir = Path(config.output_dir) / config.stop_mode
    output_dir.mkdir(parents=True, exist_ok=True)

    algorithm_classes = _algorithm_mapping(config.algorithms)
    hyperparameters = _hyperparameter_mapping(config.config_hyperparameters)
    n_particles = config.config_hyperparameters.n_particles
    budget_by_dim = (
        config.n_iterations_by_dim
        if config.stop_mode == "iterations"
        else config.max_function_evaluations_by_dim
    )

    for dimension in config.dim:
        aggregate_entries: list[dict[str, Any]] = []
        raw_entries: list[dict[str, Any]] = []
        budget_value = budget_by_dim[dimension]

        config.config_benchmarks.ndim = dimension
        benchmarks = config.config_benchmarks.build()

        for benchmark_name, benchmark_config in benchmarks.items():
            bounds = benchmark_config["bounds"]
            function = benchmark_config["function"]
            low, high = bounds

            for algorithm_name, algorithm_class in algorithm_classes.items():
                algorithm_hyperparams = hyperparameters.get(algorithm_name)

                if algorithm_hyperparams is None:
                    print(
                        "[WARN] Hiperparâmetros ausentes para "
                        f"{algorithm_name}; algoritmo ignorado."
                    )
                    continue

                params = algorithm_hyperparams.copy()

                params["n_particles"] = n_particles
                params["dim"] = dimension
                params["low"] = low
                params["high"] = high

                print(
                    "[INFO] Iniciando: "
                    f"algoritmo={algorithm_name}, "
                    f"benchmark={benchmark_name}, "
                    f"dimensão={dimension}, "
                    f"critério={config.stop_mode}, "
                    f"orçamento={budget_value}, "
                    f"runs={config.n_runs}."
                )

                tasks = [
                    delayed(_run_single)(
                        algo_class=algorithm_class,
                        algo_hyperparams=params,
                        function=function,
                        stop_mode=config.stop_mode,
                        budget_value=budget_value,
                        run_id=run_id,
                        base_seed=config.seed,
                        measure_memory=config.measure_memory,
                    )
                    for run_id in range(config.n_runs)
                ]

                results = Parallel(
                    n_jobs=config.n_jobs,
                    prefer="processes",
                )(tasks)

                for result in results:
                    result.update(
                        {
                            "benchmark": benchmark_name,
                            "dim": dimension,
                            "algorithm": algorithm_name,
                            "stop_mode": config.stop_mode,
                            "budget_value": budget_value,
                        }
                    )

                raw_entries.extend(results)

                failures = [
                    result for result in results if not result["success"]
                ]
                successful_results = [
                    result for result in results if result["success"]
                ]

                if failures:
                    failed_runs = [
                        {
                            "run_id": item["run_id"],
                            "seed": item["seed"],
                            "error_type": item["error_type"],
                            "error_message": item["error_message"],
                        }
                        for item in failures
                    ]
                    message = (
                        f"{len(failures)} de {config.n_runs} execuções "
                        f"falharam em {algorithm_name} / "
                        f"{benchmark_name} / {dimension}D: "
                        f"{failed_runs}"
                    )

                    if config.fail_on_error:
                        _save_dimension_checkpoint(
                            output_dir=output_dir,
                            dimension=dimension,
                            aggregate_entries=aggregate_entries,
                            raw_entries=raw_entries,
                        )
                        raise RuntimeError(message)

                    print(f"[WARN] {message}")

                if successful_results:
                    aggregate_entries.append(
                        _aggregate_results(
                            successful_results,
                            benchmark=benchmark_name,
                            dimension=dimension,
                            algorithm=algorithm_name,
                            stop_mode=config.stop_mode,
                            budget_value=budget_value,
                            n_particles=n_particles,
                            expected_runs=config.n_runs,
                        )
                    )

                # Checkpoint após cada combinação benchmark × algoritmo.
                _save_dimension_checkpoint(
                    output_dir=output_dir,
                    dimension=dimension,
                    aggregate_entries=aggregate_entries,
                    raw_entries=raw_entries,
                )

                print(
                    "[INFO] Concluído: "
                    f"algoritmo={algorithm_name}, "
                    f"benchmark={benchmark_name}, "
                    f"dimensão={dimension}."
                )

        print(
            f"[INFO] Resultados da dimensão {dimension} salvos em {output_dir}."
        )


if __name__ == "__main__":
    algorithms_config = Algorithms(
        PSO=PSOVectorized,
        PSOLVIW=PSOLVIW,
        PSOTVAC=PSOTVAC,
        APSOVI=APSOVI,
        APSO=APSO,
        UAPSO=UAPSO,
    )

    experiment_config = ExperimentsConfig(
        algorithms=algorithms_config,
        config_hyperparameters=HyperparametersConfig(),
        config_benchmarks=BenchmarksCEC2017Config(),
        output_dir="../results",
        n_runs=25,
        dim=[50, 100],
        stop_mode="function_evaluations",
        n_iterations_by_dim={
            30: 6_000,
            50: 10_000,
            100: 20_000,
        },
        max_function_evaluations_by_dim={
            30: 300_000,
            50: 500_000,
            100: 1_000_000,
        },
        n_jobs=11,
        seed=42,
        measure_memory=True,
        fail_on_error=True,
    )

    run_experiments(experiment_config)
