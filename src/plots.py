from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ALGORITHM_ORDER = [
    "PSO",
    "PSOLVIW",
    "PSOTVAC",
    "APSO",
    "APSOVI",
    "UAPSO",
]

ALGORITHM_NAMES = {
    "PSO": "PSO",
    "PSOLVIW": "PSO-LVIW",
    "PSOTVAC": "PSO-TVAC",
    "APSO": "APSO",
    "APSOVI": "APSO-VI",
    "UAPSO": "UAPSO-A",
}

BENCHMARK_NAMES = {
    "F1_BentCigar": "F1 — Bent Cigar",
    "F3_Zakharov": "F3 — Zakharov",
    "F5_Rastrigin_SR": "F5 — Rastrigin S+R",
    "F6_Scaffer": "F6 — Expanded Scaffer F6",
    "F7_LunacekBiRastr": "F7 — Lunacek bi-Rastrigin",
    "F9_Levy": "F9 — Levy",
    "F10_Schwefel": "F10 — Schwefel",
    "F11_Hybrid1": "F11 — Hybrid Function 1",
    "F21_Composition1": "F21 — Composition Function 1",
}


LINE_STYLES = [
    "-",
    "--",
    "-.",
    ":",
    (0, (3, 1, 1, 1)),
    (0, (5, 1)),
]

ALGORITHM_LINE_STYLES = {
    algorithm: LINE_STYLES[index % len(LINE_STYLES)]
    for index, algorithm in enumerate(ALGORITHM_ORDER)
}


def plot_single_convergence(
    df: pd.DataFrame,
    benchmark: str,
    dim: int,
    output_path: str | Path | None = None,
    *,
    statistic: str = "median",
    show_band: bool = True,
    log_scale: bool = False,
    epsilon: float = 1e-12,
    title: str | None = None,
    show: bool = False,
    dpi: int = 300,
    ax=None,
    legend: bool = True,
):
    """
    Plota a convergência de um benchmark e dimensão.

    Se `ax` for informado, o gráfico será desenhado nesse subplot.
    Caso contrário, uma nova figura será criada.
    """
    if statistic not in {"mean", "median"}:
        raise ValueError("statistic deve ser 'mean' ou 'median'.")

    data = df.loc[(df["benchmark"] == benchmark) & (df["dim"] == dim)]

    if data.empty:
        raise ValueError(f"Sem dados para {benchmark} ({dim}D).")

    created_figure = ax is None

    if created_figure:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.figure

    present_algorithms = data["algorithm"].unique().tolist()

    algorithms = [
        algorithm
        for algorithm in ALGORITHM_ORDER
        if algorithm in present_algorithms
    ]

    algorithms += sorted(set(present_algorithms) - set(algorithms))

    for algorithm_index, algorithm in enumerate(algorithms):
        algorithm_data = data.loc[data["algorithm"] == algorithm]

        histories = []

        for evaluations, fitness in zip(
            algorithm_data["function_evaluations_history"],
            algorithm_data["evaluation_fitness_history"],
        ):
            x = np.asarray(evaluations, dtype=float)
            y = np.asarray(fitness, dtype=float)

            if x.ndim != 1 or y.ndim != 1:
                raise ValueError(
                    f"Histórico inválido para o algoritmo {algorithm}."
                )

            if x.size == 0 or x.shape != y.shape:
                raise ValueError(
                    f"Histórico inválido para o algoritmo {algorithm}."
                )

            histories.append((x, y))

        if not histories:
            continue

        # Região coberta por todas as execuções.
        start = max(x[0] for x, _ in histories)
        end = min(x[-1] for x, _ in histories)

        if end < start:
            continue

        # Grade comum de avaliações da função.
        grid = np.unique(
            np.concatenate([x[(x >= start) & (x <= end)] for x, _ in histories])
        )

        if grid.size == 0:
            continue

        # Mantém o último fitness conhecido em cada avaliação.
        matrix = np.vstack(
            [
                y[np.searchsorted(x, grid, side="right") - 1]
                for x, y in histories
            ]
        )

        if statistic == "median":
            center = np.median(matrix, axis=0)
            lower = np.quantile(matrix, 0.25, axis=0)
            upper = np.quantile(matrix, 0.75, axis=0)
        else:
            center = np.mean(matrix, axis=0)

            std = np.std(
                matrix,
                axis=0,
                ddof=1 if len(histories) > 1 else 0,
            )

            lower = center - std
            upper = center + std

        if log_scale:
            center = np.maximum(center, epsilon)
            lower = np.maximum(lower, epsilon)
            upper = np.maximum(upper, epsilon)

        label = ALGORITHM_NAMES.get(algorithm, algorithm)

        linestyle = ALGORITHM_LINE_STYLES.get(
            algorithm,
            LINE_STYLES[algorithm_index % len(LINE_STYLES)],
        )

        line = ax.step(
            grid,
            center,
            where="post",
            linewidth=1.7,
            linestyle=linestyle,
            label=label,
        )[0]

        if show_band:
            ax.fill_between(
                grid,
                lower,
                upper,
                step="post",
                color=line.get_color(),
                alpha=0.15,
                linewidth=0,
            )

    if log_scale:
        ax.set_yscale("log")

    ax.set_xlabel("Avaliações da função (FEs)")
    ax.set_ylabel("Erro (escala log)" if log_scale else "Erro")
    ax.set_title(
        title or f"{BENCHMARK_NAMES.get(benchmark, benchmark)}, Dim: {dim}"
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.2, linewidth=0.6)

    if legend:
        ax.legend()

    if created_figure:
        fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(
            output_path,
            dpi=dpi,
            bbox_inches="tight",
        )

    if show:
        plt.show()

    return fig, ax


def plot_all_convergences(
    df: pd.DataFrame,
    output_path: str | Path | None = None,
    *,
    ncols: int = 3,
    statistic: str = "median",
    show_band: bool = True,
    log_scale: bool = False,
    epsilon: float = 1e-12,
    show: bool = False,
    dpi: int = 300,
):
    """
    Cria uma única figura com um subplot para cada combinação
    disponível de benchmark e dimensão.
    """
    if ncols < 1:
        raise ValueError("ncols deve ser maior ou igual a 1.")

    available_pairs = set(
        df[["benchmark", "dim"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )

    present_benchmarks = df["benchmark"].unique().tolist()

    benchmarks = [
        benchmark
        for benchmark in BENCHMARK_NAMES
        if benchmark in present_benchmarks
    ]

    benchmarks += sorted(set(present_benchmarks) - set(benchmarks))

    dimensions = sorted(df["dim"].unique())

    pairs = [
        (benchmark, dim)
        for benchmark in benchmarks
        for dim in dimensions
        if (benchmark, dim) in available_pairs
    ]

    if not pairs:
        raise ValueError(
            "Nenhuma combinação de benchmark e dimensão foi encontrada."
        )

    ncols = min(ncols, len(pairs))
    nrows = int(np.ceil(len(pairs) / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(6 * ncols, 4 * nrows),
        squeeze=False,
    )

    flat_axes = axes.ravel()

    for ax, (benchmark, dim) in zip(flat_axes, pairs):
        plot_single_convergence(
            df=df,
            benchmark=benchmark,
            dim=dim,
            statistic=statistic,
            show_band=show_band,
            log_scale=log_scale,
            epsilon=epsilon,
            ax=ax,
            legend=False,
        )

    # Remove subplots não utilizados.
    for ax in flat_axes[len(pairs) :]:
        ax.remove()

    # Cria uma única legenda para toda a figura.
    handles_by_label = {}

    for ax in flat_axes[: len(pairs)]:
        handles, labels = ax.get_legend_handles_labels()

        for handle, label in zip(handles, labels):
            handles_by_label.setdefault(label, handle)

    if handles_by_label:
        fig.legend(
            handles_by_label.values(),
            handles_by_label.keys(),
            loc="upper center",
            ncol=min(len(handles_by_label), len(ALGORITHM_ORDER)),
            frameon=True,
            bbox_to_anchor=(0.5, 1.0),
        )

    fig.tight_layout(rect=(0, 0, 1, 0.94))

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(
            output_path,
            dpi=dpi,
            bbox_inches="tight",
        )

    if show:
        plt.show()

    return fig, axes
