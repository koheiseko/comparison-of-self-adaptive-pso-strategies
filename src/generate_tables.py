from pathlib import Path

import pandas as pd

from src.statistical_tests import (
    create_ranking_dataframe,
    create_wilcoxon_dataframe,
)

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


def create_performance_table_latex(
    df: pd.DataFrame,
    dim: int,
    output_path: str | Path | None = None,
) -> str:
    """
    Cria uma tabela LaTeX com a mediana e o desvio-padrão
    do melhor fitness para a dimensão selecionada.
    """
    data = df.loc[df["dim"] == dim]

    data = data.set_index(["benchmark", "algorithm"])

    algorithms = [
        algorithm
        for algorithm in ALGORITHM_ORDER
        if algorithm in data.index.get_level_values("algorithm")
    ]

    benchmarks = [
        benchmark
        for benchmark in BENCHMARK_NAMES
        if benchmark in data.index.get_level_values("benchmark")
    ]

    column_spec = "l" + "cc" * len(algorithms)

    algorithm_header = " & ".join(
        rf"\multicolumn{{2}}{{c}}{{\textbf{{"
        rf"{ALGORITHM_NAMES.get(algorithm, algorithm)}"
        rf"}}}}"
        for algorithm in algorithms
    )

    column_rules = " ".join(
        rf"\cmidrule(lr){{{2 + 2 * index}-{3 + 2 * index}}}"
        for index in range(len(algorithms))
    )

    statistic_header = " & ".join(["Mediana & DP"] * len(algorithms))

    lines = [
        r"\begin{table}[!htbp]",
        r"    \centering",
        "    ",
        (
            r"    \caption{Comparativo de desempenho "
            rf"(mediana e desvio padrão), Dim: {dim}.}}"
        ),
        rf"    \label{{tab:resultados_dim{dim}}}",
        r"    \resizebox{\linewidth}{!}{%",
        r"        \setlength{\tabcolsep}{3pt}",
        rf"        \begin{{tabular}}{{{column_spec}}}",
        r"            \toprule",
        rf"            & {algorithm_header} \\",
        rf"            {column_rules}",
        rf"            \textbf{{Função}} & {statistic_header} \\",
        r"            \midrule",
    ]

    for benchmark_index, benchmark in enumerate(benchmarks):
        function_number = benchmark.split("_", 1)[0][1:]
        values = []

        for algorithm in algorithms:
            row = data.loc[(benchmark, algorithm)]

            values.extend(
                [
                    f"{row['median_best_fitness']:.2E}",
                    f"{row['std_best_fitness']:.2E}",
                ]
            )

        suffix = (
            r" \\ \addlinespace"
            if benchmark_index < len(benchmarks) - 1
            else r" \\"
        )

        lines.append(
            rf"            $f_{{{function_number}}}$ & "
            + " & ".join(values)
            + suffix
        )

    lines.extend(
        [
            r"            \bottomrule",
            r"        \end{tabular}%",
            r"    }",
            r"\end{table}",
        ]
    )

    latex = "\n".join(lines)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(latex + "\n", encoding="utf-8")


def create_time_memory_table_latex(
    df: pd.DataFrame,
    output_path: str | Path | None = None,
) -> str:
    """
    Cria a tabela LaTeX de tempo e pico de memória por dimensão.
    """
    dimensions = sorted(df["dim"].unique())

    algorithms = [
        algorithm
        for algorithm in ALGORITHM_ORDER
        if algorithm in df["algorithm"].unique()
    ]

    summary = df.groupby(["algorithm", "dim"])[
        [
            "median_execution_time_s",
            "median_peak_memory_mb",
        ]
    ].mean()

    n_dimensions = len(dimensions)

    column_spec = "l" + "c" * (2 * n_dimensions - 1) + r"@{\hspace{2pt}}c"

    lines = [
        r"\begin{table*}[!t]",
        r"    \centering",
        (
            r"    \caption{Tempo mediano e pico mediano "
            r"de memória por dimensão.}"
        ),
        r"    \label{tab:tempo_memoria_dimensao}",
        "",
        r"    \scriptsize",
        r"    \renewcommand{\arraystretch}{0.85}",
        r"    \setlength{\tabcolsep}{4pt}",
        "",
        r"    \begin{tabular*}{\textwidth}{",
        r"        @{\extracolsep{\fill}}",
        f"        {column_spec}",
        r"        @{}",
        r"    }",
        r"        \toprule",
        (
            rf"        & \multicolumn{{{n_dimensions}}}{{c}}"
            r"{\textbf{Tempo mediano (s)}}"
        ),
        (
            rf"        & \multicolumn{{{n_dimensions}}}{{c}}"
            r"{\textbf{Pico de memória (MB)}} \\"
        ),
        "",
        rf"        \cmidrule(lr){{2-{n_dimensions + 1}}}",
        (
            rf"        \cmidrule(lr)"
            rf"{{{n_dimensions + 2}-{2 * n_dimensions + 1}}}"
        ),
        "",
        r"        \textbf{Algoritmo}",
    ]

    # Cabeçalhos das dimensões para o tempo.
    for dim in dimensions:
        lines.append(rf"        & \textbf{{{dim}D}}")

    # Cabeçalhos das dimensões para a memória.
    for index, dim in enumerate(dimensions):
        suffix = r" \\" if index == len(dimensions) - 1 else ""
        lines.append(rf"        & \textbf{{{dim}D}}{suffix}")

    lines.extend(
        [
            "",
            r"        \midrule",
        ]
    )

    label_width = max(
        len(ALGORITHM_NAMES.get(algorithm, algorithm))
        for algorithm in algorithms
    )

    for algorithm in algorithms:
        values = []

        # Tempos: uma casa decimal.
        for dim in dimensions:
            time = summary.loc[
                (algorithm, dim),
                "median_execution_time_s",
            ]
            values.append(f"{time:.1f}".replace(".", ","))

        # Memória: duas casas decimais.
        for dim in dimensions:
            memory = summary.loc[
                (algorithm, dim),
                "median_peak_memory_mb",
            ]
            values.append(f"{memory:.2f}".replace(".", ","))

        label = ALGORITHM_NAMES.get(algorithm, algorithm)

        lines.append(
            f"        {label:<{label_width}} & " + " & ".join(values) + r" \\"
        )

    lines.extend(
        [
            r"        \bottomrule",
            r"    \end{tabular*}",
            r"\end{table*}",
        ]
    )

    latex = "\n".join(lines)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(latex + "\n", encoding="utf-8")


def _format_latex_number(value: float) -> str:
    """Formata valores-p conforme o padrão das tabelas do artigo."""
    if value < 0.01:
        mantissa, exponent = f"{value:.2e}".split("e")
        mantissa = mantissa.replace(".", "{,}")
        return rf"${mantissa} \times 10^{{{int(exponent)}}}$"

    return f"{value:.3f}".replace(".", "{,}")


def create_wilcoxon_table_latex(
    df: pd.DataFrame, output_path: str | Path
) -> None:
    """Salva a tabela de Wilcoxon–Holm."""
    comparisons = create_wilcoxon_dataframe(df)

    lines = [
        r"\begin{table}[!htbp]",
        r"    \centering",
        r"    \caption{Comparações pareadas pelo teste de Wilcoxon com correção de Holm.}",
        r"    \label{tab:wilcoxon_holm}",
        "    ",
        r"    \resizebox{\textwidth}{!}{%",
        r"        \begin{tabular}{lcccccc}",
        r"            \toprule",
        r"            \textbf{Comparação}",
        r"            & \textbf{$W$}",
        r"            & \textbf{$p$}",
        r"            & \textbf{$p_{\mathrm{Holm}}$}",
        r"            & \textbf{$|r_{\mathrm{rb}}|$}",
        r"            & \textbf{Favorece}",
        "            & \\textbf{Significativo} \\\\",
        r"            \midrule",
        "            ",
    ]

    for index, row in comparisons.iterrows():
        comparison = row["Comparação"].replace(" × ", r" $\times$ ")
        lines.extend(
            [
                f"            {comparison}",
                f"            & {row['W']:g}",
                f"            & {_format_latex_number(row['p'])}",
                f"            & {_format_latex_number(row['p_Holm'])}",
                f"            & {row['|r_rb|']:.3f}".replace(".", "{,}"),
                f"            & {row['Favorece']}",
                (
                    "            & \\textbf{Sim} \\\\"
                    if row["Significativo"] == "Sim"
                    else "            & Não \\\\"
                ),
            ]
        )
        if index < len(comparisons) - 1:
            lines.append("            ")

    lines.extend(
        [
            "            ",
            r"            \bottomrule",
            r"        \end{tabular}%",
            r"    }",
            "    ",
            r"    \vspace{0.4em}",
            "    ",
            r"    \begin{minipage}{\textwidth}",
            r"        \footnotesize",
            r"        \textit{Nota:} $W$ representa a estatística do teste de Wilcoxon;",
            r"        $p$ é o valor não corrigido; $p_{\mathrm{Holm}}$ é o valor",
            r"        ajustado para comparações múltiplas; e",
            r"        $|r_{\mathrm{rb}}|$ representa o módulo do tamanho de efeito",
            r"        rank-biserial. A decisão de significância considera",
            r"        $p_{\mathrm{Holm}} < 0{,}05$. A coluna ``Favorece'' indica",
            r"        apenas a direção do efeito.",
            r"    \end{minipage}",
            r"\end{table}",
        ]
    )

    latex = "\n".join(lines)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(latex + "\n", encoding="utf-8")


def create_ranking_table_latex(
    df: pd.DataFrame, output_path: str | Path
) -> None:
    """Salva a tabela de ranks e frequências."""
    ranking = create_ranking_dataframe(df)

    best_rank = ranking["Rank médio"].min()
    most_first = ranking["1º lugar"].max()
    most_top_two = ranking["Top-2"].max()

    lines = [
        r"\begin{table}[!htbp]",
        r"    \centering",
        r"    \caption{Rank médio e frequência de posições}",
        r"    \label{tab:ranking_algoritmos}",
        r"    \small",
        r"    \begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}llrrr@{}}",
        r"        \toprule",
        r"        \textbf{Algoritmo}",
        r"        & \textbf{Categoria}",
        r"        & \textbf{Rank médio}",
        r"        & \textbf{1º lugar}",
        "        & \\textbf{Top-2} \\\\",
        r"        \midrule",
        "",
    ]

    for index, row in ranking.iterrows():
        mean_rank = f"{row['Rank médio']:.3f}".replace(".", ",")
        first_places = str(int(row["1º lugar"]))
        top_two = str(int(row["Top-2"]))

        if row["Rank médio"] == best_rank:
            mean_rank = rf"\textbf{{{mean_rank}}}"
        if row["1º lugar"] == most_first:
            first_places = rf"\textbf{{{first_places}}}"
        if row["Top-2"] == most_top_two:
            top_two = rf"\textbf{{{top_two}}}"

        lines.extend(
            [
                f"        {row['Algoritmo']}",
                f"        & {row['Categoria']}",
                f"        & {mean_rank}",
                f"        & {first_places}",
                f"        & {top_two} " + r"\\",
            ]
        )
        if index < len(ranking) - 1:
            lines.append("")

    lines.extend(
        [
            "",
            r"        \bottomrule",
            r"    \end{tabular*}",
            r"\end{table}",
        ]
    )

    latex = "\n".join(lines)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(latex + "\n", encoding="utf-8")
