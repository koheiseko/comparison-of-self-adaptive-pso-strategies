from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

ALGORITHM_ORDER = ["PSO", "PSOLVIW", "PSOTVAC", "APSO", "APSOVI", "UAPSO"]

ALGORITHM_NAMES = {
    "PSO": "PSO",
    "PSOLVIW": "PSO-LVIW",
    "PSOTVAC": "PSO-TVAC",
    "APSO": "APSO",
    "APSOVI": "APSO-VI",
    "UAPSO": "UAPSO-A",
}

ALGORITHM_CATEGORIES = {
    "PSO": "Não adaptativo",
    "PSOLVIW": "Variante no tempo",
    "PSOTVAC": "Variante no tempo",
    "APSO": "Verdadeiramente adaptativo",
    "APSOVI": "Verdadeiramente adaptativo",
    "UAPSO": "Verdadeiramente adaptativo",
}


def _build_performance_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Organiza os resultados em blocos função–dimensão."""
    matrix = df.pivot(
        index=["benchmark", "dim"],
        columns="algorithm",
        values="median_best_fitness",
    ).reindex(columns=ALGORITHM_ORDER)

    if matrix.isna().any().any():
        raise ValueError("Cada bloco deve possuir todos os algoritmos.")

    return matrix


def _calculate_ranking(matrix: pd.DataFrame) -> pd.DataFrame:
    """Calcula rank médio, primeiros lugares e ocorrências no top-2."""
    ranks = matrix.rank(axis=1, method="average")

    return pd.DataFrame(
        {
            "mean_rank": ranks.mean(),
            "first_places": ranks.eq(1).sum(),
            "top_two": ranks.le(2).sum(),
        }
    ).sort_values("mean_rank")


def _rank_biserial(differences: np.ndarray) -> float:
    """Calcula o efeito rank-biserial para uma comparação pareada."""
    differences = differences[differences != 0]
    if len(differences) == 0:
        return 0.0

    ranks = stats.rankdata(np.abs(differences), method="average")
    better = ranks[differences < 0].sum()
    worse = ranks[differences > 0].sum()
    return float((better - worse) / ranks.sum())


def _holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Aplica a correção de Holm aos valores-p."""
    order = np.argsort(p_values, kind="stable")
    ordered = p_values[order]
    adjusted_ordered = np.maximum.accumulate(
        (len(ordered) - np.arange(len(ordered))) * ordered
    ).clip(max=1)

    adjusted = np.empty(len(p_values))
    adjusted[order] = adjusted_ordered
    return adjusted


def _calculate_wilcoxon_holm(matrix: pd.DataFrame) -> pd.DataFrame:
    """Executa Wilcoxon para todos os pares e aplica Holm."""
    rows = []

    for alg1, alg2 in combinations(ALGORITHM_ORDER, 2):
        differences = (matrix[alg1] - matrix[alg2]).to_numpy()

        if np.all(differences == 0):
            statistic, p_value = 0.0, 1.0
        else:
            test = stats.wilcoxon(differences, method="auto")
            statistic, p_value = float(test.statistic), float(test.pvalue)

        rows.append(
            {
                "alg1": alg1,
                "alg2": alg2,
                "statistic": statistic,
                "p": p_value,
                "effect": _rank_biserial(differences),
            }
        )

    comparisons = pd.DataFrame(rows)
    comparisons["holm_p"] = _holm_adjust(comparisons["p"].to_numpy())
    return comparisons.sort_values("holm_p", kind="stable").reset_index(
        drop=True
    )

def _friedman_permutation_pvalue(
    ranks: np.ndarray,
    observed_statistic: float,
    n_resamples: int,
    random_state: int | None,
) -> float:
    """Estima o valor-p por permutações internas a cada bloco."""
    if n_resamples < 1:
        raise ValueError("n_resamples deve ser maior que zero.")

    rng = np.random.default_rng(random_state)
    n_blocks, n_algorithms = ranks.shape
    denominator = n_blocks * n_algorithms * (n_algorithms + 1)
    observed_without_tie_correction = (
        12.0 / denominator * np.sum(ranks.sum(axis=0) ** 2)
        - 3.0 * n_blocks * (n_algorithms + 1)
    )
    tie_correction = (
        observed_statistic / observed_without_tie_correction
        if observed_without_tie_correction > 0
        else 1.0
    )
    extreme = 0

    for start in range(0, n_resamples, 10_000):
        batch_size = min(10_000, n_resamples - start)
        rank_sums = np.zeros((batch_size, n_algorithms))

        for row in ranks:
            permutations = np.argsort(
                rng.random((batch_size, n_algorithms)),
                axis=1,
            )
            rank_sums += row[permutations]

        statistics = (
            12.0 / denominator * np.sum(rank_sums**2, axis=1)
            - 3.0 * n_blocks * (n_algorithms + 1)
        )
        statistics *= tie_correction
        extreme += np.count_nonzero(
            statistics >= observed_statistic - 1e-12
        )

    return float((extreme + 1) / (n_resamples + 1))


def friedman_test(
    df: pd.DataFrame,
    n_resamples: int = 100_000,
    random_state: int | None = 0,
) -> tuple[float, float]:
    """Executa o teste de Friedman com valor-p por permutação."""
    matrix = _build_performance_matrix(df)
    values = matrix.to_numpy(dtype=float)
    ranks = np.vstack([
        stats.rankdata(row, method="average")
        for row in values
    ])

    result = stats.friedmanchisquare(*values.T)
    statistic = float(result.statistic)
    p_value = _friedman_permutation_pvalue(
        ranks,
        statistic,
        n_resamples,
        random_state,
    )
    return statistic, p_value

def create_wilcoxon_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Retorna a tabela de Wilcoxon–Holm como DataFrame."""
    comparisons = _calculate_wilcoxon_holm(_build_performance_matrix(df))

    favours = []
    for row in comparisons.itertuples():
        if row.effect > 0:
            favours.append(ALGORITHM_NAMES[row.alg1])
        elif row.effect < 0:
            favours.append(ALGORITHM_NAMES[row.alg2])
        else:
            favours.append("Empate")

    return pd.DataFrame(
        {
            "Comparação": [
                f"{ALGORITHM_NAMES[row.alg1]} × {ALGORITHM_NAMES[row.alg2]}"
                for row in comparisons.itertuples()
            ],
            "W": comparisons["statistic"],
            "p": comparisons["p"],
            "p_Holm": comparisons["holm_p"],
            "|r_rb|": comparisons["effect"].abs(),
            "Favorece": favours,
            "Significativo": comparisons["holm_p"]
            .lt(0.05)
            .map({True: "Sim", False: "Não"}),
        }
    )


def create_ranking_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Retorna a tabela de ranks e frequências como DataFrame."""
    ranking = _calculate_ranking(_build_performance_matrix(df))

    return pd.DataFrame(
        {
            "Algoritmo": [
                ALGORITHM_NAMES[algorithm] for algorithm in ranking.index
            ],
            "Categoria": [
                ALGORITHM_CATEGORIES[algorithm] for algorithm in ranking.index
            ],
            "Rank médio": ranking["mean_rank"].to_numpy(),
            "1º lugar": ranking["first_places"].to_numpy(dtype=int),
            "Top-2": ranking["top_two"].to_numpy(dtype=int),
        }
    )
