from pathlib import Path

import pandas as pd

from scripts.run_experiments import ExperimentsConfig


def process(
    dim: list[int],
    output_dir: str = "results",
) -> None:
    base_path = Path(output_dir)
    dataframes = []

    dimensions = dim

    for d in dimensions:
        file_path = base_path / f"benchmarks_{d}dim_results.pkl"

        if file_path.exists():
            df = pd.read_pickle(file_path)
            dataframes.append(df)
        else:
            print(f"[AVISO] Arquivo não encontrado: {file_path.name}.")

    if not dataframes:
        print(
            "\nErro: Nenhum dado foi encontrado para processar. Verifique a pasta de resultados."
        )
        return

    df_results = pd.concat(dataframes, ignore_index=True)

    path_csv_out = base_path / "benchmarks_results.csv"
    path_pkl_out = base_path / "benchmarks_results.pkl"

    df_results.to_csv(path_csv_out, index=False)
    df_results.to_pickle(path_pkl_out)


if __name__ == "__main__":
    experiment_config = ExperimentsConfig()

    process(
        dim=experiment_config.dim,
    )
