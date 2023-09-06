import json
import re
import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import groupby
from ydata_profiling import ProfileReport
import sweetviz as sv

colors = [["full", "green"], ["light", "orange"], ["extreme", "red"]]


def load_data(options, dataset: str, data_filter: str) -> pd.DataFrame:
    path = f"../Data/Training_Data/{dataset}_data_{data_filter}.parquet"
    return pd.read_parquet(path, engine="auto")


def format(df: pd.DataFrame) -> pd.DataFrame:
    # NaNs are represented as -100000.0
    df = df.replace(-100000.0, np.nan)

    return df


if __name__ == "__main__":
    with open("./options.json", "r") as file:
        options = json.load(file)

    point = "PEEP"
    datasets = ["uka", "MIMICIV", "eICU"]

    for dataset in datasets:
        fig, (axes) = plt.subplots(1, len(colors),
                                   figsize=(10, 5), sharey=True, sharex=True)
        for (data_filter, color), axis in zip(colors, axes):
            data = load_data(options, dataset, data_filter)
            data = format(data)

            column_groups = list(data.columns)
            column_groups.sort()
            column_groups = [list(i) for j, i in groupby(
                column_groups, lambda a: a.rstrip(string.digits))]

            df = pd.DataFrame()
            for cols in column_groups:
                group = cols[0].rstrip(string.digits)
                df[group] = data[cols].mean(axis=1).round()

            # Save df to csv file
            df.to_csv(f"./stats/{dataset}_{data_filter}.csv")

            means = df[point]

            report = sv.analyze(df, target_feat="ARDS")
            report.show_html(f"./stats/{dataset}_{data_filter}_sweetviz.html")

            axis.hist(means, bins=10, color=color, label=data_filter)
            axis.set_title(data_filter)

        plt.suptitle(f"Mean {point} values (per patient): {dataset}")
        plt.savefig(f"./stats/{dataset}_{point}.png")
