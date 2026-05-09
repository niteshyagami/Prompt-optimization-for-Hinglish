import argparse
import os
import re
import pandas as pd
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def sanitize(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name)


def run_anova_and_tukey(df, metric, out_prefix):
    groups = [g[metric].dropna().values for _, g in df.groupby("technique")]
    if len(groups) < 2:
        raise ValueError("Not enough groups for ANOVA.")

    f_stat, p_val = f_oneway(*groups)
    summary_txt = f"Metric: {metric}\nF-statistic: {f_stat:.4f}\nP-value: {p_val:.6f}\n"

    with open(out_prefix + f"_anova_{metric}.txt", "w", encoding="utf-8") as f:
        f.write(summary_txt)

    tukey = pairwise_tukeyhsd(endog=df[metric], groups=df["technique"], alpha=0.05)
    tukey_df = pd.DataFrame(tukey.summary().data[1:], columns=tukey.summary().data[0])
    tukey_df.to_csv(out_prefix + f"_tukey_{metric}.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="evaluated_results.csv")
    parser.add_argument("--metrics", nargs="+", default=["quality_score"])
    parser.add_argument("--out_dir", default=".")
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    if "model" in df.columns:
        models = df["model"].fillna("unknown").unique().tolist()
        for model in models:
            sub = df[df["model"].fillna("unknown") == model]
            prefix = os.path.join(args.out_dir, f"stats_{sanitize(model)}")
            for metric in args.metrics:
                run_anova_and_tukey(sub, metric, prefix)
            print(f"Saved stats for model: {model}")
    else:
        prefix = os.path.join(args.out_dir, "stats")
        for metric in args.metrics:
            run_anova_and_tukey(df, metric, prefix)
        print("Saved stats.")


if __name__ == "__main__":
    main()
