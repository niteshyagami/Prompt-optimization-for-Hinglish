import argparse
import os
import pandas as pd


def ensure_questions(df, dataset_path):
    if "question" in df.columns:
        return df

    if "sample_id" not in df.columns:
        raise ValueError("Missing 'question' and 'sample_id' columns.")

    base_df = pd.read_csv(dataset_path)
    if "question" not in base_df.columns:
        raise ValueError("Dataset file missing required 'question' column.")

    sample_ids = pd.to_numeric(df["sample_id"], errors="coerce")
    if sample_ids.isna().any():
        bad_ids = df.loc[sample_ids.isna(), "sample_id"].head(5).tolist()
        raise ValueError(f"Non-numeric sample_id values found: {bad_ids}")
    if (sample_ids < 0).any():
        raise ValueError("sample_id contains negative values.")
    sample_ids = sample_ids.astype(int)
    if sample_ids.max() >= len(base_df):
        raise ValueError("sample_id contains values outside the dataset range.")

    questions = base_df["question"].reset_index(drop=True)
    df["question"] = questions.iloc[sample_ids.tolist()].values
    return df


def stratified_sample(df, total_samples, seed):
    rng = pd.Series(range(len(df))).sample(frac=1, random_state=seed).index
    df = df.loc[rng].reset_index(drop=True)

    group_cols = ["technique"]
    if "model" in df.columns:
        group_cols = ["model", "technique"]

    groups = list(df.groupby(group_cols))
    num_groups = len(groups)
    per_group = max(1, total_samples // num_groups)

    samples = []
    remaining = total_samples
    for _, group in groups:
        take = min(per_group, len(group))
        samples.append(group.sample(n=take, random_state=seed))
        remaining -= take

    if remaining > 0:
        leftover = df.drop(pd.concat(samples).index, errors="ignore")
        if len(leftover) > 0:
            extra = leftover.sample(n=min(remaining, len(leftover)), random_state=seed)
            samples.append(extra)

    return pd.concat(samples, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="evaluated_results.csv")
    parser.add_argument("--output", default="human_eval_template.csv")
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", default="final_hinglish_qa_dataset.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df = ensure_questions(df, args.dataset)
    sampled = stratified_sample(df, args.samples, args.seed)

    cols = [
        "sample_id",
        "model",
        "technique",
        "question",
        "generated_answer",
        "ground_truth",
    ]
    for col in cols:
        if col not in sampled.columns:
            sampled[col] = ""

    template = sampled[cols].copy()
    template["correctness_1_10"] = ""
    template["relevance_1_10"] = ""
    template["hinglish_naturalness_1_10"] = ""
    template["rater_id"] = ""
    template["notes"] = ""

    template.to_csv(args.output, index=False)
    print(f"Wrote {args.output} with {len(template)} rows.")


if __name__ == "__main__":
    main()
