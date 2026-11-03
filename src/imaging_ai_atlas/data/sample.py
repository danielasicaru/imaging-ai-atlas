import pandas as pd


def build_labeling_sample(
    df: pd.DataFrame, in_filter: pd.Series, n_in: int, n_out: int, seed: int
) -> pd.DataFrame:
    """Stratified sample for the manual precision/recall check: `n_in` rows the filter
    accepted, `n_out` rows it rejected (the excluded pool, for recall). An empty `label`
    column is added for the human reviewer to fill in (e.g. "relevant" / "not relevant").
    Each stratum is capped at however many rows actually exist in it.
    """
    in_pool = df.loc[in_filter.to_numpy()]
    out_pool = df.loc[~in_filter.to_numpy()]

    in_sample = in_pool.sample(n=min(n_in, len(in_pool)), random_state=seed)
    out_sample = out_pool.sample(n=min(n_out, len(out_pool)), random_state=seed)

    sample = pd.concat([in_sample, out_sample], ignore_index=True)
    sample["in_filter"] = [True] * len(in_sample) + [False] * len(out_sample)
    sample["label"] = ""
    return sample
