from pathlib import Path

import pandas as pd


def load_topic_labels(labeling_sample_path: Path) -> dict[int, str]:
    """Human-reviewed topic_id -> short label lookup, from the topic labeling CSV written
    by `atlas register-topic`. Only rows marked coherent get a label; an incoherent topic
    has no trustworthy name to show, so callers should fall back to a generic "Topic N"
    for it rather than display one here.
    """
    df = pd.read_csv(labeling_sample_path)
    coherent = df[df["coherent"].astype(str).str.strip().str.lower() == "yes"]
    return dict(zip(coherent["topic_id"], coherent["suggested_label"], strict=True))
