from pathlib import Path

from imaging_ai_atlas.story.labels import load_topic_labels, topic_label


def test_load_topic_labels_only_keeps_coherent_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text(
        "topic_id,size,keywords,coherent,suggested_label\n"
        "0,100,mri scan,yes,Brain MRI\n"
        "1,50,x y,no,\n"
        "2,40,ct scan, Yes ,CT imaging\n",
        encoding="utf-8",
    )

    labels = load_topic_labels(csv_path)

    assert labels == {0: "Brain MRI", 2: "CT imaging"}


def test_topic_label_uses_the_reviewed_label_or_falls_back_to_the_topic_id() -> None:
    labels = {3: "Retinal imaging"}

    assert topic_label(3, labels) == "Retinal imaging"
    assert topic_label(7, labels) == "Topic 7"
