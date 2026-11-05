from pathlib import Path

from imaging_ai_atlas.story.labels import load_topic_labels


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
