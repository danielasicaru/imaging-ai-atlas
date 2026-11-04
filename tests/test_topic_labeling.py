from imaging_ai_atlas.topics.labeling import build_topic_label_sample


class FakeTopicModel:
    def __init__(self, topic_words: dict[int, list[tuple[str, float]]]) -> None:
        self._topic_words = topic_words

    def get_topic(self, topic_id: int) -> list[tuple[str, float]] | bool:
        return self._topic_words.get(topic_id, False)


def test_build_topic_label_sample_picks_largest_topics_with_keywords_and_examples() -> None:
    topics = [0, 0, 0, 1, 1, -1, -1, -1, -1]
    texts = [f"doc{i}" for i in range(len(topics))]
    model = FakeTopicModel({0: [("mri", 0.9), ("scan", 0.8)], 1: [("ct", 0.7)]})

    sample = build_topic_label_sample(model, topics, texts, n_topics=2, seed=0)

    assert list(sample["topic_id"]) == [0, 1]
    assert sample.iloc[0]["size"] == 3
    assert "mri" in sample.iloc[0]["keywords"]
    assert sample.iloc[0]["coherent"] == ""
    assert sample.iloc[0]["suggested_label"] == ""


def test_build_topic_label_sample_excludes_outlier_topic() -> None:
    topics = [-1, -1, 0, 0]
    texts = ["a", "b", "c", "d"]
    model = FakeTopicModel({0: [("x", 1.0)]})

    sample = build_topic_label_sample(model, topics, texts, n_topics=5, seed=0)

    assert -1 not in list(sample["topic_id"])


def test_build_topic_label_sample_handles_mapping_style_get_topic() -> None:
    # Some BERTopic versions return a word->score mapping from get_topic instead of a
    # list of tuples - both must normalize to the same keyword string.
    topics = [0, 0]
    texts = ["a", "b"]
    model = FakeTopicModel({})
    model.get_topic = lambda topic_id: {"mri": 0.9, "scan": 0.8} if topic_id == 0 else False  # type: ignore[method-assign]

    sample = build_topic_label_sample(model, topics, texts, n_topics=1, seed=0)

    assert sample.iloc[0]["keywords"] == "mri, scan"


def test_build_topic_label_sample_caps_at_n_topics() -> None:
    topics = [0, 0, 1, 1, 2, 2]
    texts = ["a", "b", "c", "d", "e", "f"]
    model = FakeTopicModel({0: [("x", 1.0)], 1: [("y", 1.0)], 2: [("z", 1.0)]})

    sample = build_topic_label_sample(model, topics, texts, n_topics=2, seed=0)

    assert len(sample) == 2
