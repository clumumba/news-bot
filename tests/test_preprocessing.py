from nlp_pipeline.preprocessing import normalize_text


def test_normalize_text_removes_punctuation_and_extra_spaces() -> None:
    assert normalize_text("  Hello,   WORLD!! ") == "hello world"
