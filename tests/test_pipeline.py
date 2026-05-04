from nlp_pipeline.config import load_config
from nlp_pipeline.data import load_training_data
from nlp_pipeline.modeling import train_model
from nlp_pipeline.pipeline import train_pipeline
from nlp_pipeline.newsbot import NewsBotSystem


def test_train_pipeline_returns_manifest(monkeypatch) -> None:
    config = load_config("configs/pipeline.yaml")

    saved_paths = []

    def fake_save_model(model, path):
        saved_paths.append(path)

    def fake_save_json(payload, path):
        saved_paths.append(path)

    def fake_write_text(self, content, encoding=None):
        saved_paths.append(self)
        return len(content)

    monkeypatch.setattr("nlp_pipeline.pipeline.save_model", fake_save_model)
    monkeypatch.setattr("nlp_pipeline.pipeline.save_json", fake_save_json)
    monkeypatch.setattr("pathlib.Path.write_text", fake_write_text)
    monkeypatch.setattr("pathlib.Path.mkdir", lambda self, parents=False, exist_ok=False: None)

    manifest = train_pipeline(config)

    assert manifest["project_name"] == "newsbot-intelligence-system"
    assert "metrics" in manifest
    assert len(saved_paths) == 6


def test_train_model_produces_metrics() -> None:
    config = load_config("configs/pipeline.yaml")
    train_df, validation_df = load_training_data(config.data)
    artifacts = train_model(
        train_df=train_df,
        validation_df=validation_df,
        text_column=config.data.text_column,
        label_column=config.data.label_column,
        config=config.model,
    )

    assert set(artifacts.metrics) == {"accuracy", "macro_f1"}


def test_newsbot_system_produces_analysis() -> None:
    config = load_config("configs/pipeline.yaml")
    train_df, validation_df = load_training_data(config.data)
    artifacts = train_model(
        train_df=train_df,
        validation_df=validation_df,
        text_column=config.data.text_column,
        label_column=config.data.label_column,
        config=config.model,
    )

    newsbot = NewsBotSystem(classifier_pipeline=artifacts.pipeline)
    newsbot.ingest_articles(validation_df.to_dict(orient="records"))

    analysis = newsbot.analyze_article(
        "AI startup launches newsroom tool. The platform summarizes interviews for reporters."
    )
    response = newsbot.process_query("Find technology news about AI")

    assert analysis["classification"]["label"] in config.data.allowed_labels
    assert analysis["summary"]
    assert analysis["language"]["language"] == "en"
    assert response["intent"] == "search"
