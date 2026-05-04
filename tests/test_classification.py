from nlp_pipeline.config import load_config
from nlp_pipeline.data import load_training_data
from nlp_pipeline.modeling import NewsClassifier


def test_classifier_trains_and_predicts() -> None:
    config = load_config("configs/pipeline.yaml")
    train_df, validation_df = load_training_data(config.data)

    classifier = NewsClassifier(config=config.model)
    classifier.train(train_df[config.data.text_column], train_df[config.data.label_column])

    prediction = classifier.predict_with_confidence(validation_df.iloc[0][config.data.text_column])

    assert prediction["label"] in config.data.allowed_labels
    assert 0.0 <= prediction["confidence"] <= 1.0

