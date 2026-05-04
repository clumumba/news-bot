from nlp_pipeline.config import load_config
from nlp_pipeline.data import load_training_data
from nlp_pipeline.newsbot import NewsTopicModeler


def test_topic_modeler_fits_and_returns_topics() -> None:
    config = load_config("configs/pipeline.yaml")
    train_df, _ = load_training_data(config.data)

    modeler = NewsTopicModeler()
    modeler.fit_topics(train_df[config.data.text_column].tolist())

    topics = modeler.get_article_topics(train_df.iloc[0][config.data.text_column])

    assert isinstance(topics, list)

