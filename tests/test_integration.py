from nlp_pipeline.config import load_config
from nlp_pipeline.data import load_training_data
from nlp_pipeline.modeling import train_model
from nlp_pipeline.newsbot import NewsBotSystem


def test_end_to_end_newsbot_flow() -> None:
    config = load_config("configs/pipeline.yaml")
    train_df, validation_df = load_training_data(config.data)
    artifacts = train_model(
        train_df=train_df,
        validation_df=validation_df,
        text_column=config.data.text_column,
        label_column=config.data.label_column,
        config=config.model,
    )

    bot = NewsBotSystem(classifier_pipeline=artifacts.pipeline)
    bot.ingest_articles(validation_df.to_dict(orient="records"))

    analysis = bot.analyze_article("AI startup launches newsroom assistant for reporters.")
    response = bot.process_query("Find technology news about AI")

    assert analysis["summary"]
    assert response["intent"] == "search"

