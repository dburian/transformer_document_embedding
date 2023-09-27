import logging
import os

from transformer_document_embedding.baselines.paragraph_vector import (
    ParagraphVectorWikipediaSimilarities,
)
from transformer_document_embedding.tasks.wikipedia_similarities import (
    WikipediaSimilarities,
)

# %%
logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)

task = WikipediaSimilarities(
    dataset="wine",
    datasets_dir="../data",
    validation_source="test",
    validation_source_fraction=0.2,
)
model = ParagraphVectorWikipediaSimilarities(
    dbow_kwargs={
        "vector_size": 100,
        "min_count": 2,
        "epochs": 100,
        "negative": 5,
        "sample": 0,
    }
)
# %%
task.validation
# %%
len(list(filter(lambda doc: len(doc["label"]) > 0, task.validation)))
# %%
os.makedirs("./model", exist_ok=True)
model.train(task, log_dir="./logs", save_best_path="./model")
