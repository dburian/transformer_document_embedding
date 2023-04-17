import logging

from transformer_document_embedding.baselines.paragraph_vector import \
    ParagraphVectorWikipediaSimilarities
from transformer_document_embedding.tasks.wikipedia_wines import \
    WikipediaSimilarities

# %%
logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)

task = WikipediaSimilarities(dataset="wine", datasets_dir="../data")
model = ParagraphVectorWinesGames(
    dm_kwargs={
        "vector_size": 100,
        "min_count": 2,
        "epochs": 30,
        "negative": 5,
        "sample": 0,
    }
)
# %%
model.train(task)
