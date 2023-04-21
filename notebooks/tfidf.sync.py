import numpy as np
from gensim.corpora import Dictionary
from gensim.matutils import sparse2full
from gensim.models import TfidfModel

from transformer_document_embedding.tasks.wikipedia_similarities import \
    WikipediaSimilarities

# %%
task = WikipediaSimilarities(dataset="wine", datasets_dir="../data")
# %%

train_words = task.train.map(
    lambda doc: {"words": doc["text"].split()}, remove_columns=["text"]
)
# %%
for i, split_text in enumerate(train_words["words"]):
    print(split_text[:10])
    print()
    if i > 2:
        break

# %%
gensim_dict = Dictionary(train_words["words"])
print(len(gensim_dict.keys()))
# %%
gensim_dict.filter_extremes(no_below=10, no_above=0.8)
print(len(gensim_dict.keys()))
# %%
vector_dim = len(gensim_dict.keys())
print(vector_dim)
# %%
model = TfidfModel(dictionary=gensim_dict)
# %%
# vector_a = model[gensim_dict.doc2bow(train_words["words"][0])]
vector_b = model[gensim_dict.doc2bow(train_words["words"][0])]
# %%
print(vector_a)
# %%
a = sparse2full(vector_a, vector_dim)
# %%
b = sparse2full(vector_b, vector_dim)
# %%
np.testing.assert_almost_equal(a, b)
# %%
print(list(zip(a[:100], b[:100])))
# %% [markdown]
# ## Testing my model
# %%

from transformer_document_embedding.baselines.tfidf import TFIDF
from transformer_document_embedding.tasks.wikipedia_similarities import \
    WikipediaSimilarities

# %%
task = WikipediaSimilarities(dataset="wine", datasets_dir="../data")
model = TFIDF()

# %%

res = task.evaluate(model.predict(task.test))
# %%
print(res)
