# %%
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datasets.load import load_dataset

from transformer_document_embedding.tasks.wikipedia_similarities import (
    WikipediaSimilarities,
)

sns.set_theme()
# %%
articles = load_dataset(
    "../data/wikipedia_similarities.py", "game_articles", split="train"
)
# %%
sims = load_dataset("../data/wikipedia_similarities.py", "wine_sims", split="train")
# %%
print(sims[0])
# %%
print(articles[123])


# %%
def compare_articles(articles) -> None:
    print("Titles:")
    for a in articles:
        print(f"{a['id']}:{a['title']}")
        print("---")

    print()
    max_sections = max((len(a["section_titles"]) for a in articles))
    print("Sections:")
    for section_idx in range(max_sections):
        for a in articles:
            sec_titles = a["section_titles"]
            sec_texts = a["section_texts"]
            title = ""
            text = ""
            if len(sec_titles) > section_idx:
                title = sec_titles[section_idx]
                text = sec_texts[section_idx]

            print(f"{a['id']}:{title}")
            print(f"{a['id']}:{text}")
            print("---")


# %%
def compare_all():
    for i, sim in enumerate(sims):
        source_article = articles[sim["source_id"]]
        for target_id in sim["target_ids"]:
            target_article = articles[target_id]
            compare_articles([source_article, target_article])
            print("\n\n\n")
            yield i


for i in compare_all():
    if i > 10:
        break
# %%
print(articles[24])


# %%
def create_text(article: dict[str, Any]) -> dict[str, Any]:
    sections_text = [
        f"{title} {text}"
        for title, text in zip(
            article["section_titles"], article["section_texts"], strict=True
        )
    ]
    return {"text": " ".join(sections_text)}


text_articles = articles.map(create_text)
# %%
text_articles[24]
# %%


def get_found_count(text: str, to_find: list[str]) -> int:
    count = 0
    start = 0
    inds = [text.find(word) for word in to_find]
    while max(inds) > -1:
        count += 1
        start = min((ind for ind in inds if ind > -1))
        inds = [text.find(word, start + 1) for word in to_find]

    return count


source_title_freqs = []
target_title_freqs = []
for sim in sims:
    source_article = text_articles[sim["source_id"]]
    words_to_find = source_article["title"].split(" ")
    for target_id in sim["target_ids"]:
        target_article = text_articles[target_id]
        source_title_freqs.append(
            get_found_count(target_article["text"], words_to_find)
        )
        target_title_freqs.append(
            get_found_count(source_article["text"], target_article["title"].split())
        )
# %%

plt.title(
    "Occurencies of words of source's title in target's text in"
    f" {len(source_title_freqs)} source-target pairs"
)
# plt.yscale("log")
sns.histplot(source_title_freqs, bins=np.arange(-0.5, 30.5, 1))
# %%
print(len(list(filter(lambda x: x == 0, source_title_freqs))))
# %%
plt.title(
    "Occurencies of words of target's title in source's text in"
    f" {len(target_title_freqs)} source-target pairs"
)
# plt.yscale("log")
sns.histplot(target_title_freqs, bins=np.arange(-0.5, 30.5, 1))
# %% [markdown]
# ## Looking at the task
# %%
wines = WikipediaSimilarities(dataset="game", datasets_dir="../data")
# %%
len(wines.train)

# %%
len(wines.test)

# %%
replaced_char_count = 0
for article in wines.train:
    replaced_char_count += article["text"].count("??")

print(replaced_char_count)
# %%
wines_df = wines.train.to_pandas()
wines_df["Text length"] = wines_df[["text"]].map(lambda text: len(text))
wines_df["Word count"] = wines_df[["text"]].map(lambda text: len(text.split()))
# %%
wines_df.describe()
