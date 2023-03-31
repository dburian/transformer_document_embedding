import ast
import csv
import os
import pickle
import sys
import urllib.request
from typing import Any

from datasets.load import load_dataset

# %%

articles = load_dataset("../data/wikipedia_wines.py", "articles")
# %%
print(articles)
# %%
sims = load_dataset("../data/wikipedia_wines.py", "sims")
# %%
print(sims)
# %%
print(sims["train"][:10])
# %%
wines_articles_path = os.path.join("/", "home", "dburian", "downloads", "wines.txt")
wines_gt_path = os.path.join("/", "home", "dburian", "downloads", "gt")

# %%
csv.field_size_limit(sys.maxsize)
with open(wines_articles_path, newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    all_articles = list(reader)[1:]

article = all_articles[0]
print(article)
# %%
title, sections = article[0], ast.literal_eval(article[1])
print(title)
# %%
print(sections)
# %%
for article in all_articles:
    title, sections = article[0], ast.literal_eval(article[1])
    for sec_ind, section in enumerate(sections):
        if section[1] == "":
            print(f"{title}: {sec_ind}/{len(sections)}; {section}")

# %%
loaded_gt = pickle.load(open(wines_gt_path, "rb"))
print(type(loaded_gt))
# %%
print(loaded_gt.default_factory)
# %%
print(loaded_gt.keys())
print(len(loaded_gt.keys()))
# %%
for key, item in loaded_gt.items():
    # print(f"{key}[{len(item.keys())}]: {item}")
    # print()
    for sim in item.values():
        assert sim == 1, f"{item}"
# %%
def get_gt_seeds_titles(titles=None):
    popular_titles = list(pickle.load(open(wines_gt_path, "rb")).keys())
    idxs = None
    if titles is not None:
        idxs = [
            titles.index(pop_title)
            for pop_title in popular_titles
            if pop_title in titles
        ]
    return popular_titles, idxs


# %%

WINES_ARTICLES_URL = "https://zenodo.org/record/4812960/files/wines.txt?download=1"


def download_articles(save_path: str, url: str = WINES_ARTICLES_URL) -> None:
    if not os.path.exists(save_path):
        print(f"Downloading file {save_path}...", file=sys.stderr)
        urllib.request.urlretrieve(url, filename=save_path + ".tmp")
        os.rename(save_path + ".tmp", save_path)


def parse_articles(path: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
    csv.field_size_limit(sys.maxsize)
    title_to_id = {}
    split = []
    with open(path, newline="", encoding="utf-8") as articles_file:
        reader = csv.DictReader(articles_file, fieldnames=["title", "sections"])

        reader_iter = iter(reader)
        # Skipping csv header
        next(reader)

        for article_id, article in enumerate(reader_iter):
            title = article["title"]
            sections = ast.literal_eval(article["sections"])
            sections_data = []
            for section_title, section_text in sections:
                if section_text == "":
                    continue
                sections_data.append({"title": section_title, "text": section_text})

            assert (
                title not in title_to_id
            ), f"Titles are not unique: '{title}' occured twice."

            title_to_id[title] = article_id
            split.append(
                {
                    "id": article_id,
                    "title": title,
                    "sections": sections_data,
                }
            )

    return split, title_to_id


WINES_SIMS_URL = "https://github.com/microsoft/SDR/raw/main/data/datasets/wines/gt"


def download_similarities(save_path: str, url: str = WINES_SIMS_URL) -> None:
    if not os.path.exists(save_path):
        print(f"Downloading {url} to {save_path} ...", file=sys.stderr)
        urllib.request.urlretrieve(url, filename=save_path + ".tmp")
        os.rename(save_path + ".tmp", save_path)


def parse_similarities(path: str, title_to_id: dict[str, int]) -> list[dict[str, Any]]:
    sims_raw = None
    with open(path, mode="rb") as sims_file:
        sims_raw = pickle.load(sims_file)

    left_out_sources = 0
    left_out_targets = 0

    sims_data = []
    for title, sim_articles in sims_raw.items():
        if title not in title_to_id:
            left_out_sources += 1
            continue

        sim_ids = []
        for sim_title in sim_articles.keys():
            if sim_title not in title_to_id:
                left_out_targets += 1
                continue
            sim_ids.append(title_to_id[sim_title])

        sims_data.append({"source_id": title_to_id[title], "target_ids": sim_ids})

    title_to_id_name = f"{title_to_id=}".split("=")[0]
    print(
        f"Leaving out {left_out_sources} source articles. Reason: Not in"
        f" '{title_to_id_name}' map.",
        file=sys.stderr,
    )
    print(
        f"Leaving out {left_out_targets} target articles. Reason: Not in"
        f" '{title_to_id_name}' map.",
        file=sys.stderr,
    )
    return sims_data


# %%
ARTICLES_PATH = "./wines.txt"
SIMS_PATH = "./wines_sims.txt"

# %%
download_articles(ARTICLES_PATH)
# %%
all_articles, title_to_id = parse_articles(ARTICLES_PATH)
# %%
# print(title_to_id)
print("\n".join(sorted(title_to_id.keys())))
# %%
print(len(all_articles))
# %%
print(all_articles[0])
# %%
download_similarities(SIMS_PATH)
# %%
sims = parse_similarities(SIMS_PATH, title_to_id)
# %%
print(sims)
# %%
print(len(sims))
