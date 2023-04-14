# Copyright 2023 The HuggingFace Datasets Authors and the current dataset
# script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TODO: Add a description here."""

import ast
import csv
import logging
import pickle
import sys
from typing import Any, Iterable

import datasets
from datasets.builder import BuilderConfig, GeneratorBasedBuilder
from datasets.info import DatasetInfo
from datasets.splits import Split, SplitGenerator

_CITATION = """\
@misc{ginzburg2021selfsupervised,
     title={Self-Supervised Document Similarity Ranking via Contextualized Language Models and Hierarchical Inference}, 
     author={Dvir Ginzburg and Itzik Malkiel and Oren Barkan and Avi Caciularu and Noam Koenigstein},
     year={2021},
     eprint={2106.01186},
     archivePrefix={arXiv},
     primaryClass={cs.CL}
}
"""

# TODO: Add description of the dataset here
# You can copy an official description
_DESCRIPTION = """\
Parsed Wikipedia articles about wines and games, whose similarities are judged by experts.
"""

# TODO: Add a link to an official homepage for the dataset here
_HOMEPAGE = ""

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = "MIT"

CONF_NAME_WINE_ARTICLES = "wine_articles"
CONF_NAME_WINE_SIMS = "wine_sims"

WINE_CONFIGS = [CONF_NAME_WINE_ARTICLES, CONF_NAME_WINE_SIMS]

CONF_NAME_GAME_ARTICLES = "game_articles"
CONF_NAME_GAME_SIMS = "game_sims"

GAME_CONFIGS = [CONF_NAME_GAME_ARTICLES, CONF_NAME_GAME_SIMS]

ARTICLE_CONFIGS = [CONF_NAME_GAME_ARTICLES, CONF_NAME_WINE_ARTICLES]
SIMS_CONFIGS = [CONF_NAME_GAME_SIMS, CONF_NAME_WINE_SIMS]

_WINE_URLS = {
    "articles": "https://zenodo.org/record/4812960/files/wines.txt?download=1",
    "sims": "https://github.com/microsoft/SDR/raw/main/data/datasets/wines/gt",
}
_GAME_URLS = {
    "articles": "https://zenodo.org/record/4812962/files/video_games.txt?download=1",
    "sims": "https://github.com/microsoft/SDR/raw/main/data/datasets/video_games/gt",
}


class WikipediaSimilarities(GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    logger = logging.getLogger("WikipediaWines")

    VERSION = datasets.Version("1.1.0")

    BUILDER_CONFIGS = [
        # TODO: Maybe custom config for specifying topic and type of split
        BuilderConfig(
            name=CONF_NAME_WINE_ARTICLES,
            version=VERSION,
            description="Wikipedia articles about wines split into sections.",
        ),
        BuilderConfig(
            name=CONF_NAME_WINE_SIMS,
            version=VERSION,
            description="Similar wikipedia wine articles as judged by human experts.",
        ),
        BuilderConfig(
            name=CONF_NAME_GAME_ARTICLES,
            version=VERSION,
            description="Wikipedia articles about games split into sections.",
        ),
        BuilderConfig(
            name=CONF_NAME_GAME_SIMS,
            version=VERSION,
            description="Similar wikipedia game articles as judged by human experts.",
        ),
    ]

    def _info(self):
        features = None
        if self.config.name in ARTICLE_CONFIGS:
            features = datasets.Features(
                {
                    "id": datasets.Value("uint16"),
                    "title": datasets.Value("string"),
                    "section_titles": datasets.features.Sequence(
                        datasets.Value("string")
                    ),
                    "section_texts": datasets.features.Sequence(
                        datasets.Value("string")
                    ),
                }
            )
        elif self.config.name in SIMS_CONFIGS:
            features = datasets.Features(
                {
                    "source_id": datasets.Value("uint16"),
                    "target_ids": datasets.features.Sequence(datasets.Value("uint16")),
                }
            )

        if features is None:
            raise ValueError(
                f"{WikipediaGamesWines._info.__name__}: Config name not recognised"
            )

        return DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        filepaths = None
        if self.config.name in WINE_CONFIGS:
            filepaths = dl_manager.download(_WINE_URLS)
        elif self.config.name in GAME_CONFIGS:
            filepaths = dl_manager.download(_GAME_URLS)

        if filepaths is None:
            raise ValueError(
                f"{WikipediaGamesWines._info.__name__}: Config name not recognised"
            )

        return [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={
                    "filepaths": filepaths,
                },
            ),
        ]

    def _parse_articles(self, path: str) -> Iterable[tuple[int, dict[str, Any]]]:
        csv.field_size_limit(sys.maxsize)
        with open(path, newline="", encoding="utf-8") as articles_file:
            reader = csv.DictReader(articles_file, fieldnames=["title", "sections"])
            reader_iter = iter(reader)

            # Skipping csv header
            next(reader, None)

            for article_id, article in enumerate(reader_iter):
                sections = ast.literal_eval(article["sections"])
                section_titles = []
                section_texts = []
                for section_title, section_text in sections:
                    if section_text == "":
                        continue
                    section_titles.append(section_title)
                    section_texts.append(section_text)

                yield article_id, {
                    "id": article_id,
                    "title": article["title"],
                    "section_titles": section_titles,
                    "section_texts": section_texts,
                }

    def _get_title_to_id_mapping(self, articles_path: str) -> dict[str, int]:
        csv.field_size_limit(sys.maxsize)
        title_to_id = {}
        with open(articles_path, newline="", encoding="utf-8") as articles_file:
            reader = csv.DictReader(articles_file, fieldnames=["title", "sections"])
            reader_iter = iter(reader)

            # Skipping csv header
            next(reader, None)

            for article_id, article in enumerate(reader_iter):
                title = article["title"]

                assert (
                    title not in title_to_id
                ), f"Titles are not unique: '{title}' occured twice."

                title_to_id[title] = article_id

        return title_to_id

    def _parse_similarities(
        self, path: str, title_to_id: dict[str, int]
    ) -> Iterable[tuple[int, dict[str, Any]]]:
        sims_raw = None
        with open(path, mode="rb") as sims_file:
            sims_raw = pickle.load(sims_file)

        left_out_sources = 0
        left_out_targets = 0

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

            yield title_to_id[title], {
                "source_id": title_to_id[title],
                "target_ids": sim_ids,
            }

        if left_out_sources > 0:
            self.logger.warning(
                "Leaving out %s source articles. Reason: texts of the"
                " source articles are not in the dataset.",
                left_out_sources,
            )
        if left_out_targets > 0:
            self.logger.warning(
                "Leaving out %s target articles. Reason: texts of the"
                " target articles are not in the dataset.",
                left_out_targets,
            )

    # pylint: disable=arguments-differ
    def _generate_examples(
        self,
        filepaths: dict[str, str],
    ) -> Iterable[tuple[int, dict[str, Any]]]:
        if self.config.name in [CONF_NAME_WINE_ARTICLES, CONF_NAME_GAME_ARTICLES]:
            return self._parse_articles(filepaths["articles"])

        if self.config.name in [CONF_NAME_WINE_SIMS, CONF_NAME_GAME_SIMS]:
            title_to_id = self._get_title_to_id_mapping(filepaths["articles"])
            return self._parse_similarities(filepaths["sims"], title_to_id)

        raise ValueError(
            f"{WikipediaGamesWines._generate_examples.__name__}: Config name not"
            " recognised"
        )
