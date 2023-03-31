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
Wines dataset.
"""

# TODO: Add a link to an official homepage for the dataset here
_HOMEPAGE = ""

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = ""

# TODO: Add link to the official dataset URLs here

CONFIG_ARTICLES_NAME = "articles"
CONFIG_SIMS_NAME = "sims"

_URLS = {
    CONFIG_ARTICLES_NAME: (
        "https://zenodo.org/record/4812960/files/wines.txt?download=1"
    ),
    CONFIG_SIMS_NAME: (
        "https://github.com/microsoft/SDR/raw/main/data/datasets/wines/gt"
    ),
}


class WikipediaWines(GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    logger = logging.getLogger("WikipediaWines")

    VERSION = datasets.Version("1.1.0")

    BUILDER_CONFIGS = [
        BuilderConfig(
            name=CONFIG_ARTICLES_NAME,
            version=VERSION,
            description="The articles themselves.",
        ),
        BuilderConfig(
            name=CONFIG_SIMS_NAME,
            version=VERSION,
            description="Similarities.",
        ),
    ]

    def _info(self):
        features = None
        if self.config.name == CONFIG_ARTICLES_NAME:
            features = datasets.Features(
                {
                    "id": datasets.Value("uint16"),
                    "title": datasets.Value("string"),
                    "sections": datasets.features.Sequence(
                        {
                            "title": datasets.Value("string"),
                            "text": datasets.Value("string"),
                        }
                    ),
                }
            )
        elif self.config.name == CONFIG_SIMS_NAME:
            features = datasets.Features(
                {
                    "source_id": datasets.Value("uint16"),
                    "target_ids": datasets.features.Sequence(datasets.Value("uint16")),
                }
            )

        if features is None:
            raise ValueError(
                f"{WikipediaWines._info.__name__}: Config name not recognised"
            )

        return DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        filepaths = dl_manager.download(_URLS)

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
                sections_data = []
                for section_title, section_text in sections:
                    if section_text == "":
                        continue
                    sections_data.append({"title": section_title, "text": section_text})

                yield article_id, {
                    "id": article_id,
                    "title": article["title"],
                    "sections": sections_data,
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
        if self.config.name == CONFIG_ARTICLES_NAME:
            return self._parse_articles(filepaths[CONFIG_ARTICLES_NAME])

        if self.config.name == CONFIG_SIMS_NAME:
            title_to_id = self._get_title_to_id_mapping(filepaths[CONFIG_ARTICLES_NAME])
            return self._parse_similarities(filepaths[CONFIG_SIMS_NAME], title_to_id)

        raise ValueError(
            f"{WikipediaWines._generate_examples.__name__}: Config name not recognised"
        )
