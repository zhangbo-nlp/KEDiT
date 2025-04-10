import json

import datasets


class WizardOfWikipediaData(datasets.GeneratorBasedBuilder):
    """WizardOfWikipedia dataset."""

    def _info(self):
        if self.config.name == "knowledge":
            features = {
                "knowledge": datasets.Value("string"),
            }
        else:
            features = {
                "knowledge": datasets.Value("string"),
                "context": datasets.features.Sequence(feature=datasets.Value("string")),
                "response": datasets.Value("string"),
            }

        return datasets.DatasetInfo(
            features=datasets.Features(features),
        )

    def _split_generators(self, dl_manager):
        if self.config.name == "knowledge":
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "file": "data/wizard_of_wikipedia/wiki.jsonl",
                    },
                ),
            ]
        else:
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "file": "data/wizard_of_wikipedia/train.jsonl",
                    },
                ),
                datasets.SplitGenerator(
                    name="valid_random",
                    gen_kwargs={
                        "file": "data/wizard_of_wikipedia/valid_random_split.jsonl",
                    },
                ),
                datasets.SplitGenerator(
                    name="valid_topic",
                    gen_kwargs={
                        "file": "data/wizard_of_wikipedia/valid_topic_split.jsonl",
                    },
                ),
                datasets.SplitGenerator(
                    name="test_random",
                    gen_kwargs={
                        "file": "data/wizard_of_wikipedia/test_random_split.jsonl",
                    },
                ),
                datasets.SplitGenerator(
                    name="test_topic",
                    gen_kwargs={
                        "file": "data/wizard_of_wikipedia/test_topic_split.jsonl",
                    },
                ),
            ]

    def _generate_examples(self, file):
        with open(file) as f:
            c = 0
            for l in f:
                episode = json.loads(l)
                if self.config.name == "knowledge":
                    c += 1
                    yield (
                        c,
                        {
                            "knowledge": episode["text"],
                        },
                    )
                else:
                    yield (
                        episode["idx"],
                        {
                            "knowledge": episode["knowledge"],
                            "context": episode["history"],
                            "response": episode["response"],
                        },
                    )
