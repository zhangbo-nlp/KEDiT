import json

import datasets


class PubMedDialogData(datasets.GeneratorBasedBuilder):
    """PubMedDialog dataset."""

    def _info(self):
        features = {
            "knowledge": datasets.Value("string"),
            "context": datasets.features.Sequence(feature=datasets.Value("string")),
            "response": datasets.Value("string"),
        }

        return datasets.DatasetInfo(
            features=datasets.Features(features),
        )

    def _split_generators(self, dl_manager):
        data_file = "data/pubmed_dialog/pubmed_dialog.jsonl"
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "file": data_file,
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "file": data_file,
                    "split": "validation",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "file": data_file,
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(self, file, split):
        with open(file) as f:
            for id_, line in enumerate(f):
                episode = json.loads(line)
                conversation_id = episode["conversation_id"]
                if (
                    (split == "train" and conversation_id.startswith("train_"))
                    or (split == "validation" and conversation_id.startswith("val_"))
                    or (split == "test" and conversation_id.startswith("test_"))
                ):
                    for i in range(len(episode["conversation"])):
                        context = [
                            episode["conversation"][j]["human"] for j in range(i)
                        ]
                        context += [
                            episode["conversation"][j]["assistant"] for j in range(i)
                        ]
                        context.append(episode["conversation"][i]["human"])
                        yield (
                            f"{id_}_{i}",
                            {
                                "knowledge": episode["knowledge"],
                                "context": context,
                                "response": episode["conversation"][i]["assistant"],
                            },
                        )
