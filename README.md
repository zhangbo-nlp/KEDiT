# KEDiT

This repository contains the code for Efficient Tuning of Large Language Models for Knowledge-Grounded Dialogue Generation.

## Datasets

### Wikipedia

Load the Wikipedia dataset using the `datasets` library in Python. To process and save it in JSONL format, run:

```bash
cd data/wizard_of_wikipedia
python process_wiki.py
```

### Wizard Of Wikipedia

Download the processed Wizard of Wikipedia dataset from [this link](https://drive.google.com/file/d/1yzjqACw2_MPZ1YRo6sS2iAd2z7nEzrfF/view?usp=sharing). Extract the files and organize them as follows:


```
data
|-- wizard_of_wikipedia
    ├-- test_random_split.jsonl
    ├-- test_topic_split.jsonl
    ├-- train.jsonl
    ├-- valid_random_split.jsonl
    ├-- valid_topic_split.jsonl
```

### PubMed-Dialog

The PubMed-Dialog dataset evaluates the model's ability to generate dialogues in specialized domains. It contains multi-turn dialogues generated by GPT-4o based on PubMed article abstracts. Download the processed PubMed-Dialog dataset from [this link](https://drive.google.com/file/d/1PQLyi44rrhaewwcA11Z7jwnBEmoTsxec/view?usp=sharing).

**Construction Process:**

1. **Data Collection:** Selected relevant research articles from PubMed, using abstracts as the knowledge context.
2. **Dialogue Generation:** Used specific prompts to guide GPT-4o in generating dialogues.
3. **Iterative Validation:** Implemented a three-round evaluation to ensure quality and accuracy.

**Dataset Structure:**

- **10,930** dialogues, averaging **4.36** turns per dialogue.
- Each entry in `pubmed_dialog.jsonl` includes:
  - `conversation_id`: The unique identifier
  - `category`: `"pubmed_dialog"`
  - `knowledge`: PubMed article abstract
  - `conversation`: List of dialogue turns (`human` and `assistant`)

**Data Split:**

- 80% training, 10% validation, 10% testing.
- Split by `conversation_id` prefix: `"train_"`, `"val_"`, `"test_"`.

**Usage:**

To load and preprocess, use `pubmed_dialog.py` in `data/pubmed_dialog`:

```
data
|-- pubmed_dialog
     ├-- pubmed_dialog.jsonl
     ├-- pubmed_dialog.py
```

## Running Codes

### Phase 1: Knowledge Compression Training

```bash
bash scripts/run_train_stage1.sh
```

### Phase 2: Knowledge Integration Training

#### On Wizard of Wikipedia

```bash
bash scripts/run_train_stage2_wow.sh
```

#### On PubMed-Dialog

```bash
bash scripts/run_train_stage2_pmd.sh
```
