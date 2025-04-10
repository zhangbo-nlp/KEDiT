import sys
from collections import Counter, defaultdict

import language_evaluation
import numpy as np
import torch
from nltk import word_tokenize
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu


class DialogEvaluator:
    def __init__(
        self, metric_name=None, tokenizer=None, eval_selection=False, eval_gen=True
    ):
        metric_names = metric_name.split("&")
        metric_fns = {"gen": [], "cls": []}
        for name in metric_names:
            if name.strip() == "f1":
                metric_fn = self.get_unigram_F1
                keys = ["f1"]
                metric_type = "gen"
            elif name.strip() == "bleu":
                metric_fn = self.compute_corpus_bleu
                keys = ["bleu1", "bleu2", "bleu3", "bleu4", "bleu"]
                metric_type = "gen"
            elif name.strip() == "rouge":
                self.rouge_evaluator = language_evaluation.RougeEvaluator(
                    num_parallel_calls=1,
                )
                metric_fn = self.compute_rouge
                keys = ["rouge1", "rouge2", "rougeL", "rouge"]
                metric_type = "gen"
            elif name.strip() == "dist":
                metric_fn = self.calc_corpus_distinct
                keys = ["distinct-1", "distinct-2"]
                metric_type = "gen"
            else:
                raise NotImplementedError
            metric_fns[metric_type].append((metric_fn, keys))

        self.metric_fns = metric_fns
        self.tokenizer = tokenizer
        self.eval_selection = eval_selection
        self.eval_gen = eval_gen

    def __call__(self, eval_predictions):
        predictions, labels = eval_predictions
        cls_predictions, cls_labels = None, None
        if isinstance(predictions, tuple):
            if self.eval_selection:
                cls_predictions = predictions[1]
                cls_labels = labels[1]

            predictions = predictions[0]
            labels = labels[0] if isinstance(labels, tuple) else labels

        results = (
            self.compute(predictions, labels, post_proc=True) if self.eval_gen else {}
        )
        if self.eval_selection:
            assert cls_predictions is not None
            cls_results = self.compute_cls(cls_predictions, cls_labels)
            results.update(cls_results)

        return results

    def compute(self, pred, gold, post_proc=False, prefix=""):
        if post_proc:
            # decode both pred and gold if ther are tensors
            if torch.is_tensor(pred):
                assert self.tokenizer is not None
                pred = self.tokenizer.batch_decode(pred, skip_special_tokens=True)

            if torch.is_tensor(gold):
                assert self.tokenizer is not None
                gold = np.where(gold != -100, gold, self.tokenizer.pad_token_id)
                gold = self.tokenizer.batch_decode(gold, skip_special_tokens=True)

            # double check to remove \n
            pred = [p.strip() for p in pred]
            gold = [g.strip() for g in gold]

        results = {}
        for metric_fn, keys in self.metric_fns["gen"]:
            metric_results = metric_fn(pred, gold)
            for k, v in zip(keys, metric_results):
                results[prefix + k] = round(v * 100, 2)
        return results

    def compute_cls(self, pred, gold, prefix=""):
        results = {}
        for metric_fn, keys in self.metric_fns["cls"]:
            metric_results = metric_fn(pred, gold)

            for k, v in zip(keys, metric_results):
                results[prefix + k] = round(v * 100, 2)
        return results

    def _preproc_preds_golds(self, pred, gold=None):
        cands = []
        golds = []
        help_tokenize = lambda x: word_tokenize(x.lower())
        # help_tokenize = lambda x:re.findall(r"\w+|[^\w\s]", x.lower())
        for idx, p in enumerate(pred):
            cands.append(help_tokenize(p.lower()))
            if gold is not None:
                golds.append(help_tokenize(gold[idx].lower()))
        return cands, golds

    def _get_ngrams(self, text, n):
        """
        Returns all ngrams that are in the text.
        Note: this function does NOT lowercase text. If you want to lowercase, you should
        do so before calling this function.
        Inputs:
        text: string, space-separated
        n: int
        Returns:
        list of strings (each is a ngram, space-separated)
        """
        tokens = text.split()
        return [
            " ".join(tokens[i : i + n]) for i in range(len(tokens) - (n - 1))
        ]  # list of str

    """
    Compute unigram-F1 score
    """

    def _prec_recall_f1_score(self, pred_items, gold_items):
        """
        PARLAI
        Computes precision, recall and f1 given a set of gold and prediction items.
        :param pred_items: iterable of predicted values
        :param gold_items: iterable of gold values
        :return: tuple (p, r, f1) for precision, recall, f1
        """
        common = Counter(gold_items) & Counter(pred_items)
        num_same = sum(common.values())
        if num_same == 0:
            return 0, 0, 0
        precision = 1.0 * num_same / len(pred_items)
        recall = 1.0 * num_same / len(gold_items)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1, precision, recall

    def get_unigram_F1(self, pred, gold):
        f1, precision, recall = [], [], []
        for p, g in zip(pred, gold):
            f1_i, precision_i, recall_i = self._prec_recall_f1_score(p, g)

            f1.append(f1_i)
            precision.append(precision_i)
            recall.append(recall_i)
        return np.mean(f1), np.mean(precision), np.mean(recall)

    def compute_corpus_bleu(self, pred, gold):
        hypothesis, references = self._preproc_preds_golds(pred, gold)

        references = [[ref] for ref in references]
        sf = SmoothingFunction(epsilon=1e-12).method1
        b1 = corpus_bleu(
            references, hypothesis, weights=(1.0 / 1.0,), smoothing_function=sf
        )
        b2 = corpus_bleu(
            references,
            hypothesis,
            weights=(1.0 / 2.0, 1.0 / 2.0),
            smoothing_function=sf,
        )
        b3 = corpus_bleu(
            references,
            hypothesis,
            weights=(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
            smoothing_function=sf,
        )
        b4 = corpus_bleu(
            references,
            hypothesis,
            weights=(1.0 / 4.0, 1.0 / 4.0, 1.0 / 4.0, 1.0 / 4.0),
            smoothing_function=sf,
        )
        return b1, b2, b3, b4, (b1 + b2 + b3 + b4) / 4

    def compute_rouge(self, pred, gold):
        pred, gold = self._preproc_preds_golds(pred, gold)
        predictions = [" ".join(c) for c in pred]
        answers = [" ".join(g) for g in gold]
        scores = self.rouge_evaluator.run_evaluation(predictions, answers)
        rouge = (scores["rouge1"] + scores["rouge2"] + scores["rougeL"]) / 3
        return scores["rouge1"], scores["rouge2"], scores["rougeL"], rouge

    def _calc_ngram_dict(self, tokens: list[str], ngram: int, dict_ref=None):
        ngram_dict = defaultdict(int) if dict_ref is None else dict_ref
        total = len(tokens)
        for i in range(0, total - ngram + 1):
            item = tuple(tokens[i : i + ngram])
            ngram_dict[item] += 1
        return ngram_dict

    def _calc_distinct_ngram(self, cands, ngram):
        ngram_total = 0.00001
        ngram_distinct_count = 0.00001
        pred_dict = defaultdict(int)
        for cand_tokens in cands:
            self._calc_ngram_dict(cand_tokens, ngram, pred_dict)
        for key, freq in pred_dict.items():
            ngram_total += freq
            ngram_distinct_count += 1
        return ngram_distinct_count / ngram_total

    def _calc_sent_distinct_ngram(self, cand, ngram):
        ngram_total = 0.0000000001
        ngram_distinct_count = 0.0
        ngram_dict = defaultdict(int)
        for i in range(0, len(cand) - ngram + 1):
            item = tuple(cand[i : i + ngram])
            ngram_dict[item] += 1
        for _, freq in ngram_dict.items():
            ngram_total += freq
            ngram_distinct_count += 1
        return ngram_distinct_count / ngram_total

    def calc_corpus_distinct(self, cands, golds=None):
        cands, _ = self._preproc_preds_golds(cands)
        distinct1 = self._calc_distinct_ngram(cands, 1)
        distinct2 = self._calc_distinct_ngram(cands, 2)
        return distinct1, distinct2


if __name__ == "__main__":
    evaluator = DialogEvaluator(metric_name="bleu&rouge&f1&dist")

    pred_path = sys.argv[1]
    gold_path = sys.argv[2]

    with open(pred_path) as f:
        pred = f.readlines()
    with open(gold_path) as f:
        gold = f.readlines()

    results = evaluator.compute(pred, gold, post_proc=True)

    print(results)

