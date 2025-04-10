import json
import multiprocessing

from datasets import load_dataset
from nltk.tokenize import word_tokenize
from tqdm import tqdm


def process_data(d):
    example = {}
    example["title"] = d["title"]
    paragraphs = d["text"].split("\n")
    # Calculate the number of words and add sentences based on the condition
    word_count = 0
    selected_sentences = []
    for paragraph in paragraphs:
        if paragraph == "":
            continue
        words = word_tokenize(paragraph)
        if word_count + len(words) > 500:
            break
        selected_sentences.append(paragraph)
        word_count += len(words)

    # If the total number of words is less than 50, ignore this example
    if word_count < 50:
        return None

    example["text"] = "\n".join(selected_sentences)
    return json.dumps(example) + "\n"


def init_worker():
    import signal

    signal.signal(signal.SIGINT, signal.SIG_IGN)


def main():
    data = load_dataset("wikipedia", "20220301.en")
    with open("wiki.jsonl", "w") as f:
        pool = multiprocessing.Pool(
            processes=multiprocessing.cpu_count(), initializer=init_worker
        )
        results = []
        for result in tqdm(
            pool.imap(process_data, data["train"]),
            total=len(data["train"]),
            desc="Processing",
        ):
            if result is not None:
                results.append(result)

        for result in results:
            f.write(result)


if __name__ == "__main__":
    main()
