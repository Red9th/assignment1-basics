import argparse
import pickle
import time
from pathlib import Path

from cs336_basics.BPETrainer import byte_level_bpe


def save_results(
    vocab,
    merges,
    input_path,
    vocab_size,
    output_root="output",
):
    dataset_name = Path(input_path).stem

    output_dir = Path(output_root) / f"{dataset_name}_{vocab_size}"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    with open(output_dir / "merges.pkl", "wb") as f:
        pickle.dump(merges, f)

    return output_dir


def analyze_vocab(vocab):
    longest = max(vocab.values(), key=len)

    print("\n===== Longest token =====")
    print("token(bytes):", longest)
    print("length:", len(longest))

    try:
        print("decode:", longest.decode("utf-8"))
    except UnicodeDecodeError:
        print("decode failed")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer.")

    parser.add_argument(
        "--input",
        type=str,
        default="data/TinyStoriesV2-GPT4-train.txt",
        help="Training corpus path.",
    )

    parser.add_argument(
        "--vocab-size",
        type=int,
        default=10000,
        help="Target vocabulary size.",
    )

    parser.add_argument(
        "--num-processes",
        type=int,
        default=2,
        help="Number of worker processes.",
    )

    parser.add_argument(
        "--output-root",
        type=str,
        default="output",
        help="Root directory for outputs.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    special_tokens = ["<|endoftext|>"]

    print("===== Configuration =====")
    print(f"Input          : {args.input}")
    print(f"Vocab size     : {args.vocab_size}")
    print(f"Num processes  : {args.num_processes}")
    print(f"Output root    : {args.output_root}")
    print("=========================\n")

    print("Start training...")

    start = time.perf_counter()

    vocab, merges = byte_level_bpe(
        args.input,
        args.vocab_size,
        special_tokens,
        args.num_processes,
    )

    total_time = time.perf_counter() - start

    output_dir = save_results(
        vocab,
        merges,
        args.input,
        args.vocab_size,
        output_root=args.output_root,
    )

    print("\n===== Training Result =====")
    print(f"Total time : {total_time:.2f}s")
    print(f"Saved to   : {output_dir}")

    analyze_vocab(vocab)


if __name__ == "__main__":
    main()
