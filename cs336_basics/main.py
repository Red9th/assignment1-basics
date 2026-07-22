import argparse
import os
import pickle
import time

from cs336_basics.BPETrainer import byte_level_bpe


def save_results(vocab, merges, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "vocab.pkl"), "wb") as f:
        pickle.dump(vocab, f)

    with open(os.path.join(output_dir, "merges.pkl"), "wb") as f:
        pickle.dump(merges, f)


def analyze_vocab(vocab):
    # Find the longest token (bytes)
    longest = max(vocab.values(), key=len)

    print("\n===== Longest token =====")
    print("token(bytes):", longest)
    print("length:", len(longest))

    try:
        print("decode:", longest.decode("utf-8"))
    except UnicodeDecodeError:
        print("decode failed")

    print("========================")

    if len(longest) > 100:
        print("Warning: token is unusually long")
    else:
        print("Token length looks reasonable")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer.")

    parser.add_argument(
        "--input",
        type=str,
        default="data/TinyStoriesV2-GPT4-train.txt",
        help="Path to training corpus.",
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
        default=int(os.environ.get("BPE_NUM_PROCESSES", "2")),
        help="Number of worker processes.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory for saving vocab and merges.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    special_tokens = ["<|endoftext|>"]

    print("===== Configuration =====")
    print(f"Input          : {args.input}")
    print(f"Vocab size     : {args.vocab_size}")
    print(f"Num processes  : {args.num_processes}")
    print(f"Output dir     : {args.output_dir}")
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

    print("\n===== Training Result =====")
    print(f"Total time: {total_time:.2f}s")

    save_results(vocab, merges, args.output_dir)
    print(f"Saved vocab.pkl and merges.pkl to '{args.output_dir}'")

    analyze_vocab(vocab)


if __name__ == "__main__":
    main()
