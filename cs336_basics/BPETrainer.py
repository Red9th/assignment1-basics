import regex as re
import os
from collections import defaultdict
import multiprocessing
import heapq
import math

from cs336_basics.pretokenization_example import find_chunk_boundaries


class candidate:
    __slots__ = ("freq", "pair")

    def __init__(self, freq, pair):
        self.freq = freq
        self.pair = pair

    def __lt__(self, other):
        if self.freq != other.freq:
            return self.freq < other.freq
        return self.pair > other.pair


class TokenChain:
    def __init__(self, frequency=0):
        self.val: list[int] = []
        self.prev: list[int] = []
        self.next: list[int] = []
        self.head: int = -1
        self.tail: int = -1
        self.frequency = frequency

    def append(self, b: bytes) -> int:
        idx = len(self.val)
        self.val.append(b)
        self.prev.append(self.tail)
        self.next.append(-1)

        if self.tail != -1:
            self.next[self.tail] = idx
        else:
            self.head = idx

        self.tail = idx
        return idx

    def get_val(self, idx: int) -> bytes:
        if idx == -1:
            return None
        return self.val[idx]

    def get_next(self, idx: int) -> int:
        if idx == -1:
            return -1
        return self.next[idx]

    def get_prev(self, idx: int) -> int:
        if idx == -1:
            return -1
        return self.prev[idx]

    def merge(self, idx: int) -> bytes:
        l = idx
        r = self.next[l]
        merged = self.val[l] + self.val[r]

        pre = self.prev[l]
        nxt = self.next[r]

        self.val[l] = merged
        self.next[l] = nxt

        self.prev[r] = -1
        self.next[r] = -1

        if pre != -1:
            self.next[pre] = l

        if nxt != -1:
            self.prev[nxt] = l

        return merged

    def print(self):
        now = self.head
        while now != -1:
            print(self.val[now], end=" ")
            now = self.next[now]


class BPETrainer:
    def __init__(
        self, vocab_size: int, special_tokens: list[str], pre_tokens: list[TokenChain]
    ):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.pre_tokens = pre_tokens

    def train_bpe(
        self, vocab: list[bytes]
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        merge_proc = []
        # candidate pairs ordered by frequency; heapq is min-heap, so store negative counts.
        merge_candidates = []

        # pair -> frequency
        pair_freq = defaultdict(int)
        # pair -> locations
        pair_locs = defaultdict(set)
        for idx, chain in enumerate(self.pre_tokens):
            now = chain.head
            while now != -1 and chain.get_next(now) != -1:
                pair = (chain.get_val(now), chain.get_val(chain.get_next(now)))
                pair_freq[pair] += chain.frequency
                pair_locs[pair].add((idx, now))
                now = chain.get_next(now)

        # max heap
        merge_candidates = []
        for pair, freq in pair_freq.items():
            heapq.heappush(merge_candidates, candidate(-freq, pair))

        # merges
        while len(vocab) < self.vocab_size:
            if merge_candidates:
                top = heapq.heappop(merge_candidates)
            else:
                break
            freq = top.freq
            pair = top.pair
            if -freq != pair_freq[pair]:
                continue

            merge_proc.append(pair)

            merged_bytes = None

            locations = sorted(pair_locs[pair])
            for idx, loc in locations:
                token = self.pre_tokens[idx]
                b = loc
                c = token.get_next(b)

                # get adjacent nodes
                a = token.get_prev(b)
                d = token.get_next(c)

                if token.get_val(b) != pair[0] or token.get_val(c) != pair[1]:
                    continue

                # remove old pair and merge
                bc = (token.get_val(b), token.get_val(c))
                pair_freq[bc] -= token.frequency
                pair_locs[bc].discard((idx, b))
                if pair_freq[bc] > 0:
                    heapq.heappush(merge_candidates, candidate(-pair_freq[bc], bc))
                if a != -1:
                    ab = (token.get_val(a), token.get_val(b))
                    pair_freq[ab] -= token.frequency
                    pair_locs[ab].discard((idx, a))
                    if pair_freq[ab] > 0:
                        heapq.heappush(merge_candidates, candidate(-pair_freq[ab], ab))
                if d != -1:
                    cd = (token.get_val(c), token.get_val(d))
                    pair_freq[cd] -= token.frequency
                    pair_locs[cd].discard((idx, c))
                    if pair_freq[cd] > 0:
                        heapq.heappush(merge_candidates, candidate(-pair_freq[cd], cd))

                # add new pair
                merged = token.merge(b)
                if merged_bytes == None:
                    merged_bytes = merged

                if a != -1:
                    am = (token.get_val(a), merged)
                    pair_freq[am] += token.frequency
                    pair_locs[am].add((idx, a))
                    if pair_freq[am] > 0:
                        heapq.heappush(merge_candidates, candidate(-pair_freq[am], am))

                if d != -1:
                    md = (merged, token.get_val(d))
                    pair_freq[md] += token.frequency
                    pair_locs[md].add((idx, b))
                    if pair_freq[md] > 0:
                        heapq.heappush(merge_candidates, candidate(-pair_freq[md], md))

            if merged_bytes != None:
                vocab.append(merged_bytes)

        idx_vocab = {}
        for i, token in enumerate(vocab):
            idx_vocab[i] = token

        return (idx_vocab, merge_proc)


def get_counts(text: str) -> dict[tuple[bytes, bytes], int]:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    counts = defaultdict(int)
    for match in re.finditer(PAT, text):
        token = match.group()
        token_byte_tuple = tuple(bytes([char]) for char in token.encode("utf-8"))
        counts[token_byte_tuple] += 1
    return counts


def count_pre_tokens(segments: list[str]) -> dict[tuple[bytes, ...], int]:
    counts = defaultdict(int)
    for segment in segments:
        for key, val in get_counts(segment).items():
            counts[key] += val
    return counts


def process_chunk(args):
    input_path, start, end, special_tokens = args

    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    segments = re.split("|".join(map(re.escape, special_tokens)), chunk)
    return count_pre_tokens(segments)


def byte_level_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int,
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # initialization
    if num_processes < 1:
        raise ValueError("num_processes must be at least 1")
    assert vocab_size >= 256 + len(special_tokens)
    vocab: list[bytes] = [bytes([i]) for i in range(256)]
    for token in special_tokens:
        vocab.append(token.encode())

    # Bound per-worker input size separately from worker concurrency. Large
    # chunks otherwise make every worker hold hundreds of MiB of Python data.
    target_chunk_bytes = int(kwargs.get("target_chunk_bytes", 64 * 1024 * 1024))
    file_size = os.path.getsize(input_path)
    num_chunks = max(num_processes, math.ceil(file_size / target_chunk_bytes))
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f,
            num_chunks,
            special_tokens[0].encode(),
        )

    tasks = [
        (input_path, start, end, special_tokens)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]

    counts = defaultdict(int)
    with multiprocessing.Pool(num_processes) as pool:
        for chunk_counts in pool.imap(process_chunk, tasks, chunksize=1):
            for key, val in chunk_counts.items():
                counts[key] += val

    pre_tokens = []
    for key, frequency in counts.items():
        token_chain = TokenChain(frequency=frequency)
        for token_byte in key:
            token_chain.append(token_byte)
        pre_tokens.append(token_chain)

    bpe_trainer = BPETrainer(vocab_size, special_tokens, pre_tokens)
    return bpe_trainer.train_bpe(vocab)
