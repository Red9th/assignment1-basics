class Tokenizer:
    # Construct a tokenizer from a given vocabulary, list of merges, and (optionally) a list of special tokens.
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        pass

    # Class method that constructs and returns a Tokenizer from a serialized vocabulary and list of merges (in the same format that your BPE training code output) and (optionally) a list of special tokens.
    # This method should accept the following additional parameters:
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ):
        pass

    # Encode an input text into a sequence of token IDs.
    def encode(self, text: str) -> list[int]:
        pass

    # Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs.
    # This is required for memory-efficient tokenization of large files that we cannot directly load into memory.
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        pass

    # str Decode a sequence of token IDs into text.
    def decode(self, ids: list[int]):
        pass
