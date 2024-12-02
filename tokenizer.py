import unicodedata
import regex as re
import json

text = "Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺‌🇳‌🇮‌🇨‌🇴‌🇩‌🇪! 😄 The very name strikes fear and awe into the hearts of programmers worldwide. We all know we ought to “support Unicode” in our software (whatever that means—like using wchar_t for all the strings, right?). But Unicode can be abstruse, and diving into the thousand-page Unicode Standard plus its dozens of supplementary annexes, reports, and notes can be more than a little intimidating. I don’t blame programmers for still finding the whole thing mysterious, even 30 years after Unicode’s inception."


def replace_control_characters(s: str) -> str:
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch)  # this character is ok
        else:
            chars.append(f"\\u{ord(ch):04x}")  # escape
    return "".join(chars)


class Tokenizer:
    def __init__(self):
        self.pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s"""
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = {}

    def _find_pairs(self, encoded, pairs_to_update=None):
        pairs = {}

        for i in range(len(encoded) - 1):
            pair = (encoded[i], encoded[i + 1])
            if pairs_to_update is None:
                pairs[pair] = pairs.get(pair, 0) + 1
            else:
                pairs_to_update[pair] = pairs_to_update.get(pair, 0) + 1

        return pairs

    def _merge(self, tokens, pair, new_token):
        i = 0
        new_tokens = []
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                new_tokens.append(new_token)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens

    def train(self, text_data: str, max_new_tokens):
        chunks: list[str] = re.findall(self.pattern, text_data)
        tokenized = [list(chunk.encode("utf-8")) for chunk in chunks]

        for i in range(max_new_tokens):
            print(i)
            pairs_to_replace = {}

            for chunk in tokenized:
                self._find_pairs(chunk, pairs_to_replace)

            try:
                pair = max(pairs_to_replace, key=pairs_to_replace.get)
            except ValueError:
                break

            new_token = 256 + i
            self.vocab[new_token] = self.vocab[pair[0]] + self.vocab[pair[1]]
            tokenized = [self._merge(chunk, pair, new_token) for chunk in tokenized]
            # result = merge(result, pair, new_token)

        return self.vocab

    def decode(self, encoded):
        result = b""
        for i in encoded:
            result += self.vocab[i]
        # print(result)
        return result.decode("utf-8", errors="replace")
        # result = b"".join(self.vocab[i] for i in encoded)
        # return result.decode("utf-8", errors="replace")

    def encode(self, text):
        chunks = re.findall(self.pattern, text)
        result = []

        for chunk in chunks:
            chunk_bytes = list(chunk.encode("utf-8"))

            while True:
                # print(chunk_bytes)
                pairs = self._find_pairs(chunk_bytes)

                merges = []
                for pair in pairs:
                    bytes_to_replace = bytes(self.vocab[pair[0]] + self.vocab[pair[1]])
                    for k, v in self.vocab.items():
                        if v == bytes_to_replace:
                            merges.append((k, pair))
                            break

                if len(merges) == 0:
                    break

                merge = max(merges)
                chunk_bytes = self._merge(chunk_bytes, merge[1], merge[0])

            result.extend(chunk_bytes)

        return result

    def save(self, path):
        with open(path, "w") as file:
            json.dump({k: list(v) for k, v in self.vocab.items()}, file)

    def load(self, path):
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
            self.vocab = {int(k): bytes(v) for k, v in data.items()}
