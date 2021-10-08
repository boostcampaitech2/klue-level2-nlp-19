import json
import os
from typing import List

from konlpy.tag import Mecab

from tokenizer.base import BaseTokenizer


class MeCabTokenizer(BaseTokenizer):
    def __init__(self, config_path: str):
        self.mecab = Mecab()

    def tokenize(self, text: str) -> List[str]:
        text = text.strip()
        text_ptr = 0
        tokenized = []
        for mor in self.mecab.morphs(text):
            tokenized.append(mor)
            text_ptr += len(mor)
        return tokenized

    def detokenize(self, tokens: List[str]) -> str:
        text = "".join(tokens).replace("▃", " ").strip()
        return text