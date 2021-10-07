from konlpy.tag import Mecab
from tokenizers import BertWordPieceTokenizer
import hgtk



class Tokenizer:
    def __init__(self):
        self.tokenizer_type_list = ["word", "morph", "syllable", "jaso", "wordPiece"]
        self.pad_token = "<pad>"
        self.max_seq_length = 10
        self.padding = False
        self.mecab = Mecab()
        self.wp_tokenizer = BertWordPieceTokenizer("../usage_test/wordPieceTokenizer/my_tokenizer_1sen-vocab.txt", lowercase=True)
        
    def tokenize(self, text, tokenizer_type): 
        assert tokenizer_type in self.tokenizer_type_list, "정의되지 않은 tokenizer_type입니다."
        if tokenizer_type == "word":
            tokenized_text = text.split(" ")
        elif tokenizer_type == "morph":
            tokenized_text = [lemma[0] for lemma in self.mecab.pos(text)]
        elif tokenizer_type == "syllable":
            tokenized_text = list(text)
        elif tokenizer_type == "jaso":
            tokenized_text = list(hgtk.text.decompose(text))
        elif tokenizer_type == "wordPiece":
            tokenized_text = self.wp_tokenizer.encode(text).tokens
        if self.padding:
            tokenized_text += [self.pad_token] * (self.max_seq_length - len(tokenized_text))
            return tokenized_text[:self.max_seq_length]
        else:
            return tokenized_text[:self.max_seq_length]
    def batch_tokenize(self, texts, tokenizer_type):
        for i, text in enumerate(texts):
            texts[i] = self.tokenize(text, tokenizer_type)
        return texts