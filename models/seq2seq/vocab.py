import torch
import torch.nn as nn
from typing import List, Dict
from .custom_types import Token


class Vocab(nn.Module):
    def __init__(self, messages: List[Dict]):
        super().__init__()
        self.word2index: Dict[str, int] = {"<pad>": Token.PAD_TOKEN.value, "<bos>": Token.BOS_TOKEN.value, "<eos>": Token.EOS_TOKEN.value, "<unk>": Token.UNK_TOKEN.value}
        self.index2word: Dict[int, str] = {Token.PAD_TOKEN.value: "<pad>", Token.BOS_TOKEN.value: "<bos>", Token.EOS_TOKEN.value: "<eos>", Token.UNK_TOKEN.value: "<unk>"}
        self.word_count: Dict[str, int] = dict()
        self.size = 4

        for message in messages:
            self.add_sentence(message["text"])

        self.embedding = nn.Embedding(self.size, 300)

    def add_sentence(self, sentence):
        for word in sentence:
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.size
            self.index2word[self.size] = word
            self.word_count[word] = 1
            self.size += 1
        else:
            self.word_count[word] += 1

    def sentence_indices(self, sentence: List[str]) -> torch.LongTensor:
        indices = torch.LongTensor(len(sentence))
        for i, word in enumerate(sentence):
            indices[i] = self.word2index[word] if word in self.word2index else Token.UNK_TOKEN.value
        return indices

    def forward(self, indices: torch.LongTensor):
        return self.embedding(indices)

    def __len__(self):
        return self.size