import spacy
import torch
from spacy.attrs import ORTH
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import build_vocab_from_iterator

spacy_es = spacy.load('es_core_news_lg')
spacy_en = spacy.load('en_core_web_lg')

s_case = [{ORTH: "<sod>"}]
e_case = [{ORTH: "<eod>"}]
spacy_es.tokenizer.add_special_case('<sod>', s_case)
spacy_es.tokenizer.add_special_case('<eod>', e_case)
spacy_en.tokenizer.add_special_case('<sod>', s_case)
spacy_en.tokenizer.add_special_case('<eod>', e_case)


def tokenize_es(text: str) -> [str]:
    return [tok.text for tok in spacy_es.tokenizer(text)]


def tokenize_en(text: str) -> [str]:
    return [tok.text for tok in spacy_en.tokenizer(text)]


raw_english = [
    "<sod> This is just a test of machine translation. <eod>",
    "<sod> My dog's name is Lanie and she's pretty cute. <eod>",
    "<sod> Foo and bar are the de facto dummy words in Computer Science. <eod>",
    "<sod> March Madness is a bit boring this year. <eod>"
]

raw_spanish = [
    "<sod> Esta es solo una prueba de traducción automática. <eod>",
    "<sod> El nombre de mi perro es Lanie y es muy linda. <eod>",
    "<sod> Foo y bar son las palabras ficticias de facto en Ciencias de la "
    "Computación. <eod>",
    "<sod> March Madness es un poco aburrido este año. <eod>"
]

BATCH_SIZE = len(raw_english)


class GenericTranslationDataset(object):

    def __init__(self):
        tokenized_english = [tokenize_en(s) for s in raw_english]
        tokenized_spanish = [tokenize_es(s) for s in raw_spanish]
        self.english = build_vocab_from_iterator(tokenized_english)
        self.spanish = build_vocab_from_iterator(tokenized_spanish)

        self.src_pad_idx = self.english.stoi['<pad>']
        self.tgt_pad_idx = self.spanish.stoi['<pad>']
        self.src_vocab_len = len(self.english)
        self.tgt_vocab_len = len(self.spanish)

        self.src = [[self.english.stoi[t] for t in s] for s in
                    tokenized_english]
        self.tgt = [[self.spanish.stoi[t] for t in s] for s in
                    tokenized_spanish]

        self.src = pad_sequence([torch.tensor(x) for x in self.src],
            padding_value=self.src_pad_idx)
        self.tgt = pad_sequence([torch.tensor(x) for x in self.tgt],
            padding_value=self.tgt_pad_idx)

    def __iter__(self):
        yield self.src, self.tgt


def main():
    dataset = GenericTranslationDataset()
    for (src, tgt) in dataset:
        print(src)
        print(tgt)


if __name__ == '__main__':
    main()
