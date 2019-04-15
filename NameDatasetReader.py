from typing import Iterator, List, Dict
import torch
import torch.optim as optim
import numpy as np
from allennlp.data import Instance
from allennlp.data.fields import TextField,LabelField,SequenceLabelField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer
from allennlp.predictors import SentenceTaggerPredictor
from allennlp.data.tokenizers import Token, CharacterTokenizer

from io import open
import glob
import os
import unicodedata
import string



@DatasetReader.register('data-reader')
class NameDatasetReader(DatasetReader):
    def __init__(self,tokenizer = None ,token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.tokenizer = tokenizer or CharacterTokenizer()

        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    def text_to_instance(self, tokens: List[Token], tags: List[str] = None) -> Instance:
        name_field = TextField(tokens, self.token_indexers)
        fields = {"name": name_field}

        if tags:
#            label_field = LabelField(tags)
            label_field = SequenceLabelField(labels=[tags]*len(tokens), sequence_field=name_field)
            fields["label"] = label_field

        return Instance(fields)

    def findFiles(self,path): return glob.glob(path)


    all_letters = string.ascii_letters + " .,;'"
    n_letters = len(all_letters)

    def unicodeToAscii(self,s):
     return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in self.all_letters
     )
    category_lines = {}
    all_categories = []

    # Read a file and split into lines
    def readLines(self,filename):
      lines = open(filename, encoding='utf-8').read().strip().split('\n')
      return [self.unicodeToAscii(line) for line in lines]

    def _read(self, file_path: str)-> Iterator[Instance]:
      for filename in self.findFiles(file_path):
#            print(filename)

            category = os.path.splitext(os.path.basename(filename))[0]
#            print (category)
            self.all_categories.append(category)
            lines = self.readLines(filename)
            for line in lines:
              yield self.text_to_instance([Token(word) for word in line], category)
            self.category_lines[category] = lines
