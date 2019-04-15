#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 8 16:22:28 2019

@author: nesma
"""

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

torch.manual_seed(1)

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
         

@Model.register('name-reader')  
class NamesTagger(Model):
    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 vocab: Vocabulary) -> None:
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        self.hidden2tag = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                          out_features=vocab.get_vocab_size('labels'))
        self.accuracy = CategoricalAccuracy()
        self.loss = torch.nn.CrossEntropyLoss()
#        self.loss = torch.nn.BCEWithLogitsLoss()
#    def forward(self,sentence,labels) -> Dict[str, torch.Tensor]:

    def forward(self,
                name: Dict[str, torch.Tensor],
                label: torch.Tensor = None) -> torch.Tensor:
        
        mask = get_text_field_mask(name)
#        print("MY names",name)
#        print("Mask",mask)
        embeddings = self.word_embeddings(name)
        encoder_out = self.encoder(embeddings, mask)
#        print(encoder_out.shape)
        logits = self.hidden2tag(encoder_out)
        
        
        output = {"logits": logits}
#        print("output",output)
#        print("labels",label)
#        print(label.shape)
#        print(logits.shape)
        if label is not None:
            self.accuracy(logits, label,mask)
            output["loss"] = sequence_cross_entropy_with_logits(logits, label, mask)
#            output["loss"] = self.loss(logits,label)
        return output
       
    
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}
    
reader = NameDatasetReader()
train_dataset = reader.read("names/*.txt")
validation_dataset = reader.read('names/validation.txt')

vocab = Vocabulary.from_instances(train_dataset + validation_dataset)
#print("vocab", vocab)
EMBEDDING_DIM = 6
HIDDEN_DIM = 6

token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                            embedding_dim=EMBEDDING_DIM)
#print("vocab.get_vocab_size('tokens')",vocab.get_vocab_size('tokens'))


word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})

#print("word_embeddings",word_embeddings)
lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True))

model = NamesTagger(word_embeddings, lstm, vocab)
cuda_device = -1
optimizer = optim.SGD(model.parameters(), lr=0.1)
iterator = BucketIterator(batch_size=2, sorting_keys=[("name", "num_tokens")])
iterator.index_with(vocab)

trainer = Trainer(model=model,
                  optimizer=optimizer,
                  iterator=iterator,
                  train_dataset=train_dataset,
                  patience=10,
                  num_epochs=10,
                  cuda_device=cuda_device)
trainer.train()
#
#
#predictor = SentenceTaggerPredictor(model, dataset_reader=reader)
#tag_logits = predictor.predict("Nesma")['logits']
#tag_ids = np.argmax(tag_logits, axis=-1)
#print([model.vocab.get_token_from_index(i, 'labels') for i in tag_ids])
