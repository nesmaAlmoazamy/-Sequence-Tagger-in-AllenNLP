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
