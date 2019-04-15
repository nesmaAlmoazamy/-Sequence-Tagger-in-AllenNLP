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

@Predictor.register('name-Predictor')
class Names_Predictor(Predictor):

  
    def predict_json(self, json_dict: JsonDict) -> JsonDict:
        name = json_dict['name']
        instance = self._dataset_reader.text_to_instance(name=name)

        label_dict = self._model.vocab.get_index_to_token_vocabulary('label')
        all_labels = [label_dict[i] for i in range(len(label_dict))]

        return {"instance": self.predict_instance(instance), "all_labels": all_labels}

    
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        name = json_dict['name']
        return self._dataset_reader.text_to_instance(name=name)
