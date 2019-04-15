from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor


@Predictor.register('name-Predictor')
class Names_Predictor(Predictor):

    @overrides
    def predict_json(self, json_dict: JsonDict) -> JsonDict:
        name = json_dict['name']
        instance = self._dataset_reader.text_to_instance(name=name)

        label_dict = self._model.vocab.get_index_to_token_vocabulary('label')
        all_labels = [label_dict[i] for i in range(len(label_dict))]

        return {"instance": self.predict_instance(instance), "all_labels": all_labels}

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        name = json_dict['name']
        return self._dataset_reader.text_to_instance(name=name)
