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
