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
