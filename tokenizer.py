import string

ALL_CHARS = (
    string.ascii_lowercase +   # a-z
    string.ascii_uppercase +   # A-Z
    string.digits +            # 0-9
    string.punctuation +       # !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
    " "                        # space
)

class Vocab:
    def __init__(self, tokens=None):
        if tokens is None:
            tokens = ALL_CHARS
        self.stoi = {tok: i+1 for i, tok in enumerate(tokens)}  # 0 = PAD
        self.itos = {i: tok for tok, i in self.stoi.items()}
        self.pad_id = 0

    def encode(self, text):
        return [self.stoi.get(ch, self.pad_id) for ch in text]

    def decode(self, ids):
        return "".join([self.itos.get(i, "") for i in ids if i != self.pad_id])

VOCAB = Vocab()
def pad_id():
    return VOCAB.pad_id

def vocab_size():
    return len(VOCAB.stoi) + 1

def tokenize(string):
    return VOCAB.encode(string)
