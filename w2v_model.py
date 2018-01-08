import os
from tqdm import tqdm

class W2V(object):
    def __init__(self):
        pass

    def import_model(self, path):
        self.path = path
        self.table = {}

        with open(path, 'r') as f:
            lines = f.readlines()
            vocab_size, embed_size = lines[0].split()

            self.vocab_size = int(vocab_size)
            self.embed_size = int(embed_size)

            for n in tqdm(range(self.vocab_size), desc='w2v', ncols=80):
                word, *vec = lines[n+1].split()
                vec = [ float(v) for v in vec ]
                assert(len(vec) == self.embed_size)

                self.table[word] = vec

        self.keys = list(self.table.keys())
        self.values = list(self.table.values())
        self.keys = ['<blank>', '<rare>'] + self.keys

        self.default_vec = [0.0] * self.embed_size

        self.values = [self.default_vec] * 2 + self.values

