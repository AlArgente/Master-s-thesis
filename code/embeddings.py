import io
import time
import numpy as np
from abc import ABC, abstractmethod

class Embeddings(ABC):
    def __init__(self):
        self.d = 300
        self.__wordvecs = {}

    @property
    def wordvecs(self):
        return self.__wordvecs

    @property
    def vocabulary(self):
        return list(self.__wordvecs.keys())

    @abstractmethod
    def load_vectors(self, fname):
        pass

    def calc_embeddings(self, text):
        """Function that apply the vector to get the embeddings from the text.
        Arguments:
            - text: list of lists with the text at least tokenized.
        Returns:
            - embeddings: list of embeddings.
        """
        embeddings = []
        # null_embeddings = np.zeros(self.d)

        for sentence in text:
            embedding_sentence = []
            for word in sentence:
                if word in self.__wordvecs:
                    embedding_word = self.__wordvecs[word]
                else:
                    embedding_word = np.random.normal(0, 1, self.d)
                    self.__wordvecs[word] = embedding_word
                embedding_sentence.append(embedding_word)
            embeddings.append(embedding_sentence)

        """
        # Alternative:
        for i in range(len(text)):
            embedding_sentence = []
            for j in range(len(text[i])):
                if text[i][j] in self.data:
                    embedding_word = self.data[text[i][j]]
                else:
                    embedding_word = np.random.normal(0, 1, self.d)
                embedding_sentence.append(embedding_word)
            embeddings.append(embedding_sentence)
        """
        return np.asarray(embeddings)



class GloveEmbeddings(Embeddings):
    def __init__(self):
        super(GloveEmbeddings, self).__init__()
        self.load_vectors()
        print('Embeddings cargados')

    def load_vectors(self, fname='../glove.6B.300d.txt'):
        with open(fname, 'r') as f:
            lines = f.readlines()
            for line in lines:
                token = line.split(' ')
                vec = np.array(token[1:], dtype=np.float32)
                self.wordvecs[token[0]] = vec


class FTEmbeddings(Embeddings):
    def __init__(self):
        super(FTEmbeddings, self).__init__()
        self.load_vectors()
        print('Embeddings cargados')

    def load_vectors(self, fname='../wiki-news-300d-1M.vec'):
        """Function to load the Fasttext embeddings instead of random initialize them
        """
        fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        # n, d = map(int, fin.readline().split())
        print('Paso aquí')
        data = {}
        for line in fin:
            tokens = line.rstrip().split(' ')
            # data[tokens[0]] = map(float, tokens[1:])
            data[tokens[0]] = np.array(tokens[1:])
        return data
