import io
import numpy as np

class FTEmbeddings:
    def __init__(self):
        self.d = 300
        self.data = self.__load_vectors__()

    def __load_vectors__(self, fname='/home/alberto/TFM/cc.en.300.bin'):
        """Function to load the Fasttext embeddings instead of random initialize them
        """
        fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        n, d = map(int, fin.readline().split())
        data = {}
        for line in fin:
            tokens = line.rstrip().split(' ')
            data[tokens[0]] = map(float, tokens[1:])
        return data

    def apply_vectors(self, text):
        """Function that apply the vector to get the embeddings from the text.
        Arguments:
            - text: list of lists with the text at least tokenized.
        Returns:
            - embeddings: list of embeddings.
        """
        assert isinstance(text, list)

        embeddings = []
        null_embeddings = np.zeros(self.d)

        for sentence in text:
            embedding_sentence = []
            for word in sentence:
                if word in self.data:
                    embedding_word = self.data[word]
                else:
                    embedding_word = np.random.normal(0, 1, self.d)
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
        return embeddings
