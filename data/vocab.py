import pickle

class Vocab:
    def __init__(self, text):
        self._word_to_id = {}
        self._id_to_word = {}
        self._word_to_id['<pad>'] = 0
        self._id_to_word[0] = '<pad>'
        self._word_to_id['<unk>'] = 1
        self._id_to_word[1] = '<unk>'
        self._count = 2
        for line in text:
            for word in line:
                if word not in self._word_to_id:
                    self._word_to_id[word] = self._count
                    self._id_to_word[self._count] = word
                    self._count += 1
        
                

    def word_to_id(self, word):
        if word not in self._word_to_id:
            return self._word_to_id['<unk>']
        return self._word_to_id[word]

    def id_to_word(self, word_id):
        if word_id not in self._id_to_word:
            raise ValueError('Id not found in vocab: %d' % word_id)
        return self._id_to_word[word_id]

    def size(self):
        return self._count

    @staticmethod
    def load_vocab(vocab_file):
        with open(vocab_file, 'rb') as f:
            return pickle.load(f)
    
    @staticmethod
    def dump_vocab(vocab_file, vocab):
        with open(vocab_file, 'wb') as f:
            pickle.dump(vocab, f)
    

    