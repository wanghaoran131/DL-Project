import json

import numpy as np
import torch
import re
import h5py
from sklearn.feature_extraction.text import CountVectorizer
from torch.utils import data
from transformers import BertTokenizerFast
from numpy.random import permutation, seed
import random
from torch.autograd import Variable
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from random import shuffle

class Dataset(data.Dataset):
    def __init__(self, hparams, aspect_init_file, maxlen=10):
        self.aspects, vocab = self.load_aspect_init(aspect_init_file)
        self.hparams = hparams

        self.vector_list = []
        for seeds in self.aspects:
            seeds = list(set(seeds))
            while len(seeds) < 30: seeds.append(seeds[-1]+'-')

            cv = CountVectorizer(vocabulary=sorted(seeds))
            cv.fixed_vocabulary_ = True
            self.vector_list.append(cv)

        self.vectorizer = CountVectorizer(vocabulary=sorted(list(set(vocab))))
        self.vectorizer.fixed_vocabulary_ = True
        self.maxlen = maxlen
        self.id2asp = {idx: feat for idx, feat in enumerate(
            self.vectorizer.get_feature_names())}
        self.asp2id = {feat: idx for idx, feat in enumerate(
            self.vectorizer.get_feature_names())}
        self.aspect_ids = [[self.asp2id[asp] for asp in aspect] for aspect in self.aspects]

        self.init_train()

    def init_train(self):

        self.id2word = {}
        self.word2id = {}

        fvoc = open('./data/preprocessed/' + self.hparams.dataset + '_MATE_word_mapping.txt', 'r')
        for line in fvoc:
            word, id = line.split()
            self.id2word[int(id)] = word
            self.word2id[word] = int(id)
        fvoc.close()

        f = h5py.File('./data/preprocessed/' + self.hparams.dataset + '_MATE' + '.hdf5', 'r')
        self.bz_batches = []
        self.bz_original = []
        self.bz_scodes = []
        for b in f['data']:
            if Variable(torch.from_numpy(f['data/' +  b][()]).long()).shape[0] == 1: continue
            self.bz_batches.append(Variable(torch.from_numpy(f['data/' +  b][()]).long()))
            self.bz_original.append(list(f['original/' + b][()]))
            self.bz_scodes.append(list(f['scodes/' + b][()]))
        f.close()

        self.batches = []
        self.original = []
        self.scodes = []
        index_shuf = list(range(len(self.bz_batches)))
        shuffle(index_shuf)
        for i in index_shuf:
            self.batches.append(self.bz_batches[i])
            self.original.append(self.bz_original[i])
            self.scodes.append(self.bz_scodes[i])

    def get_idx2asp(self):
        """idx2asp

        Returns:
            : bow_size
        """
        result = []
        for feat in self.vectorizer.get_feature_names():
            for i in range(len(self.aspects)):
                if feat in self.aspects[i]:
                    result.append(i)
                    break
        return result

    @staticmethod
    def load_data(file):
        with open(file) as f:
            data = json.load(f)
        data = [s for d in data['original'] for s in d]
        return data

    @staticmethod
    def load_aspect_init(file):
        with open(file) as f:
            text = f.read()
        text = text.strip().split('\n')
        result = [t.strip().split() for t in text]
        return result, [i for r in result for i in r]

    def __getitem__(self, index: int):

        bows = []
        for ids in self.batches[index]:
            temp = []
            sentence = ' '.join(self.id2word[int(id)] for id in ids)
            for cv in self.vector_list:
                temp.append(cv.transform([sentence]).toarray()[0])
            bows.append(temp)

        bows = np.array(bows)
        idx = self.batches[index]

        return torch.from_numpy(bows), torch.LongTensor(idx), [self.original[index]]

    def __len__(self):
        return len(self.batches)


class TestDataset(Dataset):
    def __init__(self, hparams, aspect_init_file, maxlen=10):
        super(TestDataset, self).__init__(hparams, aspect_init_file, maxlen)

    def init_train(self):

        self.id2word = {}
        self.word2id = {}

        fvoc = open('./data/preprocessed/' + self.hparams.dataset + '_MATE_word_mapping.txt', 'r')
        for line in fvoc:
            word, id = line.split()
            self.id2word[int(id)] = word
            self.word2id[word] = int(id)
        fvoc.close()

        self.bz_test_batches = []
        self.bz_test_labels = []
        self.bz_test_original = []
        self.bz_test_scodes = []
        f = h5py.File("./data/preprocessed/" + self.hparams.dataset + "_MATE_TEST" + '.hdf5', 'r')
        for b in f['data']:
            self.bz_test_batches.append(Variable(torch.from_numpy(f['data/' +  b][()]).long()))
            self.bz_test_labels.append(Variable(torch.from_numpy(f['labels/' +  b][()]).long()))
            self.bz_test_original.append(list(f['original/' + b][()]))
            self.bz_test_scodes.append(list(f['scodes/' + b][()]))
        f.close()

        self.batches = []
        self.labels = []
        self.original = []
        self.scodes = []
        index_shuf = list(range(len(self.bz_test_batches)))
        shuffle(index_shuf)
        for i in index_shuf:
            self.batches.append(self.bz_test_batches[i])
            self.labels.append(self.bz_test_labels[i])
            self.original.append(self.bz_test_original[i])
            self.scodes.append(self.bz_test_scodes[i])
        
    def __getitem__(self, index: int):

        bows = []
        for ids in self.batches[index]:
            temp = []
            sentence = ' '.join(self.id2word[int(id)] for id in ids)
            for cv in self.vector_list:
                temp.append(cv.transform([sentence]).toarray()[0])
            bows.append(temp)

        bows = np.array(bows)
        idx = self.batches[index]

        return torch.from_numpy(bows), torch.LongTensor(idx), self.labels[index], [self.original[index]]
