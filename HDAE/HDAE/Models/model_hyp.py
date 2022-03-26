from torch import nn, rsub
import torch.nn.functional as F
from torch.nn.init import xavier_uniform
# from transformers import AutoModel
import torch
import h5py
import numpy as np
import manifolds

class Student_Hyp(nn.Module):
    def __init__(self, hparams) -> None:
        super(Student_Hyp, self).__init__()

        self.manifold = getattr(manifolds, "PoincareBall")()

        self.hparams = hparams
        # self.bert = AutoModel.from_pretrained(hparams['pretrained'])
        # self.fc = nn.Sequential(
            # nn.Linear(hparams['pretrained_dim'], hparams['num_aspect']))
        self.init()

    def init(self):

        id2word = {}
        word2id = {}
        data_file = './data/preprocessed/' + self.hparams.dataset + '_MATE'
        fvoc = open(data_file + '_word_mapping.txt', 'r')
        for line in fvoc:
            word, id = line.split()
            id2word[int(id)] = word
            word2id[word] = int(id)
        fvoc.close()

        h5py_file = './data/preprocessed/' + self.hparams.dataset + '_MATE'
        f = h5py.File(h5py_file + '.hdf5', 'r')
        
        w_emb_array = f['w2v'][()]
        w_emb_array = w_emb_array * 0.1
        w_emb = torch.from_numpy(w_emb_array)
        vocab_size, emb_size = w_emb.size()

        aspect_seeds_file = self.hparams.aspect_seeds

        fseed = open(aspect_seeds_file, 'r')
        aspects_ids = []

        seed_weights = []

        for line in fseed:
            seeds = []
            weights = []
            for tok in line.split():
                word, weight = tok.split(':')
                if word in word2id:
                    seeds.append(word2id[word])
                    weights.append(float(weight))
                else:
                    seeds.append(0)
                    weights.append(0.0)
            aspects_ids.append(seeds)
            seed_weights.append(weights)

        fseed.close()

        seed_w = torch.Tensor(seed_weights)
        seed_w /= seed_w.norm(p=1, dim=1, keepdim=True)

        clouds = []
        for seeds in aspects_ids:
            clouds.append(w_emb_array[seeds])
        a_emb = torch.from_numpy(np.array(clouds))

        self.lookup = nn.Embedding(vocab_size, emb_size)
        self.lookup.weight.data.copy_(w_emb)
        self.lookup.weight.requires_grad = True

        self.a_emb = nn.Parameter(torch.Tensor(a_emb.size()))
        self.a_emb.data.copy_(a_emb)
        self.a_emb.requires_grad = True


        num_aspect, self.num_seed = seed_w.size()
        print('seed_w.size() = ', seed_w.size())

        self.seed_w = nn.Parameter(torch.Tensor(seed_w.size()))
        self.seed_w.data.copy_(seed_w)
        self.seed_w.requires_grad = False

        self.lin = nn.Linear(emb_size, self.hparams.st_num_aspect)
        self.softmax = nn.Softmax(dim=1)

        self.M = nn.Parameter(torch.Tensor(emb_size, emb_size))
        xavier_uniform(self.M.data)

    def encoder(self, inputs):

        x_wrd = self.lookup(inputs)
        x_avg = x_wrd.mean(dim=1)

        x = x_wrd.matmul(self.M)
        x = x.matmul(x_avg.unsqueeze(1).transpose(1,2))

        a = F.softmax(x, dim=1)
        z = a.transpose(1,2).matmul(x_wrd)
        z = z.squeeze()

        if z.dim() == 1:
            return z.unsqueeze(0)

        return z
        
    def forward(self, x):

        enc = self.encoder(x)
 
        bsz, dim = enc.shape[0], enc.shape[1]

        enc = self.manifold.proj_tan0_exp(enc.view(-1, dim), 1)

        enc_aspects = (enc.unsqueeze(1)).repeat(1,  self.hparams.st_num_aspect, 1)

        bz_a_emb_w = self.a_emb.mul(self.seed_w.view(1,  self.hparams.st_num_aspect, self.num_seed, 1))
        
        bz_a_emb_w = bz_a_emb_w.sum(dim=2)

        bz_a_emb_w = bz_a_emb_w.repeat(bsz, 1,1)

        enc_aspects_2 = (enc.unsqueeze(1)).repeat(1, self.hparams.st_num_aspect,1)

        hyp_a_w = self.manifold.proj_tan0_exp(bz_a_emb_w.view(-1, dim), 1)

        probs_pre = self.manifold.sqdist(enc_aspects_2.view(-1, dim), hyp_a_w.view(-1, dim), 1)

        probs_pre = probs_pre.view(-1,1)

        probs_exp = torch.exp(- 0.02 * probs_pre - 0.05)

        probs_exp = probs_exp.view(bsz, self.hparams.st_num_aspect)

        a_probs = (probs_exp / (probs_exp.sum(1).unsqueeze(1))).view(bsz, self.hparams.st_num_aspect)

        return a_probs


