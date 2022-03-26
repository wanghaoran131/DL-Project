import json
import logging
from os import name
import time
import torch
import os

from torch import nn, optim
from torch.utils import data
from torch.utils.data import sampler
from tqdm import tqdm

from Models.get_model import *
from Dataset.get_dataset import *
from loss import TripletMarginCosineLoss, OrthogonalityLoss, TripletMarginCosineLoss_hyperdis
from sklearn.metrics import f1_score

torch.backends.cudnn.benchmark = True
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.basicConfig(level=logging.INFO)
torch.set_printoptions(profile="full")
logging.getLogger('transformers').setLevel(logging.WARNING)
orth_loss = OrthogonalityLoss()
rec_loss = TripletMarginCosineLoss()
rec_loss_dis = TripletMarginCosineLoss_hyperdis()

class EntropyLoss(nn.Module):
    def __init__(self):
        super(EntropyLoss, self).__init__()

    def forward(self, p, q):
        b = q * torch.log(p)
        b = -1. * b.sum()
        return b

class Trainer:
    def __init__(self, hparams) -> None:
        self.hparams = hparams
        # print('self.hparams = ', self.hparams)
        # input()
        self.asp_cnt = self.hparams.st_num_aspect
        self.start = time.time()
        logging.info('loading dataset...')

        self.ds = Dataset(hparams, hparams.aspect_init_file, hparams.train_file,
                          hparams.st_pre, hparams.maxlen)
        test_ds = TestDataset(
            hparams, hparams.aspect_init_file, hparams.test_file)
        self.test_loader = data.DataLoader(test_ds, batch_size=1, num_workers=1)

        logging.info(f'dataset_size: {len(self.ds)}')

        logging.info('loading model...')
        self.idx2asp = torch.tensor(self.ds.get_idx2asp()).cuda()

        self.teacher = Teacher(self.idx2asp, self.asp_cnt,
                               self.hparams.general_asp).cuda()
        self.student = Student(hparams).cuda()
        params = filter(lambda p: p.requires_grad, self.student.parameters())
        # self.student_opt = optim.Adam(self.student.parameters(), hparams.lr, weight_decay=1e-5)
        self.student_opt = torch.optim.Adam(params, lr = hparams.lr)

        self.criterion = EntropyLoss().cuda()

        self.z = self.reset_z()
        logging.debug(f'__init__: {time.time() - self.start}')

        self.tolorence = 3
        self.early_stop = 0
        self.early_stop_criteria = 5

    def save_model(self, path, model_name):
        if not os.path.exists(path):
            os.mkdir(path)
        with open(os.path.join(path, 'config.json'), 'w') as f:
            json.dump(self.hparams, f)
        torch.save({'teacher_z': self.z, 'student': self.student.state_dict()}, os.path.join(
            path, f'epoch_{model_name}_student.pt'))

    def train_loader(self, ds):
        # sampler = data.RandomSampler(ds, replacement=True, num_samples=10000)
        return data.DataLoader(ds, batch_size=1, num_workers=1)

    def train(self):
        prev_best = 0
        # input()
        loader = self.train_loader(self.ds)
        # input()
        # for epoch in range(self.hparams.epoch):
        for epoch in range(30):
            logging.info(f'Epoch: {epoch}')
            loss = self.train_per_epoch(loader)
            score = self.test(epoch)
            logging.info(f'epoch: {epoch}, f1_mid: {score:.3f}, prev_best: {prev_best:.3f}')

            if prev_best < score:
                self.early_stop = 0
                prev_best = score
                # self.save_model(self.hparams.save_dir, f'{epoch}')
            else:
                print('prev_best = ', prev_best)
                print('score = ', score)
                if epoch >= self.tolorence:
                    self.early_stop += 1
                    if self.early_stop == self.early_stop_criteria: break


    def reset_z(self):
        z = torch.ones(
            (self.asp_cnt, 30)).cuda()
        return z / z.sum(-1).unsqueeze(-1)

    def train_per_epoch(self, loader):

        s_logits_tmp = []
        x_bow_tmp = []
        losses = []
        pbar = tqdm(total=len(loader))
        bt_times = 0
        # k = 0
        self.z = self.reset_z()

        for x_bow, x_id, ori in loader:

            x_bow, x_id, ori = x_bow[0,:,:], x_id[0,:,:], ori
            x_bow, x_id = x_bow.cuda(), x_id.cuda()
            loss, s_logits, x_bow = self.train_step(x_bow, x_id, ori, bt_times)

            s_logits_tmp.append(s_logits)
            x_bow_tmp.append(x_bow)

            if len(s_logits_tmp) == 1000:
                # self.z = self.calc_z(torch.cat(s_logits_tmp, 0), torch.cat(x_bow_tmp, 0))
                s_logits_tmp = []
                x_bow_tmp = []

            bt_times += 1
            losses.append(loss.item())
            print('bt total = ', bt_times, len(loader), end='\r')

            # if bt_times == 100:
            #     break

        pbar.close()
        losses = sum(losses) / len(losses)
        logging.info(f'train_loss: {losses}')
        return losses
    
    def train_step(self, x_bow, x_id, x_ori, bt_times):
        # apply teacher


        if self.hparams.student_type[:len('hyper_rec_dis')] == 'hyper_rec_dis' and bt_times % 20 == 0:
            for g in self.student_opt.param_groups:
                g['lr'] = self.hparams.lr * 2

            # print('tmp pause ')
            # input()

            for _ in range(10):
                loss_2 = self.student.intra_aspect_model()
                self.student_opt.zero_grad()
                loss_2.backward()
                self.student_opt.step()

            for g in self.student_opt.param_groups:
                g['lr'] = self.hparams.lr


        t_logits = self.teacher(x_bow, self.z)
        loss = 0.
        prev = -1

        # train student Eq. 2
        self.student_opt.zero_grad()

        eu_r, _, s_logits = self.student(x_id)

        positives, negatives = self.student.get_targets()

        loss = rec_loss(eu_r, positives, negatives)

        loss += self.hparams.mt_ratio * self.criterion(s_logits, t_logits)

        aspects = self.student.get_aspects()

        loss += 0.05 * orth_loss(aspects)

        if bt_times == 1000:
            print(f'teacher:{t_logits.max(-1)[1]}')
            print(f'student:{s_logits.max(-1)[1]}')

        loss.backward()
        self.student_opt.step()
        tmp = (t_logits.max(-1)[1] == s_logits.max(-1)[1]).sum()
        prev = tmp
        # update teacher Eq.4

        # apply teacher Eq. 3
        t_logits = self.teacher(x_bow, self.z)

        return loss, s_logits, x_bow

    def test(self, epoch):
        ref = []
        pred = []
        score = []
        result = []
        ground = []
        self.student.eval()
        # return torch.from_numpy(bosw), torch.LongTensor(idx), self.labels[index], self.original[index]
        for batch in self.test_loader:
            bow, idx, labels, ori = batch
            # print('bow = ', bow.shape)
            # input() 
            bow, idx, labels, ori = bow[0,:,:,:], idx[0,:,:], labels[0,:,:], ori

            bow = bow.cuda()
            idx = idx.cuda()

            with torch.no_grad():
                _, _, logits = self.student(idx)
                # logits = self.student(idx)
            
            lg = logits.max(-1)[1].detach().cpu().tolist()
            
            for lg_i in range(len(lg)):
                asps = labels[lg_i].tolist()

                # for as_i in range(self.hparams.st_num_aspect):
                #     if asps[as_i] == 1:
                #         ref.append(1)
                #     else:
                #         ref.append(0)
                #     if lg[lg_i] == as_i:
                #         pred.append(1)
                #     else:
                #         pred.append(0)
                                    # for lg_i in range(len(lg)):
                pred.append(lg[lg_i])
                if asps[lg[lg_i]] == 1:
                    ref.append(lg[lg_i])
                else:
                    for as_i in range(self.hparams.st_num_aspect):
                        if asps[as_i] == 1:
                            ref.append(as_i)
                            break

        score = f1_score(y_true=ref, y_pred=pred, average='micro')
        print('f1 = ',score)

        eva_file = open(self.hparams.sumout, "a")
        eva_result = 'f1 score ={}'.format(score)
        eva_file.write('Epoch = ' + str(epoch) + ' ,' + str(eva_result) + '\n')
        eva_file.write('\n')
        eva_file.close()

        eva_file = open(self.hparams.sumout_z, "a")
        # eva_result = 'f1 score ={}'.format(f1)
        eva_file.write('Epoch = ' + str(epoch) + '\n')
        tmp = self.z.tolist()
        tmpz_2 = [[round(k, 3) for k in kk] for kk in tmp] 
        eva_file.write(str(tmpz_2))
        eva_file.write('\n')
        eva_file.close()
        
        self.student.train()
        return score

    def calc_z(self, logits, bow):
        """z
        Args:
            logits: B, asp_cnt
            bow: B, bow_size
        Returns:
            : asp_cnt, bow_size
        """
        val, trg = logits.max(1)
        num_asp = logits.shape[1]

        tmp_asp = []
        for k in range(num_asp):

            if len(bow[torch.where(trg == k)]) != 0:
                true_tmp_asp_z = bow[torch.where(trg == k)][:, k, :].float().sum(0)
            else:
                true_tmp_asp_z = torch.zeros(30).to(bow.device)

            tmp_asp_z = bow[:, k, :].float().sum(0)

            tmp_asp_z = tmp_asp_z.masked_fill(tmp_asp_z == 0., 1e-10)

            tmp_q = true_tmp_asp_z / tmp_asp_z

            tmp_asp.append(tmp_q.unsqueeze(0))

        tmp = torch.cat(tmp_asp, 0)
        tmp = tmp + 0.05
        tmp = tmp / tmp.sum(-1).unsqueeze(-1)

        return tmp

