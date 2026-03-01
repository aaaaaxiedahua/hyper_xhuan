import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from models import GNNModel, expmap0, hyp_distance
from utils import cal_ranks, cal_performance

class BaseModel(object):
    def __init__(self, args, loader):
        self.model = GNNModel(args, loader)
        self.model.cuda()
        self.loader = loader
        self.n_ent = loader.n_ent
        self.n_ent_ind = loader.n_ent_ind
        self.n_batch = args.n_batch
        self.n_train = loader.n_train
        self.n_valid = loader.n_valid
        self.n_test  = loader.n_test
        self.n_layer = args.n_layer
        self.optimizer = Adam(self.model.parameters(), lr=args.lr, weight_decay=args.lamb)
        self.scheduler = ExponentialLR(self.optimizer, args.decay_rate)
        self.smooth = 1e-5
        self.params = args
        self.use_hyp_cl = getattr(args, 'use_hyp_cl', False)
        self.lambda_cl = getattr(args, 'lambda_cl', 0.1)
        self.tau_cl = getattr(args, 'tau_cl', 0.5)

    def train_batch(self,):
        epoch_loss = 0
        i = 0
        batch_size = self.n_batch
        n_batch = self.n_train // batch_size + (self.n_train % batch_size > 0)
        self.model.train()
        self.time_1 = 0
        self.time_2 = 0
        
        for i in range(n_batch):
            start = i*batch_size
            end = min(self.n_train, (i+1)*batch_size)
            batch_idx = np.arange(start, end)
            triple = self.loader.get_batch(batch_idx)

            self.model.zero_grad()
            scores = self.model(triple[:,0], triple[:,1])
            pos_scores = scores[[torch.arange(len(scores)).cuda(),torch.LongTensor(triple[:,2]).cuda()]]
            self.time_1 += self.model.time_1
            self.time_2 += self.model.time_2
            t_2 = time.time()

            max_n = torch.max(scores, 1, keepdim=True)[0]
            loss = torch.sum(- pos_scores + max_n + torch.log(torch.sum(torch.exp(scores - max_n),1)))

            if self.use_hyp_cl:
                loss_cl = self.contrastive_loss(triple)
                loss = loss + self.lambda_cl * loss_cl

            loss.backward()
            self.optimizer.step()
            self.time_2 += time.time() - t_2

            for p in self.model.parameters():
                X = p.data.clone()
                flag = X != X
                X[flag] = np.random.random()
                p.data.copy_(X)
            epoch_loss += loss.item()
            
        self.loader.shuffle_train()
        self.scheduler.step()
        valid_mrr, test_mrr, out_str = self.evaluate()
        return valid_mrr, test_mrr, out_str

    def contrastive_loss(self, triple):
        hidden = self.model.cl_hidden   # [n_node, d]
        nodes = self.model.cl_nodes     # [n_node, 2] (batch_idx, entity_id)
        c = self.model.layers[-1].curvature
        n = len(triple)

        batch_ids = nodes[:, 0]
        ent_ids = nodes[:, 1]
        query_ents = torch.LongTensor(triple[:, 0]).cuda()
        pos_ents = torch.LongTensor(triple[:, 2]).cuda()

        h_hyp = expmap0(hidden, c)

        anchor_idx = []
        pos_idx = []
        for i in range(n):
            a_mask = (batch_ids == i) & (ent_ids == query_ents[i])
            p_mask = (batch_ids == i) & (ent_ids == pos_ents[i])
            if a_mask.any() and p_mask.any():
                anchor_idx.append(a_mask.nonzero()[0, 0])
                pos_idx.append(p_mask.nonzero()[0, 0])

        if len(anchor_idx) == 0:
            return torch.tensor(0.0).cuda()

        anchor_idx = torch.stack(anchor_idx)
        pos_idx = torch.stack(pos_idx)
        n_valid = len(anchor_idx)

        anchor_hyp = h_hyp[anchor_idx]
        pos_hyp = h_hyp[pos_idx]

        pos_dist = hyp_distance(anchor_hyp, pos_hyp, c)  # [n_valid, 1]

        k_neg = 32
        neg_idx = torch.randint(0, len(hidden), (n_valid, k_neg)).cuda()
        neg_hyp = h_hyp[neg_idx]  # [n_valid, k_neg, d]

        anchor_exp = anchor_hyp.unsqueeze(1).expand(-1, k_neg, -1)
        neg_dist = hyp_distance(
            anchor_exp.reshape(-1, anchor_hyp.size(-1)),
            neg_hyp.reshape(-1, anchor_hyp.size(-1)),
            c
        ).reshape(n_valid, k_neg)  # [n_valid, k_neg]

        logits = torch.cat([-pos_dist / self.tau_cl, -neg_dist / self.tau_cl], dim=1)
        labels = torch.zeros(n_valid, dtype=torch.long).cuda()
        loss_cl = F.cross_entropy(logits, labels)

        return loss_cl

    def evaluate(self, ):
        batch_size = self.n_batch
        n_data = self.n_valid
        n_batch = n_data // batch_size + (n_data % batch_size > 0)
        ranking = []
        masks = []
        self.model.eval()
        time_3 = time.time()
        for i in range(n_batch):
            start = i*batch_size
            end = min(n_data, (i+1)*batch_size)
            batch_idx = np.arange(start, end)
            subs, rels, objs = self.loader.get_batch(batch_idx, data='valid')
            scores = self.model(subs, rels).data.cpu().numpy()
            filters = []
            for i in range(len(subs)):
                filt = self.loader.val_filters[(subs[i], rels[i])]
                filt_1hot = np.zeros((self.n_ent, ))
                filt_1hot[np.array(filt)] = 1
                filters.append(filt_1hot)

                masks += [self.n_ent - len(filt)] * int(objs[i].sum())
             
            filters = np.array(filters)
            ranks = cal_ranks(scores, objs, filters)
            ranking += ranks
            
        ranking = np.array(ranking)
        v_mrr, v_mr, v_h1, v_h3, v_h10, v_h1050 = cal_performance(ranking, masks)

        n_data = self.n_test
        n_batch = n_data // batch_size + (n_data % batch_size > 0)
        ranking = []
        masks = []
        self.model.eval()
        for i in range(n_batch):
            start = i*batch_size
            end = min(n_data, (i+1)*batch_size)
            batch_idx = np.arange(start, end)
            subs, rels, objs = self.loader.get_batch(batch_idx, data='test')
            scores = self.model(subs, rels, 'inductive').data.cpu().numpy()
            filters = []
            for i in range(len(subs)):
                filt = self.loader.tst_filters[(subs[i], rels[i])]
                filt_1hot = np.zeros((self.n_ent_ind, ))
                filt_1hot[np.array(filt)] = 1
                filters.append(filt_1hot)
                masks += [self.n_ent_ind - len(filt)] * int(objs[i].sum())
             
            filters = np.array(filters)
            ranks = cal_ranks(scores, objs, filters)
            ranking += ranks
        ranking = np.array(ranking)
        t_mrr, t_mr, t_h1, t_h3, t_h10, t_h1050 = cal_performance(ranking, masks)
        time_3 = time.time() - time_3
        
        out_str = '%.4f %.4f %.4f\t%.4f %.1f %.4f %.4f %.4f %.4f\t\t%.4f %.1f %.4f %.4f %.4f %.4f\n'%(self.time_1, self.time_2, time_3, v_mrr, v_mr, v_h1, v_h3, v_h10, v_h1050, t_mrr, t_mr, t_h1, t_h3, t_h10, t_h1050)
        return v_h10, t_h10, out_str
    