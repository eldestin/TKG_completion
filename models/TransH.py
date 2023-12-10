import torch
from torch import nn
from utils import *
class TransH(nn.Module):
    def __init__(self, num_ent, num_rel, embd_dim):
        '''
        Params:
            1. num_ent: entity number
            2. num_rel: relation number
            3. embd_dim: embedding dimension
            4. margin: margin ranking loss hyperparameter
            5. norm: normalization type
            6. C: Lagrange multiplier unconstraint importance
        '''
        super(TransH,self).__init__()
        self.num_entity = num_ent
        self.num_rel = num_rel
        self.embd_dim = embd_dim
        self.ent_embd = nn.Embedding(num_ent, embd_dim)
        self.rel_embd = nn.Embedding(num_rel, embd_dim)
        self.rel_Hyper = nn.Embedding(num_rel, embd_dim)
        self.init_params(self.ent_embd.weight.data)
        self.init_params(self.rel_embd.weight.data)
        self.init_params(self.rel_Hyper.weight.data)
    def score_func(self, h,r,t):
        return torch.norm(h + r - t, dim = 1, keepdim = True)
    
    def init_params(self, input_):
        '''
        This function initialize the embedding matrix as paper described
        '''
        nn.init.xavier_uniform_(input_.data)
    def forward(self, pos, neg):
        pos_head, pos_relation, pos_tail, _= pos
        neg_head, neg_relation, neg_tail, _= neg
        pos_head_e, pos_rel_e, pos_tail_e, pos_rel_hyper_e = self.ent_embd(pos_head), self.rel_embd(pos_relation), self.ent_embd(pos_tail),\
                                                             self.rel_Hyper(pos_relation)
        #print(pos_rel_e.shape)
        # Project head, tail embedding into relation hyperplane
        # Formula: entity - w^T @ entity * W 
        pos_head_p = pos_head_e - pos_rel_hyper_e * torch.sum(pos_rel_hyper_e * pos_head_e, dim = 1, keepdim = True)
        pos_tail_p = pos_tail_e - pos_rel_hyper_e * torch.sum(pos_rel_hyper_e * pos_tail_e, dim = 1, keepdim = True)
        #print(pos_head_p + pos_rel_e - pos_tail_p)
        pos_score = self.score_func(pos_head_p, pos_rel_e, pos_tail_p)
        # For negative part
        neg_head_e, neg_rel_e, neg_tail_e, neg_rel_hyper_e = self.ent_embd(neg_head), self.rel_embd(neg_relation), self.ent_embd(neg_tail),\
                                                             self.rel_Hyper(neg_relation)
        # Project head, tail embedding into relation hyperplane
        # Formula: entity - w^T @ entity * W 
        neg_head_p = neg_head_e - neg_rel_hyper_e * torch.sum(neg_rel_hyper_e * neg_head_e, dim = 1, keepdim = True)
        neg_tail_p = neg_tail_e - neg_rel_hyper_e * torch.sum(neg_rel_hyper_e * neg_tail_e, dim = 1, keepdim = True)
        neg_score = self.score_func(neg_head_p, neg_rel_e, neg_tail_p)
        return pos_score, neg_score, self.ent_embd, self.rel_embd, self.rel_Hyper
    def predict(self, pos, label,k=10):
        h,r,t, _ = pos
        bs = h.shape[0]
        ent_ids = torch.arange(end = self.num_entity, device = h.device).unsqueeze(0)
        all_ent = ent_ids.repeat(h.shape[0], 1)
        hs = h.reshape(-1,1).repeat(1, all_ent.size()[1])
        rs = r.reshape(-1,1).repeat(1, all_ent.size()[1])
        ts = t.reshape(-1,1).repeat(1, all_ent.size()[1])
        triplets = torch.stack((hs, rs, all_ent), dim=2).reshape(-1, 3)
        h_,r_,t_ = triplets[:,0], triplets[:,1], triplets[:,2]
        pos_head_e, pos_rel_e, pos_tail_e, pos_rel_hyper_e = self.ent_embd(h_), self.rel_embd(r_), self.ent_embd(t_),\
                                                             self.rel_Hyper(r_)
        pos_head_p = pos_head_e - pos_rel_hyper_e * torch.sum(pos_rel_hyper_e * pos_head_e, dim = 1, keepdim = True)
        #obj_embd = pos_head_p + pos_rel_e
        pos_tail_p = pos_tail_e - pos_rel_hyper_e * torch.sum(pos_rel_hyper_e * pos_tail_e, dim = 1, keepdim = True)       
        model_predict = self.score_func(pos_head_p, pos_rel_e, pos_tail_p)
        tail_predict = model_predict.reshape(bs, -1)
        prediction = tail_predict
        h,r,t,_ = pos
        gt_id = t.reshape(-1,1)
        hit_10 = hit_at_k(prediction, gt_id)
        hit_3 = hit_at_k(prediction, gt_id, k=3)
        hit_1 = hit_at_k(prediction, gt_id, k=1)
        mrr_ = mrr(prediction, gt_id)
        return hit_10, hit_3, hit_1, mrr_
class Lagrange_loss(nn.Module):
    def __init__(self, margin, norm = 2, eps = 0.001, C = 1):
        '''
        This is the Loss function of TransE loss
        '''
        super(Lagrange_loss, self).__init__()
        self.margin = margin
        self.norm = norm
        self.eps = eps
        self.C = C
        print("Initialize loss successfully!")
    def forward(self, pos_dis, neg_dis, ent_embd, rel_embd, rel_hyper_embd):
        # 过relu因为其只sum positive part
        margin_loss = torch.sum(torch.relu(self.margin + torch.norm(pos_dis, p = self.norm, dim = 1) - \
                torch.norm(neg_dis, p = self.norm, dim = 1)))
        # First constraint
        entity_loss = torch.sum(torch.relu(torch.norm(ent_embd.weight.data, p = self.norm, dim = 1, keepdim = False) - 1))
        # Second constraint
        loss_2 = torch.sum(torch.relu(torch.sum(rel_embd.weight * rel_hyper_embd.weight, dim = 1, keepdim = False)/ \
                                      torch.norm(rel_embd.weight, p = self.norm, dim = 1,keepdim = False) - self.eps**2))
        return margin_loss + self.C*(entity_loss + loss_2)