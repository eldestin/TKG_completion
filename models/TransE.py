from utils import *
import math
class TransE(nn.Module):
    def __init__(self, num_entity, num_relation, embed_dim):
        '''
        This is the torch - TransE implementation.
        Params:
            1. num_entity: entity number of dataset
            2. num_relation: relation number of dataset
            3. embedding dimension: hidden dimension
            4. norm: dissimilarity measurement: whether L1 norm or L2 norm
        '''
        super(TransE,self).__init__()
        # Initialize embedding matrix
        self.num_entity = num_entity
        self.num_relation = num_relation
        self.ent_embd = nn.Embedding(num_entity, embed_dim)
        self.rel_embd = nn.Embedding(num_relation, embed_dim)
        self.embed_dim = embed_dim
        # initialize parameter
        self.init_params(self.ent_embd.weight.data)
        self.init_params(self.rel_embd.weight.data)
        # normalization
        # normalize relation 
        rel_norm = torch.norm(self.rel_embd.weight.data, dim = 1, keepdim = True)
        self.rel_embd.weight.data /= rel_norm
    def init_params(self, input_):
        '''
        This function initialize the embedding matrix as paper described
        '''
        nn.init.uniform_(input_, -6/math.sqrt(self.embed_dim), 6/math.sqrt(self.embed_dim))
    
    def forward(self, pos, neg):
        '''
        Params:
            Pos_head, rel, tail [batch_size]
            Neg_head, rel, tail [batch_size]
        '''
        pos_head, pos_relation, pos_tail,_ = pos
        neg_head, neg_relation, neg_tail,_ = neg
        pos_distance = self.ent_embd(pos_head) + self.rel_embd(pos_relation) - self.ent_embd(pos_tail)
        neg_distance = self.ent_embd(neg_head) + self.rel_embd(neg_relation) - self.ent_embd(neg_tail)
        
        return pos_distance, neg_distance
    def predict(self, pos, label, k=10):
        h,r,t, _ = pos
        bs = h.shape[0]
        ent_ids = torch.arange(end = self.num_entity, device = h.device).unsqueeze(0)
        all_ent = ent_ids.repeat(h.shape[0], 1)
        hs = h.reshape(-1,1).repeat(1, all_ent.size()[1])
        rs = r.reshape(-1,1).repeat(1, all_ent.size()[1])
        ts = t.reshape(-1,1).repeat(1, all_ent.size()[1])
        triplets = torch.stack((hs, rs, all_ent), dim=2).reshape(-1, 3)
        h_,r_,t_ = triplets[:,0], triplets[:,1], triplets[:,2]
        obj_embd = self.ent_embd(h_) + self.rel_embd(r_)
        model_predict = torch.norm(obj_embd-self.ent_embd(t_), p = 2, dim = 1)
        tail_predict = model_predict.reshape(bs, -1)
        prediction = tail_predict
#         print(prediction.shape)
        h,r,t,_ = pos
#         gt_id = torch.cat((t.reshape(-1,1), h.reshape(-1,1)),dim = 0)     
        gt_id = t.reshape(-1,1)
#         print(gt_id.shape)
        hit_10 = hit_at_k(prediction, gt_id)
        hit_3 = hit_at_k(prediction, gt_id, k=3)
        hit_1 = hit_at_k(prediction, gt_id, k=1)
        mrr_ = mrr(prediction, gt_id)
#         print(hit)
        return hit_10, hit_3, hit_1, mrr_


class Rank_loss(nn.Module):
    def __init__(self, margin = 1, norm = 2):
        '''
        This is the Loss function of TransE loss
        '''
        super(Rank_loss, self).__init__()
        self.margin = margin
        self.norm = norm
        print("Initialize loss successfully!")
    def forward(self, pos_dis, neg_dis):
        # 过relu因为其只sum positive part
        return torch.sum(torch.relu(self.margin + torch.norm(pos_dis, p = self.norm, dim = 1) - \
                torch.norm(neg_dis, p = self.norm, dim = 1)))