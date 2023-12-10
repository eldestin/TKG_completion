import torch_scatter
from utils import *
from torch.nn import Parameter


class CompGcnBasis(nn.Module):
    nodes_dim = 0
    head_dim = 0
    tail_dim = 1
    def __init__(self, in_channels, out_channels, num_relations, num_basis_vector,act = torch.tanh,cache = True,dropout = 0.2, comp_type = "sub"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_basis_vector = num_basis_vector
        self.act = act
        self.device = None
        self.cache = cache
        self.comp_type = comp_type
        #----------- Creating learnable basis vector , shape is (num_basis, feature_size(in channel))
        self.basis_vector = get_param((num_basis_vector, in_channels))
        # this weight matrix initialize the weight features for each relation(including inverse), shape is (2*num_relations, num_basis)
        self.rel_weight = get_param((num_relations, self.num_basis_vector))
        # this learnable weight matrix is for projection, that project each relation to the same dimension of node_dimension
        self.weight_rel = get_param((in_channels,out_channels))
        # add another embedding for loop
        self.loop_rel = get_param((1,in_channels))
        #----------- Creating three updated matrix, as three kind of relations updating, in, out, loop
        # using for updating weight
        self.w_in = get_param((in_channels,out_channels))
        self.w_out = get_param((in_channels,out_channels))
        self.w_loop = get_param((in_channels,out_channels))
        
        # define some helpful parameter
        self.in_norm, self.out_norm = None, None
        self.in_index, self.out_index = None, None
        self.in_type, self.out_type = None, None
        self.loop_index, self.loop_type =None, None
        
        self.drop = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(out_channels)
    def relation_transform(self, entity_embedding, relation_embedding,type_):
        '''
        This function given entity embedding and relation embedding, in order return three types of 
        non-parameterized operations, which is subjection, corr, multiplication
        '''
        assert type_ in ["mul","sub","corr"], "not implemented now"
        if type_ == "mul":
            out = entity_embedding*relation_embedding
        elif type_ == "sub":
            out = entity_embedding - relation_embedding
        else:
            out = ccorr(entity_embedding,relation_embedding)
        return out
    
    def normalization(self, edge_index, num_entity):
        '''
        As normal GCN, this function calculate the normalization adj matrix 
        '''
        head, tail = edge_index
        edge_weight = torch.ones_like(head).float()
        degree = torch_scatter.scatter_add(edge_weight,head,dim_size=num_entity,dim = self.nodes_dim)
        degree_inv = degree.pow(-0.5)
        # if inf, in order to prevent nan in scatter function
        degree_inv[degree_inv == float("inf")] = 0
        norm = degree_inv[head] * edge_weight * degree_inv[tail]
        return norm
    def scatter_function(self,type_, src, index, dim_size = None):
        '''
        This function given scatter_ type, which should me max, mean,or sum, given source array, given index array, given dimension size
        '''
        assert type_.lower() in ["sum","mean","max"]
        return torch_scatter.scatter(src, index, dim=0,out=None,dim_size = dim_size, reduce= type_)
    
    def propogating_message(self, method, node_features,edge_index,edge_type, rel_embedding, edge_norm,mode,type_):
        '''
        This function done the basic aggregation
        '''
        assert method in ["sum", "mean", "max"]
        assert mode in ["in","out","loop"]
        size = node_features.shape[0]
        coresponding_weight = getattr(self, 'w_{}'.format(mode))
        #-------------- this index selection: given relation embedding and relation_basic representation, choose the inital basis vector part
        relation_embedding = torch.index_select(rel_embedding,dim = 0, index = edge_type)
        # ------------- using index of tail in edge index to represent head by relation
        node_features = node_features[edge_index[1]]
        out = self.relation_transform(node_features, relation_embedding,type_)
        out = torch.matmul(out,coresponding_weight)
        out = out if edge_norm is None else out * edge_norm.view(-1, 1)
        out = self.scatter_function(method,out,edge_index[0],  size)
        return out    
    def forward(self, nodes_features, edge_index,edge_type):
        '''
        Forward propogate function:
            Given input nodes_features, adj_matrix, relation_matrix
        '''
        if self.device is None:
            self.device = edge_index.device
        # ----------- First done the basis part, which means represent each relation using a vector space defining previously
        relation_embedding = torch.mm(self.rel_weight,self.basis_vector)
        # ----------- add a self-loop dimension
        relation_embedding = torch.cat([relation_embedding,self.loop_rel],dim = 0)
        # print(edge_index.shape)
        num_edges = edge_index.shape[1]//2
        num_nodes = nodes_features.shape[self.nodes_dim]
        if not self.cache or self.in_norm == None:
            #---------------- in represent in_relation, out represent out_relation
            self.in_index, self.out_index = edge_index[:,:num_edges], edge_index[:,num_edges:]
            self.in_type, self.out_type = edge_type[:num_edges], edge_type[num_edges:]
            # --------------- create self-loop part
            self.loop_index = torch.stack([torch.arange(num_nodes), torch.arange(num_nodes)]).to(self.device)
            self.loop_type = torch.full((num_nodes,), relation_embedding.shape[0]-1, dtype = torch.long).to(self.device)
            # -------------- create normalization part
            self.in_norm = self.normalization(self.in_index, num_nodes)
            self.out_norm = self.normalization(self.out_index, num_nodes)
        #print(self.in_norm.isinf().any())
        in_res = self.propogating_message('sum',nodes_features,self.in_index,self.in_type, relation_embedding,self.in_norm,"in",self.comp_type)
        loop_res = self.propogating_message('sum',nodes_features,self.loop_index,self.loop_type, relation_embedding,None,"loop",self.comp_type)
        out_res = self.propogating_message('sum',nodes_features,self.out_index,self.out_type, relation_embedding,self.out_norm,"out",self.comp_type)
        # I don't know why but source code done it
        out = self.drop(in_res)*(1/3) + self.drop(out_res)*(1/3) + loop_res*(1/3)
        # update the relation embedding
        out_2 = torch.matmul(relation_embedding,self.weight_rel)
        return self.act(out),out_2
            
            

class CompGcn_non_first_layer(nn.Module):
    nodes_dim = 0
    head_dim = 0
    tail_dim = 1
    def __init__(self, in_channels, out_channels, num_relations,act = torch.tanh,dropout = 0.2, comp_type = "sub"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.act = act
        self.device = None
        self.comp_type = comp_type
        # this learnable weight matrix is for projection, that project each relation to the same dimension of node_dimension
        self.weight_rel = get_param((in_channels,out_channels))
        # add another embedding for loop
        self.loop_rel = get_param((1,in_channels))
        #----------- Creating three updated matrix, as three kind of relations updating, in, out, loop
        # using for updating weight
        self.w_in = get_param((in_channels,out_channels))
        self.w_out = get_param((in_channels,out_channels))
        self.w_loop = get_param((in_channels,out_channels))
        self.drop = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(out_channels)
    def relation_transform(self, entity_embedding, relation_embedding,type_):
        '''
        This function given entity embedding and relation embedding, in order return three types of 
        non-parameterized operations, which is subjection, corr, multiplication
        '''
        assert type_ in ["mul","sub","corr"], "not implemented now"
        if type_ == "mul":
            out = entity_embedding*relation_embedding
        elif type_ == "sub":
            out = entity_embedding - relation_embedding
        else:
            out = ccorr(entity_embedding,relation_embedding)
        return out
    
    def normalization(self, edge_index, num_entity):
        '''
        As normal GCN, this function calculate the normalization adj matrix 
        '''
        head, tail = edge_index
        edge_weight = torch.ones_like(head).float()
        degree = torch_scatter.scatter_add(edge_weight,head,dim_size=num_entity,dim = self.nodes_dim)
        degree_inv = degree.pow(-0.5)
        # if inf, in order to prevent nan in scatter function
        degree_inv[degree_inv == float("inf")] = 0
        norm = degree_inv[head] * edge_weight * degree_inv[tail]
        return norm
    def scatter_function(self,type_, src, index, dim_size = None):
        '''
        This function given scatter_ type, which should me max, mean,or sum, given source array, given index array, given dimension size
        '''
        assert type_.lower() in ["sum","mean","max"]
        return torch_scatter.scatter(src, index, dim=0,out=None,dim_size = dim_size, reduce= type_)
    
    def propogating_message(self, method, node_features,edge_index,edge_type, rel_embedding, edge_norm,mode,type_):
        '''
        This function done the basic aggregation
        '''
        assert method in ["sum", "mean", "max"]
        assert mode in ["in","out","loop"]
        size = node_features.shape[0]
        coresponding_weight = getattr(self, 'w_{}'.format(mode))
        #-------------- this index selection: given relation embedding and relation_basic representation, choose the inital basis vector part
        # print(edge_type)
        relation_embedding = torch.index_select(rel_embedding,dim = 0, index = edge_type)
        # ------------- using index of tail in edge index to represent head by relation
        node_features = node_features[edge_index[1]]
        out = self.relation_transform(node_features, relation_embedding,type_)
        out = torch.matmul(out,coresponding_weight)
        out = out if edge_norm is None else out * edge_norm.view(-1, 1)
        out = self.scatter_function(method,out,edge_index[0],  size)
        return out    
    def forward(self, nodes_features, edge_index,edge_type,relation_embedding):
        '''
        Forward propogate function:
            Given input nodes_features, adj_matrix, relation_matrix
        '''
        if self.device is None:
            self.device = edge_index.device
        # ----------- add a self-loop dimension
        relation_embedding = torch.cat([relation_embedding,self.loop_rel],dim = 0)
        # print(edge_index.shape)
        num_edges = edge_index.shape[1]//2
        num_nodes = nodes_features.shape[self.nodes_dim]
        #---------------- in represent in_relation, out represent out_relation
        self.in_index, self.out_index = edge_index[:,:num_edges], edge_index[:,num_edges:]
        self.in_type, self.out_type = edge_type[:num_edges], edge_type[num_edges:]
        # --------------- create self-loop part
        self.loop_index = torch.stack([torch.arange(num_nodes), torch.arange(num_nodes)]).to(self.device)
        self.loop_type = torch.full((num_nodes,), relation_embedding.shape[0]-1, dtype = torch.long).to(self.device)
        # -------------- create normalization part
        self.in_norm = self.normalization(self.in_index, num_nodes)
        self.out_norm = self.normalization(self.out_index, num_nodes)
        #print(self.in_norm.isinf().any())
        in_res = self.propogating_message('sum',nodes_features,self.in_index,self.in_type, relation_embedding,self.in_norm,"in",self.comp_type)
        loop_res = self.propogating_message('sum',nodes_features,self.loop_index,self.loop_type, relation_embedding,None,"loop",self.comp_type)
        out_res = self.propogating_message('sum',nodes_features,self.out_index,self.out_type, relation_embedding,self.out_norm,"out",self.comp_type)
        # I don't know why but source code done it
        out = self.drop(in_res)*(1/3) + self.drop(out_res)*(1/3) + loop_res*(1/3)
        # update the relation embedding
        out_2 = torch.matmul(relation_embedding,self.weight_rel)
        return self.act(out),out_2[:-1]# ignoring self loop inserted 
    
class Comp_base(nn.Module):
    def __init__(self, num_entities, num_relation, edge_index, edge_type, in_channel, out_channel,num_basis_vector,comp_type = "sub"):
        super(Comp_base, self).__init__()
        self.num_entities = num_entities
        self.node_features = get_param((num_entities, in_channel))
        device = torch.device("cuda")
        self.edge_index = torch.tensor(edge_index).to(device)
        self.edge_type = torch.tensor(edge_type).to(device)
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.num_relation = num_relation
        self.num_basis_vector = num_basis_vector
        self.conv1 = CompGcnBasis(in_channels = in_channel, out_channels= out_channel,
                                num_relations=num_relation,
                                num_basis_vector= num_basis_vector, comp_type = comp_type)
        self.conv2 = CompGcn_non_first_layer(out_channel, out_channel, num_relation, comp_type = comp_type)
        self.drop1 = nn.Dropout(0.01)
        self.drop2 = nn.Dropout(0.01)
        self.register_parameter('bias', Parameter(torch.zeros(self.num_entities)))
    def forward_base(self, sub, rel):
        x, r = self.conv1(self.node_features, self.edge_index, self.edge_type)
        x = self.drop1(x)
        x, r = self.conv2(x, self.edge_index, self.edge_type, r)
        
        sub_embd = torch.index_select(x, 0, sub)
        rel_embd = torch.index_select(r, 0, rel)
        
        return sub_embd, rel_embd, x
    

class Comp_dismult(Comp_base):
    def __init__(self, data, in_channel, out_channel, num_basis_vector, comp_type = "sub"):
        super(self.__class__, self).__init__(data.num_ents, data.num_rels, data.train_edge_idx, data.train_edge_type, in_channel, out_channel, num_basis_vector, comp_type)
        self.num_entity = data.num_ents
        self.linear = nn.Linear(out_channel, 1)
    def forward(self, sub, rel, obj_ = None):
        sub, rel, all_ent = self.forward_base(sub, rel)
        obj = sub*rel
        x = torch.mm(obj, all_ent.transpose(1,0))
        x += self.bias.expand_as(x)
        # if obj_ is None:
        #     score = obj.unsqueeze(1) - all_ent
        #    # print("score: ", score.shape)
        # else:
        #     #print(obj_.shape)
        #     score = obj - all_ent[obj_]
        # return self.linear(score).squeeze(-1)
        # if not test:
        #     x = torch.norm(obj.unsqueeze(1) - all_ent, p=2, dim = 2)
        #     return x
        # else:
        #     return obj, all_ent
        return x

    
    def predict(self, pos, label, k=10):
        sub, rel, obj, _ = pos
        pred	= self.forward(sub, rel)
        pred = torch.sigmoid(pred)
        b_range			= torch.arange(pred.size()[0], device= torch.device("cuda"))
        target_pred		= pred[b_range, obj]
        pred 			= torch.where(label.byte(), -torch.ones_like(pred) * 10000000, pred)
        pred[b_range, obj] 	= target_pred
        ranks			= 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[b_range, obj]

        ranks 			= ranks.float()
        # count	= torch.numel(ranks) 		
        # mr		= torch.sum(ranks).item() 
        mrr		= torch.sum(1.0/ranks).item()   
        hits_10 = torch.numel(ranks[ranks <= 10])
        hits_3 = torch.numel(ranks[ranks <= 3])
        hits_1 = torch.numel(ranks[ranks <= 1])
        return hits_10, hits_3, hits_1, mrr