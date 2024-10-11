# coding: utf-8
# 

import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, degree
import torch_geometric
from tqdm import tqdm
from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss
from common.init import xavier_uniform_initialization
from torch_scatter import scatter_mean, scatter_sum, scatter_softmax


class RMR(GeneralRecommender):
    def __init__(self, config, dataset):
        super(RMR, self).__init__(config, dataset)

        num_user = self.n_users
        num_item = self.n_items
        batch_size = config['train_batch_size']         
        dim_x = config['embedding_size']                
        self.feat_embed_dim = config['feat_embed_dim']  
        self.n_layers = config['n_mm_layers']           
        self.knn_k = config['knn_k']                    
        self.mm_image_weight = config['mm_image_weight']
        has_id = True
        self.dropout = nn.Dropout(p=0.3)
        self.batch_size = batch_size
        self.num_user = num_user                       
        self.num_item = num_item                       
        self.k = 40
        self.num_interest = 3
        self.aggr_mode = config['aggr_mode']           
        self.user_aggr_mode = 'softmax'
        self.num_layer = 1
        self.dataset = dataset
        #self.construction = 'weighted_max'
        self.construction = 'cat'
        self.reg_weight = config['reg_weight']        
        self.drop_rate = 0.1
        self.v_rep = None
        self.t_rep = None

        self.review_score = None
        self.review_soft_list = None
        
        self.dim_latent = 384

        self.mm_adj = None
        self.a_feat = None
        
        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        # self.user_graph_dict = np.load(os.path.join(dataset_path, config['user_graph_dict_file']), allow_pickle=True).item()
        
        mm_adj_file = os.path.join(dataset_path, 'mm_adj_{}.pt'.format(self.knn_k))

        if self.v_feat is not None:
            # self.v_feat = nn.Parameter(self.v_feat)
            self.v_feat = nn.Embedding.from_pretrained(self.v_feat, freeze=False).weight  
            # self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)            
        if self.t_feat is not None:
            # self.t_feat = nn.Parameter(self.t_feat)
            self.t_feat = nn.Embedding.from_pretrained(self.t_feat, freeze=False).weight

        
        train_interactions = dataset.inter_matrix(form='coo').astype(np.float32)                     
        
        self.ui_graph = self.matrix_to_tensor(self.csr_norm(train_interactions, mean_flag=False))
        self.iu_graph = self.matrix_to_tensor(self.csr_norm(train_interactions.T, mean_flag=False))
 
        dense_interactions = torch.tensor(train_interactions.toarray(),dtype=torch.float32).to(self.device)
        self.dense_adj_mask = (dense_interactions != 0).float().transpose(1,0)
        self.dense_adj_bool = (self.dense_adj_mask.sum(-1) == 0.).float()
        

        self.user_id_embedding = nn.Parameter(nn.init.uniform_(torch.zeros(self.n_users, self.dim_latent),a=-1.0,b=1.0))

        self.MLP_t = nn.Linear(self.t_feat.shape[1], self.dim_latent)
        nn.init.uniform_(self.MLP_t.weight,a=-1.0,b=1.0)
        self.MLP_v = nn.Linear(self.v_feat.shape[1], self.dim_latent)
        nn.init.uniform_(self.MLP_v.weight,a=-1.0,b=1.0)

        self.MLP_t_1 = nn.Linear(self.dim_latent, self.dim_latent)
        # nn.init.uniform_(self.MLP_t.weight,a=-1.0,b=1.0)
        self.MLP_v_1 = nn.Linear(self.dim_latent, self.dim_latent)

        self.k = 1
        self.MLP_review = nn.Linear(self.dim_latent, int(self.dim_latent))
        self.MLP_review_v = nn.Linear(self.dim_latent, int(self.dim_latent / (self.k)))
        self.MLP_review_t = nn.Linear(self.dim_latent, int(self.dim_latent / (self.k)))
        nn.init.uniform_(self.MLP_review.weight,a=-1.0,b=1.0)
        nn.init.uniform_(self.MLP_review_v.weight,a=-1.0,b=1.0)
        nn.init.uniform_(self.MLP_review_t.weight,a=-1.0,b=1.0)

        self.t_preference = nn.Parameter(nn.init.xavier_normal_(torch.tensor(
            np.random.randn(num_user, self.dim_latent), dtype=torch.float32, requires_grad=True),
            gain=1).to(self.device))  
        self.v_preference = nn.Parameter(nn.init.xavier_normal_(torch.tensor(
            np.random.randn(num_user, self.dim_latent), dtype=torch.float32, requires_grad=True),
            gain=1).to(self.device))  

        if self.a_feat:
            self.a_gcn = GCN(self.dataset, batch_size, num_user, num_item, dim_x, self.aggr_mode,        
                            num_layer=self.num_layer, has_id=has_id, dropout=self.drop_rate, dim_latent=self.dim_latent,
                            device=self.device, features=self.t_feat)
            self.a_preference = nn.Parameter(nn.init.xavier_normal_(torch.tensor(
            np.random.randn(num_user, self.dim_latent), dtype=torch.float32, requires_grad=True),
            gain=1).to(self.device))  
            
            self.MLP_a = nn.Linear(self.a_feat.shape[1], self.dim_latent)
        self.t_score = None
        self.v_score = None
        if os.path.exists(mm_adj_file):
            self.mm_adj = torch.load(mm_adj_file)                                           
        else:
            if self.v_feat is not None:
                indices, image_adj = self.get_knn_adj_mat(self.image_embedding.weight.detach())
                self.mm_adj = image_adj
            if self.t_feat is not None:
                indices, text_adj = self.get_knn_adj_mat(self.text_embedding.weight.detach())
                self.mm_adj = text_adj
            if self.v_feat is not None and self.t_feat is not None:
                self.mm_adj = self.mm_image_weight * image_adj + (1.0 - self.mm_image_weight) * text_adj
                del text_adj
                del image_adj
            torch.save(self.mm_adj, mm_adj_file)
            
    def _reset_paramaters(self):
        with torch.no_grad():
            self.user_id_embedding.data = nn.init.normal_(torch.zeros(self.n_users, self.dim_latent)).to(self.device)
            self.item_id_embedding.data = nn.init.normal_(torch.zeros(self.n_items, self.dim_latent)).to(self.device)

            self.weight_u.data = nn.init.xavier_normal_(
                torch.tensor(np.random.randn(self.num_user, 2), dtype=torch.float32, requires_grad=True)).to(self.device)
            # self.weight_u.data = F.softmax(self.weight_u, dim=1)                                        

            self.t_preference.data = nn.init.xavier_normal_(torch.tensor(
                np.random.randn(self.num_user, self.dim_latent), dtype=torch.float32, requires_grad=True),
                gain=1).to(self.device)
            self.v_preference.data = nn.init.xavier_normal_(torch.tensor(
                np.random.randn(self.num_user, self.dim_latent), dtype=torch.float32, requires_grad=True),
                gain=1).to(self.device)

            # self.MLP_t = nn.Linear(self.t_feat.shape[1], self.dim_latent)
            self.MLP_t.weight.data = torch.empty(self.dim_latent,self.t_feat.shape[1]).to(self.device)
            nn.init.xavier_normal_(self.MLP_t.weight)
            # self.MLP_v = nn.Linear(self.v_feat.shape[1], self.dim_latent).to(self.device)
            self.MLP_v.weight.data = torch.empty( self.dim_latent,self.v_feat.shape[1]).to(self.device)
            nn.init.xavier_normal_(self.MLP_v.weight)
            # self.MLP_review = nn.Linear(self.dim_latent, self.dim_latent)
            # nn.init.xavier_normal_(self.MLP_review.weight)
            
    def pca(self, x, k=2):
        x_mean = torch.mean(x, 0)
        x = x - x_mean
        cov_matrix = torch.matmul(x.t(), x) / (x.size(0) - 1)
        eigenvalues, eigenvectors = torch.linalg.eig(cov_matrix)
        sorted_eigenvalues, indices = torch.sort(eigenvalues.real, descending=True)
        components = eigenvectors[:, indices[:k]]
        x_pca = torch.matmul(x, components)

        return x_pca
    
    def matrix_to_tensor(self, cur_matrix):
        if type(cur_matrix) != sp.coo_matrix:
            cur_matrix = cur_matrix.tocoo()  
        ui_indices = torch.from_numpy(np.vstack((cur_matrix.row, cur_matrix.col)).astype(np.int64))  
        iu_indices = torch.from_numpy(np.vstack((cur_matrix.col, cur_matrix.row)).astype(np.int64))
        values = torch.from_numpy(cur_matrix.data)  
        shape = torch.Size(cur_matrix.shape)

        return torch.sparse.FloatTensor(ui_indices, values, shape).to(torch.float32).cuda()

    def csr_norm(self, csr_mat, mean_flag=False):
        rowsum = np.array(csr_mat.sum(1))
        rowsum = np.power(rowsum+1e-8, -0.5).flatten()
        rowsum[np.isinf(rowsum)] = 0.
        rowsum_diag = sp.diags(rowsum)

        colsum = np.array(csr_mat.sum(0))
        colsum = np.power(colsum+1e-8, -0.5).flatten()
        colsum[np.isinf(colsum)] = 0.
        colsum_diag = sp.diags(colsum)

        if mean_flag == False:
            return rowsum_diag*csr_mat*colsum_diag
        else:
            return rowsum_diag*csr_mat
    def mm(self, x, y): 
        return torch.sparse.mm(x, y)
    def get_knn_adj_mat(self, mm_embeddings):
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()
        del sim
        # construct sparse adj
        indices0 = torch.arange(knn_ind.shape[0]).to(self.device)
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, self.knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        # norm
        return indices, self.compute_normalized_laplacian(indices, adj_size)
    
    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse.FloatTensor(indices, values, adj_size)

    
    def pre_epoch_processing(self):
        self.epoch_user_graph, self.user_weight_matrix = self.topk_sample(self.k)   
        self.user_weight_matrix = self.user_weight_matrix.to(self.device)

    def pack_edge_index(self, inter_mat):
        rows = inter_mat.row
        cols = inter_mat.col + self.n_users
        # ndarray([598918, 2]) for ml-imdb
        return np.column_stack((rows, cols))

    def _gcn_pp(self, item_embed, user_embed, uig, iug, norm=False):
        if norm == True:
            item_res =  item_embed = F.normalize(item_embed)
            user_res =  user_embed = F.normalize(user_embed)
            for _ in range(2):
                user_agg = self.mm(uig, item_embed)
                item_agg = self.mm(iug, user_embed)
                item_embed = item_agg
                user_embed = user_agg
                
                item_res = item_res + item_embed
                user_res = user_res + user_embed
        
        else:
            item_res =  item_embed = F.normalize(item_embed)
            user_res =  user_embed = F.normalize(user_embed)
            for _ in range(2):
                user_agg = self.mm(uig, item_embed)
                item_agg = self.mm(iug, user_embed)
                item_embed = F.normalize(item_agg)
                user_embed = F.normalize(user_agg)

                # item_embed = item_agg
                # user_embed = user_agg
                
                item_res = item_res + item_embed
                user_res = user_res + user_embed
        
        
        return user_res, item_res
    
    def self_attention(self, t_feat, v_feat,pattern="train"):
        if pattern == "infer":
            t_feat = self.MLP_t(t_feat)
            v_feat = self.MLP_v(v_feat)
        
        t_feat = self.MLP_t_a_2(t_feat)
        v_feat = self.MLP_v_a_2(v_feat)
        t_score = torch.sigmoid(self.MLP_t_a(t_feat))
        v_score = torch.sigmoid(self.MLP_v_a(v_feat))
        if pattern == "infer":
            t_score = t_score.squeeze(-1)
            v_score = v_score.squeeze(-1)
        return t_score, v_score
    

    def review_modal(self, modal_feat, index=None, check_pattern="none"):
        user_id_feat = self.MLP_review(self.dropout(self.user_id_embedding)) 
        user_id_feat = user_id_feat.view((user_id_feat.shape[0],self.k,-1))
        
        if check_pattern=="text":
            # modal_feat[0] = self.MLP_t(modal_feat[0])
            modal_feat[0] = self.MLP_review_t(modal_feat[0])
            adj_mask = self.dense_adj_mask[index]
            adj_bool = self.dense_adj_bool[index]
        elif check_pattern == "image":
            # modal_feat[0] = self.MLP_v(modal_feat[0])
            modal_feat[0] = self.MLP_review_v(modal_feat[0])
            adj_mask = self.dense_adj_mask[index]
            adj_bool = self.dense_adj_bool[index]
        else:
            adj_mask = self.dense_adj_mask
            adj_bool = self.dense_adj_bool
            pass
            
        review_list = []
        for feat in modal_feat:
            score_mat = torch.sigmoid(torch.matmul(user_id_feat.half(),feat.transpose(1,0).half())) 
            score_mat = score_mat.mean(dim=1).transpose(1,0)#(7050, 19445)
            score_mat = score_mat * adj_mask
            feat_score = (score_mat.sum(dim=1)/(adj_mask.sum(dim=1) + adj_bool)).unsqueeze(1)     
            review_list.append(feat_score)       
        
        return  torch.cat(review_list,dim=1)    




    def update_modal_feat(self, n_t_feat, n_v_feat):
        del self.t_feat, self.v_feat
        torch.cuda.empty_cache()
        
        self.t_feat = torch.from_numpy(n_t_feat).to(self.device).float()
        self.v_feat = torch.from_numpy(n_v_feat).to(self.device).float()
        self.t_feat = nn.Embedding.from_pretrained(self.t_feat, freeze=False).weight
        self.v_feat = nn.Embedding.from_pretrained(self.v_feat, freeze=False).weight
        
    def forward(self):

        t_feat = self.t_feat
        v_feat = self.v_feat
        

        modal_scores = self.review_modal([self.MLP_review_t(t_feat), self.MLP_review_v(v_feat)])
        # t_score, v_score = self.self_attention(t_feat, v_feat)
        t_score,v_score = modal_scores[:,0].unsqueeze(-1), modal_scores[:,1].unsqueeze(-1)

        self.t_score = t_score.squeeze(-1)
        self.v_score = v_score.squeeze(-1)
        # self.t_score = t_score
        # self.v_score = v_score
        temp_user = torch.tanh(self.mm(self.iu_graph,self.dropout(self.user_id_embedding)))

        t_feat = t_feat * (t_score + 1e-3) + (1-t_score) * temp_user
        v_feat = v_feat * (v_score + 1e-3) + (1-v_score) * temp_user
        
        v_user_embed, v_item_embed = self._gcn_pp(v_feat , self.v_preference, self.ui_graph, self.iu_graph,norm=True)
        t_user_embed, t_item_embed = self._gcn_pp(t_feat , self.t_preference, self.ui_graph, self.iu_graph,norm=True)


        item_rep = torch.cat([t_item_embed, v_item_embed], dim=-1)
        user_rep = torch.cat([t_user_embed, v_user_embed], dim=-1)

        return user_rep, item_rep

       
    def _sparse_dropout(self, x, rate=0.0):
        noise_shape = x._nnz()                                      

        random_tensor = rate
        random_tensor += torch.rand(noise_shape).to(x.device)       
        dropout_mask = torch.floor(random_tensor).type(torch.bool)   
        i = x._indices()                                            
        v = x._values()                                             

        i = i[:, dropout_mask]                                      
        v = v[dropout_mask]                                        

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)   
        # return out * (1. / (1 - rate))                             
        return out

    def bpr_loss(self, interaction):
        user_embed, item_embed = self.forward()
        user_nodes, pos_item_nodes, neg_item_nodes = interaction[0], interaction[1], interaction[2]


        user_embed = user_embed[user_nodes]
        pos_item_embed = item_embed[pos_item_nodes]
        neg_item_embed = item_embed[neg_item_nodes]
        # users, pos_items, neg_items = self.forward(interaction)
        pos_scores = torch.sum(torch.mul(user_embed, pos_item_embed), dim=1)
        neg_scores = torch.sum(torch.mul(user_embed, neg_item_embed), dim=1)


        regularizer = 1./2*(user_embed**2).sum() + 1./2*(pos_item_embed**2).sum() + 1./2*(neg_item_embed**2).sum()        
        regularizer = regularizer / self.batch_size

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        emb_loss = 1e-4 * regularizer
        
        loss = mf_loss + emb_loss
        return loss

    def calculate_loss(self, interaction):
        user = interaction[0]
        pos_scores, neg_scores = self.forward(interaction)
        loss_value = -torch.mean(torch.log2(torch.sigmoid(pos_scores - neg_scores)))
        reg_embedding_loss_v = (self.v_preference[user] ** 2).mean() if self.v_preference is not None else 0.0
        reg_embedding_loss_t = (self.t_preference[user] ** 2).mean() if self.t_preference is not None else 0.0

        reg_loss = self.reg_weight * (reg_embedding_loss_v + reg_embedding_loss_t)
        if self.construction == 'weighted_sum':
            reg_loss += self.reg_weight * (self.weight_u ** 2).mean()
            reg_loss += self.reg_weight * (self.weight_i ** 2).mean()
        elif self.construction == 'cat':
            reg_loss += self.reg_weight * (self.weight_u ** 2).mean()
        elif self.construction == 'cat_mlp':
            reg_loss += self.reg_weight * (self.MLP_user.weight ** 2).mean()
        return loss_value + reg_loss

    def _get_review_res(self):
        return self.review_score.cpu().numpy(), self.review_soft_list.cpu().numpy()

    def full_sort_predict(self, interaction):
        user_tensor, item_tensor = self.forward()
        # user_tensor = self.result_embed[:self.n_users]
        # item_tensor = self.result_embed[self.n_users:]

        temp_user_tensor = user_tensor[interaction[0], :]
        score_matrix = torch.matmul(temp_user_tensor, item_tensor.t())
        return score_matrix
    


class Modal_Reviewer(torch.nn.Module):
    def __init__(self,num_user, num_item, dim_latent, n_interests, t_dim=None, v_dim =None, a_dim=None, dense_adj=None, select_value=0.3):
        super(Modal_Reviewer,self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.dim_latent = dim_latent
        self.num_interests = n_interests
        self.select_value = select_value
        
        # self.dense_adj_mask = (dense_adj != 0).float() #(19945,7040)
        self.dense_adj_mask = (dense_adj != 0).float().transpose(1,0)
        self.dense_adj_bool = (self.dense_adj_mask.sum(-1) == 0.).float()
        self.t_dim = t_dim
        self.v_dim = v_dim
        self.a_dim = a_dim
        
    def STE(self, feat_score):

        return feat_score
        # z_q = z + (z_q - z).detach() 
    
    def forward(self,modal_feat,preference_list):
        preference = sum(preference_list) / len(preference_list)   
        # muti_preference = self._multi_interests(preference)        
        review_list = []
        review_soft_list = []
        for feat in modal_feat:                                         
            score_mat = torch.sigmoid(torch.matmul(feat,preference.transpose(1,0)))      
            score_mat = score_mat * self.dense_adj_mask
            feat_score = (score_mat.sum(dim=1)/(self.dense_adj_mask.sum(dim=1) + +self.dense_adj_bool)).unsqueeze(1)               
            review_list.append(feat_score)
        # return torch.cat(review_list,dim=1), torch.cat(review_soft_list, dim=1), muti_preference.mean(dim=1)                                                  
        return   torch.cat(review_list,dim=1)        

        
        
class User_Graph_sample(torch.nn.Module):
    def __init__(self, num_user, aggr_mode,dim_latent):
        super(User_Graph_sample, self).__init__()
        self.num_user = num_user
        self.dim_latent = dim_latent
        self.aggr_mode = aggr_mode

    def forward(self, features,user_graph,user_matrix):
        index = user_graph
        u_features = features[index]           
        user_matrix = user_matrix.unsqueeze(1) 
        # pdb.set_trace()
        u_pre = torch.matmul(user_matrix,u_features)
        u_pre = u_pre.squeeze()
        return u_pre                           


class GCN(torch.nn.Module):
    def __init__(self,datasets, batch_size, num_user, num_item, dim_id, aggr_mode, num_layer, has_id, dropout,
                 dim_latent=None,device = None,features=None):
        super(GCN, self).__init__()
        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.datasets = datasets
        self.dim_id = dim_id
        self.dim_feat = features.size(1)
        self.dim_latent = dim_latent
        self.aggr_mode = aggr_mode
        self.num_layer = num_layer
        self.has_id = has_id
        self.dropout = dropout
        self.device = device

        if self.dim_latent:
            # self.MLP = nn.Linear(self.dim_feat, 4*self.dim_latent)                                  
            # self.MLP_1 = nn.Linear(4*self.dim_latent, self.dim_latent)                              
            self.conv_embed_1 = Base_gcn(self.dim_latent, self.dim_latent, aggr=self.aggr_mode)     
        else:
            self.conv_embed_1 = Base_gcn(self.dim_latent, self.dim_latent, aggr=self.aggr_mode)

    def forward(self,edge_index,features, preference):
        # temp_features = self.MLP_1(F.leaky_relu(self.MLP(features))) if self.dim_latent else features   
        temp_features = features
        x = torch.cat((preference, temp_features), dim=0).to(self.device)                          
        x = F.normalize(x).to(self.device)     
        h = self.conv_embed_1(x, edge_index)  
        h_1 = self.conv_embed_1(h, edge_index) 
        x_hat = x + h + h_1

        return x_hat


class Base_gcn(MessagePassing):
    def __init__(self, in_channels, out_channels, normalize=True, bias=True, aggr='add', **kwargs):
        super(Base_gcn, self).__init__(aggr=aggr, **kwargs)
        self.aggr = aggr                  #add
        self.in_channels = in_channels    #64
        self.out_channels = out_channels  #64

    def forward(self, x, edge_index, size=None):
        # pdb.set_trace()
        if size is None:
        #     edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = x.unsqueeze(-1) if x.dim() == 1 else x  #[26495, 64]
        # pdb.set_trace()
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j, edge_index, size):
        if self.aggr == 'add':
            # pdb.set_trace()
            row, col = edge_index
            deg = degree(row, size[0], dtype=x_j.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
            return norm.view(-1, 1) * x_j
        return x_j

    def update(self, aggr_out):
        return aggr_out

    def __repr(self):
        return '{}({},{})'.format(self.__class__.__name__, self.in_channels, self.out_channels)


class Linears(torch.nn.Module):
    def __init__(self,inp_dim, out_dim, ln=True) :
        super(Linears, self).__init__()
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.ln = ln
        self.layer = nn.Linear(self.inp_dim, self.out_dim)
        nn.init.xavier_uniform_(self.layer.weight)
        if self.ln:
            self.LN = nn.LayerNorm(self.out_dim)
    def forward(self,x):
        x = self.layer(x)
        if self.ln:
            x = self.LN(x) + x
        return x
    
class Encoder(torch.nn.Module):
    def __init__(self,inp_dim, out_dim,channel_list):
        super(Encoder,self).__init__()
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        # self.channel_list = channel_list.insert(0, self.inp_dim)
        self.channel_list = channel_list
        self.channel_list.insert(0, self.inp_dim)
        self.layers = [Linears(self.channel_list[i],self.channel_list[i+1],ln=True) for i in range(len(self.channel_list)-1)]
        # self.layers.append(nn.Linear(self.channel_list[-1], self.out_dim))
        self.layers.append(Linears(self.channel_list[-1], self.out_dim))
        self.model = nn.Sequential(*self.layers)
        
    def forward(self, x):
        out = self.model(x)
        return out

class Decoder(torch.nn.Module):
    def __init__(self,inp_dim, out_dim, channel_list):
        super(Decoder,self).__init__()
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.channel_list = channel_list
        self.channel_list.insert(0, self.inp_dim)
        self.layers = [Linears(self.channel_list[i],self.channel_list[i+1], ln=True) for i in range(len(self.channel_list)-1)]
        # self.layers.append(nn.Linear(self.channel_list[-1], self.out_dim))
        self.layers.append(Linears(self.channel_list[-1], self.out_dim))
        self.model = nn.Sequential(*self.layers)
        
    def forward(self, x):
        out = self.model(x)
        return out

class Cookbook(torch.nn.Module):
    def __init__(self, num_codebook_vectors, code_dim, beta=0.1):
        super(Cookbook, self).__init__()
        self.num_codebook_vectors = num_codebook_vectors
        self.code_dim = code_dim
        self.beta = beta
        self.embedding = nn.Embedding(self.num_codebook_vectors, self.code_dim)
        nn.init.xavier_normal_(self.embedding.weight, gain=1.0)
    
    def forward(self, z):
        z_r = z.view(-1, self.code_dim).contiguous()

        d_0 = torch.sum(z_r**2, dim=1, keepdim=True)
        d_1 = torch.sum(self.embedding.weight**2, dim=1)
        d_2 = 2*(torch.matmul(z_r, self.embedding.weight.t()))
        d = d_0 + d_1 - d_2
 
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        code_loss = torch.mean((z_q.detach() - z)**2) + self.beta * torch.mean((z_q - z.detach())**2)

        z_q = z + (z_q - z).detach() 
        return z_q, code_loss

class Discriminator(nn.Module):
    def __init__(self, inp_dim, dis_channel_list): 
        super(Discriminator, self).__init__()
        self.inp_dim = inp_dim
        self.channel_list = dis_channel_list
        self.channel_list.insert(0, self.inp_dim)
        self.layers = [Linears(self.channel_list[i],self.channel_list[i+1], ln=True) for i in range(len(self.channel_list)-1)]
        self.layers.append(nn.Linear(self.channel_list[-1], 1))
        self.model = nn.Sequential(*self.layers)
        
    def forward(self, x):
        out = torch.sigmoid(self.model(x))
        return out

        
class VQGAN4REC(torch.nn.Module):
    def __init__(self, ec_inp_dim, ec_out_dim, tgt_dim, prompt_dim, prompt_vocab_size, 
                 code_book_num_vector, code_dim,ec_channel_list=[512], dc_channel_list = [512]):
        super(VQGAN4REC, self).__init__()
        self.ec_inp_dim = ec_inp_dim
        self.ec_out_dim = ec_out_dim
        self.tgt_dim = tgt_dim
        self.ec_channel_list = ec_channel_list
        self.dc_channel_list = dc_channel_list
        # self.dis_channel_list = dis_channel_list
        self.prompt_dim = prompt_dim
        self.prompt_vocab_size = prompt_vocab_size
        self.prompt_embed = nn.Embedding(self.prompt_vocab_size, self.prompt_dim)
        
        self.encoder = Encoder(self.ec_inp_dim + self.prompt_dim, self.ec_out_dim, self.ec_channel_list)
        self.decoder = Decoder(self.ec_out_dim, self.tgt_dim, self.dc_channel_list)
        # self.discriminator = Discriminator(self.tgt_dim,dis_channel_list)
        self.code_book = Cookbook(num_codebook_vectors = code_book_num_vector, code_dim=code_dim)
    
    def forward(self,x):
        dense_feat, propmt_token = x[0], x[1]
        prompt_emb = self.prompt_embed(propmt_token)
        dense_feat = torch.cat([dense_feat, prompt_emb],dim=-1)
        # dense_feat = F.normalize(dense_feat)
        encode_embed = self.encoder(dense_feat)
        code_embed, code_loss = self.code_book(encode_embed)
        decode_embed = self.decoder(code_embed)
        
        return decode_embed, code_loss
        
class VQGANTrainer(torch.nn.Module):
    def __init__(self,config):
        super(VQGANTrainer, self).__init__()
        self.vqgan4rec =  VQGAN4REC(config["ec_inp_dim"], config["ec_out_dim"], config["tgt_dim"], config["prompt_dim"], config["prompt_vocab_size"],
                                    config["code_book_num_vector"], config["code_dim"], ec_channel_list=[1024,2048],dc_channel_list = [1024,2048]).to(config["device"])
        self.discriminator = Discriminator(config["tgt_dim"],dis_channel_list=[512,128]).to(config["device"])
        self.opt_vq, self.opt_disc = self.configure_optimizers(config)
        self.device = config["device"]
        self.criterion = nn.BCELoss()
        
    def configure_optimizers(self, config):
        vq_lr = config["vq_lr"]
        dis_lr = config["dis_lr"]
        opt_vq = torch.optim.AdamW(
            self.vqgan4rec.parameters(),
            lr=vq_lr, eps=1e-08, betas=(config["beta1"], config["beta2"])
        )
        opt_disc = torch.optim.Adam(self.discriminator.parameters(),
                                    lr=dis_lr, eps=1e-08, betas=(config["beta1"], config["beta2"]))

        return opt_vq, opt_disc 
    
    def train(self,dataloader,config):
        self.vqgan4rec.train()
        self.discriminator.train()
        for epoch in range(config["TS_epoch"]):
            all_rec_loss = 0
            # all_g_loss = 0
            all_gan_loss = 0
            all_code_loss = 0
            for feature, pattern, label in tqdm(dataloader,desc="training TS model"):
                feature = feature.to(self.device)
                pattern = pattern.to(self.device)
                label = label.to(self.device)
                decode_feat, code_loss = self.vqgan4rec([feature, pattern])
                rec_loss = torch.abs(decode_feat - label).mean()
                all_rec_loss += rec_loss.item()
                all_code_loss += code_loss.item()
                
                real_label = torch.ones(label.shape[0],1).to(self.device)
                fake_label = torch.zeros(label.shape[0],1).to(self.device)
                disc_real = self.discriminator(label)
                disc_fake = self.discriminator(decode_feat)


                real_loss = self.criterion(disc_real, real_label)
                fake_loss = self.criterion(disc_fake, fake_label)
                gan_loss = 0.1 * (real_loss + fake_loss)
                all_gan_loss += gan_loss.item()
                
                vq_loss = rec_loss + code_loss
                vq_loss.backward(retain_graph=True)
                self.opt_vq.step()
                self.opt_vq.zero_grad()
                
                gan_loss.backward()
                self.opt_disc.step()
                self.opt_disc.zero_grad()
                
            print("all_rec_loss: ", all_rec_loss)
            # print("all_g_loss: ", all_g_loss)
            print("all_gan_loss: ", all_gan_loss)
            print("all_code_loss: ", all_code_loss)
    
    def inference(self, dataloader, data_type="th_vl"):
        self.vqgan4rec.eval()
        infer_res = []
        for feature, pattern in tqdm(dataloader,desc="TS infer for " + data_type):
            feature = feature.to(self.device)
            pattern = pattern.to(self.device)
            
            decode_feat, code_loss = self.vqgan4rec([feature, pattern])
            decode_feat = decode_feat.detach().cpu().numpy()
            
            infer_res.append(decode_feat)
        infer_res = np.concatenate(infer_res,axis=0)
        return infer_res
        
