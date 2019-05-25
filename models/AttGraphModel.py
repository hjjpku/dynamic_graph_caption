from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
import misc.utils as utils

import copy
import math
import numpy as np
#from matplotlib import pyplot as plt
#from matplotlib.pyplot import savefig

from .CaptionModel import CaptionModel
from .AttModel import pack_wrapper, AttModel, Attention
#from .ProjectNet import ProjectNet
#from .GCN import GCN, BuildGraph

class ConvOneonOne(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(ConvOneonOne, self).__init__()
        self.k=out_channel
        self.d=in_channel
        self.weight=nn.Parameter(torch.FloatTensor(in_channel,out_channel))
        torch.nn.init.xavier_normal_(self.weight)
        self.bias=nn.Parameter(torch.zeros(out_channel))

    def forward(self, x,mask):
        #x: [batch,max_node_num,in_channel]
        #mask: [batch,max_node_num]

        y=torch.matmul(x,self.weight)+self.bias
        return y*mask.unsqueeze(2)

class ProjectNet(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, opt, num_clusters=32, dim=512,alpha = 1,normalize_input=True):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super(ProjectNet, self).__init__()
        self.num_clusters = num_clusters
        self.alpha = alpha
        self.normalize_input = normalize_input
#       self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=True)
        self.conv=ConvOneonOne(dim,num_clusters)
        self.centroids = nn.Parameter(torch.rand(dim,self.num_clusters))
        self._init_params()
        self.vis_soft_assign = opt.vis_soft_assign
        if self.vis_soft_assign:
            self.vis_dir = opt.vis_dir
            if not os.path.exists(self.vis_dir):
                os.makedirs(self.vis_dir)


    def _init_params(self):
        self.conv.weight.data=(2.0 * self.alpha * self.centroids)
        self.conv.bias.data= - self.alpha * self.centroids.norm(dim=0)

    def forward(self, x, mask):
        #B, N, C = x.shape[:3]

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=2)  # across descriptor dim

        # soft-assignment
        soft_assign = self.conv(x,mask)
        soft_assign = F.softmax(soft_assign, dim=2) # it should be size B x N x K
        # mask
        soft_assign = soft_assign * mask.unsqueeze(2)

        # entropy
        log_p = -torch.log2(soft_assign)
        log_p[torch.isinf(log_p)] = 0
        assign_entropy = (soft_assign * log_p).sum()/ mask.sum()

        # calculate assign distribution for KLD loss
        assign_dist = soft_assign.sum(1).squeeze(1)
        assign_dist = F.log_softmax(assign_dist, dim=1)



        if self.vis_soft_assign:
            assign_dist_np = assign_dist.cpu().numpy()
            for i in range(soft_assign.size()[0]):
                img_name = self.vis_dir + '/_' + str(random.random()) + '_.jpg'
                assign_mat = soft_assign[i].cpu().numpy()
                '''
                plt.imshow(assign_mat,cmap=plt.cm.hot)
                plt.xticks(np.arange(assign_mat.shape[0]),assign_mat.shape[0] , rotation=45)
                plt.yticks(np.arange(assign_mat.shape[0]), assign_mat.shape[0])
                plt.colorbar()
                plt.show()
                plt.savefig(img_name)
            '''
        x_flatten = x

        # calculate residuals to each clusters
        # x_flatten after process: B X N X C X K
        # centroids after process: C X K
        residual = x_flatten.unsqueeze(3).expand(-1,-1,-1,self.num_clusters) - self.centroids
        residual *= soft_assign.unsqueeze(2)
        vlad = residual.sum(dim=1) # B X C X K

        vlad = F.normalize(vlad, p=2, dim=1)  # intra-normalization
        vlad = torch.transpose(vlad,1,2) #B X K X C

        #vlad = vlad.contiguous().view(x.size(0), -1)  # flatten
        #vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        return vlad, assign_dist, assign_entropy

class AttGraphModel(AttModel):
    def __init__(self, opt):
        super(AttGraphModel, self).__init__(opt)
        self.opt = opt

        # new options
        self.num_k = opt.num_k # number of project centers
        self.use_proj = opt.use_proj # option for projection
        self.use_graph = opt.use_graph
        self.num_layers = 2 # keep the same with topdown
        self.p_dim = opt.p_dim if self.use_proj else opt.rnn_size  # dimension of the graph embeddings
        self.p_dim = opt.rnn_size if (not self.use_graph) or self.use_bn or (self.use_graph and opt.gcn_pool == 'att') else self.p_dim
        if hasattr(opt, 'proj_KL'):
            self.proj_KL = opt.proj_KL
        else:
            self.proj_KL = 0
        '''
        self.p_dim = opt.rnn_size if self.use_bn else opt.p_dim # dimension of the graph embeddings
        if opt.p_dim != opt.rnn_size:
            print('graph embedding dim redefined to rnn_size %d because of bn' % self.p_dim)
        '''

        #keep fc_embed temporally
        # add project_net
        if self.use_proj:
            self.project_net = ProjectNet(opt, self.num_k, self.p_dim)
            if not self.use_bn:
                self.graph_embed = nn.Sequential(nn.Linear(self.att_feat_size, self.p_dim),
                                        nn.ReLU(),
                                        nn.Dropout(self.drop_prob_lm))

        if self.use_graph:
            del self.ctx2att
        else:
            self.ctx2att = nn.Linear(self.p_dim, self.att_hid_size)

        if not self.use_bn and self.use_proj:
            del self.att_embed

        self.core = AttGraphCore(opt)
        print(self.use_proj)
        print(self.use_graph)

    def _prepare_feature(self, fc_feats, att_feats, att_masks):
        graph_embed, pp_att_feats =  None, None
        att_feats, att_masks = self.clip_att(att_feats, att_masks)

        fc_feats = self.fc_embed(fc_feats)
        # dealing with the mask issue for bn layers in att_embed
        if self.use_bn or not self.use_proj:
            att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)
        if self.use_proj:
            # graph_embed has a fixed number of nodes
            if not self.use_bn:
                graph_embed_t = self.graph_embed(att_feats)
                att_feats = graph_embed_t
            graph_embed, assign_dist, assign_entropy = self.project_net.forward(att_feats, att_masks)
	
            att_masks = att_masks.new(att_masks.size()[0],self.num_k).zero_()+1
            if self.use_graph:
                return fc_feats, graph_embed, pp_att_feats, att_masks, assign_dist, assign_entropy #is none
            else:
                # projection but no graph
                pp_att_feats = self.ctx2att(graph_embed)
                return fc_feats, graph_embed, pp_att_feats, att_masks, assign_dist, assign_entropy
        elif self.use_graph:
            #no projection but use graph
            return fc_feats, att_feats, pp_att_feats, att_masks, None, None #is none

    def _forward(self, fc_feats, att_feats, seq, att_masks=None):
        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)

        outputs = fc_feats.new_zeros(batch_size, seq.size(1) - 1, self.vocab_size+1)

        # Prepare the features
        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, assign_dist, assign_entropy = self._prepare_feature(fc_feats, att_feats, att_masks)
        # pp_att_feats is used for attention, we cache it in advance to reduce computation cost

        for i in range(seq.size(1) - 1):
            if self.training and i >= 1 and self.ss_prob > 0.0: # otherwiste no need to sample
                sample_prob = fc_feats.new(batch_size).uniform_(0, 1)
                sample_mask = sample_prob < self.ss_prob
                if sample_mask.sum() == 0:
                    it = seq[:, i].clone()
                else:
                    sample_ind = sample_mask.nonzero().view(-1)
                    it = seq[:, i].data.clone()
                    #prob_prev = torch.exp(outputs[-1].data.index_select(0, sample_ind)) # fetch prev distribution: shape Nx(M+1)
                    #it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1))
                    # prob_prev = torch.exp(outputs[-1].data) # fetch prev distribution: shape Nx(M+1)
                    prob_prev = torch.exp(outputs[:, i-1].detach()) # fetch prev distribution: shape Nx(M+1)
                    it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
            else:
                it = seq[:, i].clone()
            # break if all the sequences end
            if i >= 1 and seq[:, i].sum() == 0:
                break

            output, state = self.get_logprobs_state(it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, state)
            outputs[:, i] = output
        if self.use_proj and self.proj_KL:
            return outputs, assign_dist, assign_entropy
        else:
            return outputs

class GCN(nn.Module):
    def __init__(self, opt, input_dim, output_dim):
        super(GCN, self).__init__()

        # default to freeze the opts
        bias = True
        self.normalize_embedding = False

        self.dropout = opt.gcn_dropout
        self.relu = opt.gcn_relu
        if self.dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=self.dropout)

        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim).cuda())
        torch.nn.init.xavier_normal_(self.weight)
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_dim).cuda())
        else:
            self.bias = None

    def forward(self, x, adj, mask):
        if self.dropout > 0.001:
            x = self.dropout_layer(x)
        y = torch.matmul(adj, x)
        y = torch.matmul(y, self.weight)
        if self.bias is not None:
            y = y + self.bias
        if self.normalize_embedding:
            y = F.normalize(y, p=2, dim=2)
        if self.relu == 'relu':
            y = torch.nn.functional.relu(y)
        elif self.relu == 'lrelu':
            y = torch.nn.functional.leaky_relu(y, 0.1)
        y = mask.unsqueeze(2) * y
        return y

class StructureAttention(nn.Module):
    def __init__(self, opt, input_dim, output_dim):
        super(StructureAttention, self).__init__()
        self.topk = opt.topk # ratio
        self.norm = opt.norm_type
        self.embed1 = nn.Linear(input_dim, output_dim)
        self.embed2 = nn.Linear(input_dim, output_dim)

    def _adj_norm(self, Adj):
        if self.norm == 'row':
            rowsum = Adj.sum(2) # b x n
            d_inv_sqrt = torch.pow(rowsum, -1/2)
            d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0
            d_inv_sqrt = d_inv_sqrt.unsqueeze(2).expand(-1,-1,d_inv_sqrt.size()[1])
            Adj_norm = d_inv_sqrt * Adj * d_inv_sqrt.transpose(1,2)
        elif self.norm == 'global':
            sum = Adj.sum(2).sum(1)
            Adj_norm = Adj / sum
        else:
            print('no norm type for Adj')
            exit(-1)
        return  Adj_norm



    def forward(self, graph_embed, mask, hidden_state):
        # graph_embed = b x n x c
        # hidden_state = b x rnn_size
        # mask = b x n
        expanded_h = hidden_state.unsqueeze(1).expand(-1, graph_embed.size()[1], -1)
        input = torch.cat((graph_embed, expanded_h), 2)
        input = input * mask.unsqueeze(2)

        Adj = torch.bmm(self.embed1(input), self.embed2(input).transpose(1,2)) # b x n x n
        min_v = Adj.min(2)[0].min(1)[0]
        Adj = Adj - min_v.unsqueeze(1).unsqueeze(2).expand(Adj.size())
        Adj = Adj * mask.unsqueeze(2)

        topk = int(math.floor((graph_embed.size()[1] * self.topk)))
        topk_mat = Adj.topk(topk, 2)
        top_idx = topk_mat[1]
        #print(top_idx[0][0])
        topk_mask = torch.zeros_like(Adj).scatter_(2, top_idx, torch.ones_like(top_idx).float()) # b x n x n
        #print(topk_mask[0][0])
        Adj = Adj * topk_mask
        Adj_norm =  self._adj_norm(Adj)

        return Adj_norm

class BuildGraph(nn.Module):
    def __init__(self, opt, p_dim):
        super(BuildGraph, self).__init__()

        input_dim = p_dim + opt.rnn_size # concatenate att and h^1_t-1
        output_dim = opt.rnn_size

        self.con_att = StructureAttention(opt, input_dim, output_dim) # conditional structure attention for graph

    def forward(self, graph_embd, att_mask, hidden_state):
        # graph_embed = b x n x c
        # hidden_state = b x rnn_size
        Adj = self.con_att(graph_embd, att_mask, hidden_state) #return Adjacency matrix
        # P = Laplacian_Norm(Adj) # P = D-1/2 A~ D-1/2
        return Adj

class AttGraphCore(nn.Module):
    def __init__(self, opt):
        super(AttGraphCore, self).__init__()

        self.use_proj = opt.use_proj  # option for projection
        self.use_graph = opt.use_graph
        self.gcn_pool = opt.gcn_pool
        self.p_dim = opt.p_dim if opt.use_proj else opt.rnn_size  # dimension of the graph embed
        self.p_dim = opt.rnn_size if not self.use_graph or (self.use_graph and self.gcn_pool == 'att') else self.p_dim # keep same dim with settings in Attention module if not use graph
        self.drop_prob_lm = opt.drop_prob_lm

        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size * 2, opt.rnn_size)  # we, fc, h^2_t-1
        self.lang_lstm = nn.LSTMCell(opt.rnn_size * 2, opt.rnn_size)  # h^1_t, \hat v

        if self.use_graph:
            input_dim = self.p_dim + opt.rnn_size
            output_dim = opt.rnn_size
            self.build_graph = BuildGraph(opt, self.p_dim)
            self.layer_num = opt.layer_num
            self.gcn = nn.ModuleList()
            self.gcn.append(GCN(opt, input_dim, output_dim))
            for i in range(self.layer_num-1):
                self.gcn.append(GCN(opt, output_dim, output_dim))

            if self.gcn_pool == 'att':
                self.attention = Attention(opt)
                self.ctx2att = nn.Linear(self.p_dim, opt.att_hid_size)
        else:
            self.attention = Attention(opt)

    def _mean_pool(self,x, mask):
        return x.sum(dim=1) / (self.eps + mask.sum(dim=1, keepdim=True))

    def _max_pool(self,x):
        return x.max(dim = 1)[0] #x has gone through relu, so has no < 0 elements. no need for mask

    def forward(self, xt, fc_feats, graph_embed, p_att_feats, state, att_mask):
        prev_h = state[0][-1]
        att_lstm_input = torch.cat([prev_h, fc_feats, xt], 1)

        h_att, c_att = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))

        if self.use_graph:
            Adj = self.build_graph(graph_embed, att_mask, h_att)
            expanded_h = prev_h.unsqueeze(1).expand(-1, graph_embed.size()[1], -1)
            gcn_input = torch.cat((graph_embed, expanded_h), 2)
            gcn_input = gcn_input * att_mask.unsqueeze(2)
            for gcn_layer in self.gcn:
                gcn_input = gcn_layer(gcn_input, Adj, att_mask)
            att_graph = gcn_input.clone()
            if self.gcn_pool == 'att':
                p_att_graph = self.ctx2att(att_graph)
                att = self.attention(h_att, att_graph, p_att_graph, None)
            elif self.gcn_pool == 'mean':
                att = self._mean_pool(att_graph, att_mask)
            elif self.gcn_pool == 'max':
                att = self._max_pool(att_graph)
        else:
            #assert not p_att_feats, 'p_att_feats is None when not use graph'
            att = self.attention(h_att,graph_embed, p_att_feats, att_mask)

        lang_lstm_input = torch.cat([att, h_att], 1)
        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1]))

        output = F.dropout(h_lang, self.drop_prob_lm, self.training)
        state = (torch.stack([h_att, h_lang]), torch.stack([c_att, c_lang]))

        return output, state

