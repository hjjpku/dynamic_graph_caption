from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
from sklearn.decomposition import IncrementalPCA
import misc.utils as utils

import copy
import math
import numpy as np
#from matplotlib import pyplot as plt
#from matplotlib.pyplot import savefig

from .CaptionModel import CaptionModel
from .AttModel import pack_wrapper, AttModel, Attention


class NewTopDown(AttModel):
    def __init__(self, opt):
        super(NewTopDown, self).__init__(opt)
        self.opt = opt
        del self.fc_embed
        self.num_layers = 2  # keep the same with topdown
        self.core = NewTopDownCore(opt)

    def _prepare_feature(self, fc_feats, att_feats, att_masks):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)

        # embed fc and att feats
        # dealing with the mask issue for bn layers in att_embed
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)
        # for attention coefficients computation: att_i,t = w_1 tanh(w_2v_i + w_3h_t)
        # Project the attention feats first (i.e., w_2v_i) to reduce memory and computation comsumptions.
        p_att_feats = self.ctx2att(att_feats)
        return fc_feats, att_feats, p_att_feats, att_masks

class NewTopDownCore(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(NewTopDownCore, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm

        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size, opt.rnn_size) # we, h^2_t-1
        self.lang_lstm = nn.LSTMCell(opt.rnn_size * 2, opt.rnn_size) # h^1_t, \hat v
        self.attention = Attention(opt)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state, att_masks=None):
        prev_h = state[0][-1]
        att_lstm_input = torch.cat([prev_h, xt], 1)

        h_att, c_att = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))

        att = self.attention(h_att, att_feats, p_att_feats, att_masks)

        lang_lstm_input = torch.cat([att, h_att], 1)
        # lang_lstm_input = torch.cat([att, F.dropout(h_att, self.drop_prob_lm, self.training)], 1) ?????

        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1]))

        output = F.dropout(h_lang, self.drop_prob_lm, self.training)
        state = (torch.stack([h_att, h_lang]), torch.stack([c_att, c_lang]))

        return output, state