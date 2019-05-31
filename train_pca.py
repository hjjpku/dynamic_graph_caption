from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from sklearn.decomposition import IncrementalPCA

import time
import os
from six.moves import cPickle
import traceback

import opts
import models
from dataloader import *
#import skimage.io
import eval_utils
import misc.utils as utils
from misc.rewards import init_scorer, get_self_critical_reward
from misc.loss_wrapper import LossWrapper


try:
    import tensorboardX as tb
except ImportError:
    print("tensorboardX is not installed")
    tb = None

def add_summary_value(writer, key, value, iteration):
    if writer:
        writer.add_scalar(key, value, iteration)

def train(opt):
    # Deal with feature things before anything
    opt.use_fc, opt.use_att = True, True
    if opt.use_box: opt.att_feat_size = opt.att_feat_size + 5
    print(opt)
    loader = DataLoader(opt)
    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length


    infos = {}
    histories = {}
    infos['iter'] = 0
    infos['epoch'] = 0
    infos['iterators'] = loader.iterators
    infos['split_ix'] = loader.split_ix
    infos['vocab'] = loader.get_vocab()
    infos['opt'] = opt

    iteration = infos.get('iter', 0)
    epoch = infos.get('epoch', 0)

    loader.iterators = infos.get('iterators', loader.iterators)
    loader.split_ix = infos.get('split_ix', loader.split_ix)


    pca = IncrementalPCA(n_components=opt.att_feat_size, whiten=True, copy=True, batch_size=opt.batch_size)

    epoch_done = True

    try:
        while True:

            start = time.time()
            # Load data from train split (0)
            data = loader.get_batch('train')
            print('Read data:', time.time() - start)

            start = time.time()

            att_feats = data['att_feats']
            att_masks = data['att_masks']
            ###
            tmp_feats = torch.zeros(opt.batch_size * att_feats.size()[1], att_feats.size()[2])
            count = 0
            for i in range(opt.batch_size):
                offset = att_masks[i].sum().item()
                offset = int(offset)
                tmp = att_feats[i][0:offset]
                tmp_feats[count:count+offset] = tmp
                count = count + offset
            att_feats = tmp_feats[0:count]

            rd = int(count/opt.num_k)

            feat = torch.zeros(rd, opt.num_k*opt.att_feat_size)
            for i in range(rd):
                feat[i] = torch.cat(tuple(att_feats[i*opt.num_k:(i+1)*opt.num_k]),dim=0)
            feat = feat[0:opt.batch_size]
            feat = feat.numpy()
            pca.partial_fit(feat)
            end = time.time()

            print("iter {} (epoch {}),  time/batch = {:.3f}" \
                    .format(iteration, epoch,  end - start))

            # Update the iteration and epoch
            iteration += 1


            # Stop if reaching max epochs
            if epoch >= opt.max_epochs and opt.max_epochs != -1:
                break
    except (RuntimeError, KeyboardInterrupt):
        print('Save ckpt on exception ...')

        stack_trace = traceback.format_exc()
        print(stack_trace)

    np.save(opt.PCA_dir+ str(opt.num_k) +'.npy',[pca])

opt = opts.parse_opt()
train(opt)
