#!/usr/bin/env python
# coding:utf8
# Provide some utilities for task finetune scripts.


import collections
import json
import os
import re
import torch
import torch.utils.data as data
from model_define.optimization import BERTAdam
import logging


def eval_pr(labels, preds):
    TP, TN, FP, FN = 0, 0, 0, 0
    for label, pred in zip(labels, preds):
        if label == 1 and pred == 1:
            TP += 1
        elif label == 0 and pred == 0:
            TN += 1
        elif label == 1 and pred == 0:
            FP += 1
        elif label == 0 and pred == 1:
            FN += 1
    #print('TP', TP)
    #print('TN', TN)
    #print('FP', FP)
    #print('FN', FN)
    precise = TP/(TP+FN+0.0001)
    recall = TP/(TP+FP+0.0001)
    return precise, recall


def load_dict_from_file(file_dir):
    """
    Load the dictionary from file, every line is a key info
    in file, and each line must at least have one word for KEY.
    The value is the index of KEY in file.
    """
    d = collections.OrderedDict()
    index = 0
    with open(file_dir, "r", encoding="utf-8") as f:
        for line in f:
            token = line.strip().split("\t")[0]
            d[token] = index
            index += 1
    return d


def load_json_config(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def save_model(path, model, epoch):
    if not os.path.exists(path):
        os.mkdir(path)
    model_weight = model.state_dict()
    new_state_dict = collections.OrderedDict()
    for k, v in model_weight.items():
        if k.startswith("module"):
            name = k[7:]
        else:
            name = k
        new_state_dict[name] = v
    model_name = "Epoch_" + str(epoch) + ".bin"
    model_file = os.path.join(path, model_name)
    torch.save(new_state_dict, model_file)
    logging.info('dumped model file to:%s', model_file)


def load_saved_model(model, saved_model_path, model_file=None):
    if model_file == None:
        files = os.listdir(saved_model_path)
        max_idx = 0
        max_fname = ''
        for fname in files:
            idx = re.sub('Epoch_|\.bin', '',fname)
            if int(idx) > max_idx:
                max_idx = int(idx)
                max_fname = fname
        model_file = max_fname
    model_file = os.path.join(saved_model_path, model_file)
    model_weight = torch.load(model_file, map_location="cpu")
    new_state_dict = collections.OrderedDict()
    for k, v in model_weight.items():
        if k.startswith("module"):
            name = k[7:]
        else:
            name = k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    logging.info('loaded saved model file:%s', model_file)
    return model_file


def init_bert_adam_optimizer(model, training_data_len, epoch, batch_size,
                             gradient_accumulation_steps, init_lr, warmup_proportion):
    no_decay = ["bias", "gamma", "beta"]
    optimizer_parameters = [
        {"params" : [p for name, p in model.named_parameters() \
            if name not in no_decay], "weight_decay_rate" : 0.01},
        {"params" : [p for name, p in model.named_parameters() \
            if name in no_decay], "weight_decay_rate" : 0.0}
    ]
    num_train_steps = int(training_data_len / batch_size / \
        gradient_accumulation_steps * epoch)
    optimizer = BERTAdam(optimizer_parameters,
                         lr=init_lr,
                         warmup=warmup_proportion,
                         t_total=num_train_steps)
    return optimizer

