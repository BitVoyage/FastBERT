#Auth:zhanglusheng@outlook.com
#Implementation of FastBERT, paper refer:https://arxiv.org/pdf/2004.02178.pdf

import argparse
import json
import time
import logging

import torch
import torch.utils.data as data
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter

from model_define.model_fastbert import FastBertModel, BertConfig
from data_utils.dataset_preparing import PrepareDataset, TextCollate
import torch.nn.functional as F
from utils import load_json_config, init_bert_adam_optimizer, load_saved_model, save_model

#随机数固定，RE-PRODUCIBLE
seed = 9999
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
debug_break = False


def eval_model(train_stage, master_gpu_id, model, dataset, batch_size=1,
               use_cuda=False, num_workers=1):
    global global_step
    global debug_break
    model.eval()
    dataloader = data.DataLoader(dataset=dataset,
                                 collate_fn=TextCollate(dataset),
                                 pin_memory=use_cuda,
                                 batch_size=batch_size,
                                 num_workers=num_workers,
                                 shuffle=False)
    total_loss = 0.0
    correct_sum = 0
    proc_sum = 0
    num_sample = dataloader.dataset.__len__()
    num_batch = dataloader.__len__()
    predicted_probs = []
    true_labels = []
    logging.info("Evaluating Model...")
    infos = []
    for step, batch in enumerate(tqdm(dataloader, unit="batch", ncols=100, desc="Evaluating process: ")):
        texts = batch["texts"]
        tokens = batch["tokens"].cuda(master_gpu_id) if use_cuda else batch["tokens"]
        segment_ids = batch["segment_ids"].cuda(master_gpu_id) if use_cuda else batch["segment_ids"]
        attn_masks = batch["attn_masks"].cuda(master_gpu_id) if use_cuda else batch["attn_masks"]
        labels = batch["labels"].cuda(master_gpu_id) if use_cuda else batch["labels"]
        with torch.no_grad():
            loss, logits = model(tokens, token_type_ids=segment_ids, attention_mask=attn_masks, labels=labels,
                            training_stage=train_stage, inference=False)
        loss = loss.mean()
        loss_val = loss.item()
        total_loss += loss_val
        #writer.add_scalar('eval/loss', total_loss/num_batch, global_step)
        if debug_break and step > 50:
            break
        if train_stage == 0:
            _, top_index = logits.topk(1)
            correct_sum += (top_index.view(-1) == labels).sum().item()
            proc_sum += labels.shape[0]
    logging.info('eval total avg loss:%s', format(total_loss/num_batch, "0.4f"))
    if train_stage == 0:
        logging.info("Correct Prediction: " + str(correct_sum))
        logging.info("Accuracy Rate: " + format(correct_sum / proc_sum, "0.4f"))


def train_epoch(train_stage, master_gpu_id, model, optimizer, dataloader, gradient_accumulation_steps, use_cuda, dump_info=False):
    global global_step
    global debug_break
    model.train()
    dataloader.dataset.is_training = True

    total_loss = 0.0
    correct_sum = 0
    proc_sum = 0
    num_batch = dataloader.__len__()
    num_sample = dataloader.dataset.__len__()
    pbar = tqdm(dataloader, unit="batch", ncols=100)
    pbar.set_description('train step loss')
    for step, batch in enumerate(pbar):
        texts = batch["texts"]
        tokens = batch["tokens"].cuda(master_gpu_id) if use_cuda else batch["tokens"]
        segment_ids = batch["segment_ids"].cuda(master_gpu_id) if use_cuda else batch["segment_ids"]
        attn_masks = batch["attn_masks"].cuda(master_gpu_id) if use_cuda else batch["attn_masks"]
        labels = batch["labels"].cuda(master_gpu_id) if use_cuda else batch["labels"]
        loss, logits = model(tokens, token_type_ids=segment_ids, attention_mask=attn_masks, labels=labels,
                        training_stage=train_stage, inference=False)
        if train_stage == 0 and dump_info:
            probs = F.softmax(logits, dim=-1)
        loss = loss.mean()
        if gradient_accumulation_steps > 1:
            loss /= gradient_accumulation_steps
        loss.backward()
        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            model.zero_grad()
        loss_val = loss.item()
        total_loss += loss_val
        if train_stage == 0:
            _, top_index = logits.topk(1)
            correct_sum += (top_index.view(-1) == labels).sum().item()
            proc_sum += labels.shape[0]

        #writer.add_scalar('train/loss', loss_val, global_step)
        pbar.set_description('train step loss '+format(loss_val, "0.4f"))
        if debug_break and step > 50:
            break
    pbar.close()

    logging.info("Total Training Samples:%s ", num_sample)
    logging.info('train total avg loss:%s', total_loss/num_batch)
    if train_stage == 0:
        logging.info("Correct Prediction: " + str(correct_sum))
        logging.info("Accuracy Rate: " + format(correct_sum / proc_sum, "0.4f"))
    return total_loss / num_batch


def train_model(train_stage, save_model_path, master_gpu_id, model, optimizer, epochs, 
                train_dataset, eval_dataset,
                batch_size=1, gradient_accumulation_steps=1,
                use_cuda=False, num_workers=1):
    logging.info("Start Training".center(60, "="))
    training_dataloader = data.DataLoader(dataset=train_dataset,
                                          collate_fn=TextCollate(train_dataset),
                                          pin_memory=use_cuda,
                                          batch_size=batch_size,
                                          num_workers=num_workers,
                                          shuffle=True)
    for epoch in range(1, epochs + 1):
        logging.info("Training Epoch: " + str(epoch))
        avg_loss = train_epoch(train_stage, master_gpu_id, model, optimizer, training_dataloader,
                               gradient_accumulation_steps, use_cuda)
        logging.info("Average Loss: " + format(avg_loss, "0.4f"))
        eval_model(train_stage, master_gpu_id, model, eval_dataset, batch_size=batch_size, use_cuda=use_cuda, num_workers=num_workers)
        save_model(save_model_path, model, epoch)


def main(args):
    config = load_json_config(args.model_config_file)
    logging.info(json.dumps(config, indent=2, sort_keys=True))
    logging.info("Load HyperParameters Done")

    #---------------------MODEL GRAPH INIT--------------------------#
    bert_config = BertConfig.from_json_file(config.get("bert_config_path"))
    if args.run_mode == 'train':
        #初始训练
        if args.train_stage == 0:
            model = FastBertModel.load_pretrained_bert_model(bert_config, config,
                        pretrained_model_path=config.get("bert_pretrained_model_path"))
            save_model_path_for_train = args.save_model_path
        #蒸馏训练
        elif args.train_stage == 1:
            model = FastBertModel(bert_config, config)
            load_saved_model(model, args.save_model_path)
            save_model_path_for_train = args.save_model_path_distill

            #Freeze Part Model
            for name, p in model.named_parameters():
                if "branch_classifier" not in name:
                    p.requires_grad = False
            logging.info("Main Graph and Teacher Classifier Freezed, Student Classifier will Distilling")
        else:
            raise RuntimeError('Operation Train Stage(0 or 1) not Legal')

    elif args.run_mode == 'eval':
        model = FastBertModel(bert_config, config)
        load_saved_model(model, args.save_model_path)
    else:
        raise RuntimeError('Operation Mode not Legal')
        
    logging.info(model)
    logging.info("Initialize Model Done".center(60, "="))

    #---------------------GPU SETTING--------------------------#
    use_cuda = args.gpu_ids != '-1'
    if len(args.gpu_ids) == 1 and use_cuda:
        master_gpu_id = int(args.gpu_ids)
        model = model.cuda(int(args.gpu_ids)) if use_cuda else model
    elif use_cuda:
        gpu_ids = [int(each) for each in args.gpu_ids.split(",")]
        master_gpu_id = gpu_ids[0]
        model = model.cuda(gpu_ids[0])
        logging.info("Start multi-gpu dataparallel training/evaluating...")
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    else:
        master_gpu_id = None

    #-----------------------Dataset Init --------------------------------#
    if args.train_data:
        train_dataset = PrepareDataset(vocab_file=config.get("vocab_file"),
                             max_seq_len=config.get("max_seq_len"),
                             num_class=config.get("num_class"),
                             data_file=args.train_data)
        logging.info("Load Training Dataset Done, Total training line: %s", train_dataset.__len__())
    if args.eval_data:
        eval_dataset = PrepareDataset(vocab_file=config.get("vocab_file"),
                             max_seq_len=config.get("max_seq_len"),
                             num_class=config.get("num_class"),
                             data_file=args.eval_data)
        logging.info("Load Eval Dataset Done, Total eval line: %s", eval_dataset.__len__())

    #-----------------------Running Mode Start--------------------------------#
    if args.run_mode == "train":
        optimizer = init_bert_adam_optimizer(model, train_dataset.__len__(), args.epochs, args.batch_size,
                                         config.get("gradient_accumulation_steps"),
                                         config.get("init_lr"), config.get("warmup_proportion"))
        train_model(args.train_stage,
                    save_model_path_for_train,
                    master_gpu_id, model,
                    optimizer, args.epochs, 
                    train_dataset, eval_dataset,
                    batch_size=args.batch_size,
                    gradient_accumulation_steps=config.get("gradient_accumulation_steps"),
                    use_cuda=use_cuda, num_workers=args.data_load_num_workers)
    elif args.run_mode == "eval":
        eval_model(args.train_stage, master_gpu_id, model, eval_dataset, batch_size=args.batch_size, 
                   use_cuda=use_cuda, num_workers=args.data_load_num_workers)
    else:
        raise RuntimeError("Mode not support: " + args.mode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Textclassification training script arguments.")
    parser.add_argument("--model_config_file", dest="model_config_file", action="store",
                          help="The path of configuration json file.")

    parser.add_argument("--run_mode", dest="run_mode", action="store", default="train",
                                help="Running mode: train or eval")
    parser.add_argument("--train_stage", dest="train_stage", action="store", type=int, default=0,
                        help="Running train stage, 0 or 1.")

    parser.add_argument("--save_model_path", dest="save_model_path", action="store",
                          help="The path of trained checkpoint model.")
    parser.add_argument("--save_model_path_distill", dest="save_model_path_distill", action="store",
                          help="The path of trained checkpoint model.")

    parser.add_argument("--train_data", dest="train_data", action="store", help="")
    parser.add_argument("--eval_data", dest="eval_data", action="store", help="")

    parser.add_argument("--inference_speed", dest="inference_speed", action="store", 
                            type=float, default=1.0, help="")

    # -1 for NO GPU
    parser.add_argument("--gpu_ids", dest="gpu_ids", action="store", default="0", 
                                help="Device ids of used gpus, split by ',' , IF -1 then no gpu")

    parser.add_argument("--epochs", dest="epochs", action="store", type=int, default=1, help="")
    parser.add_argument("--batch_size", dest="batch_size", action="store",type=int, default=32, help="")
    parser.add_argument("--data_load_num_workers", dest="data_load_num_workers", action="store",type=int, default=1, help="")
    parser.add_argument("--debug_break", dest="debug_break", action="store", type=int, default=0,
                        help="Running debug_break, 0 or 1.")

    parsed_args = parser.parse_args()
    debug_break = (parsed_args.debug_break == 1)
    main(parsed_args)
