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
from utils import load_json_config, init_bert_adam_optimizer, load_saved_model, save_model, eval_pr

#随机数固定，RE-PRODUCIBLE
seed = 9999
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
debug_break = False

def infer_model(master_gpu_id, model, dataset,
               use_cuda=False, num_workers=1, inference_speed=None, dump_info_file=None):
    global global_step
    global debug_break
    model.eval()
    infer_dataloader = data.DataLoader(dataset=dataset,
                                      collate_fn=TextCollate(dataset),
                                      pin_memory=use_cuda,
                                      batch_size=1,
                                      num_workers=num_workers,
                                      shuffle=False)
    correct_sum = 0
    num_sample = infer_dataloader.dataset.__len__()
    predicted_probs = []
    true_labels = []
    infos = []
    logging.info("Inference Model...")
    cnt = 0 
    stime_all = time.time()
    for step, batch in enumerate(tqdm(infer_dataloader, unit="batch", ncols=100, desc="Inference process: ")):
        texts = batch["texts"]
        tokens = batch["tokens"].cuda(master_gpu_id) if use_cuda else batch["tokens"]
        segment_ids = batch["segment_ids"].cuda(master_gpu_id) if use_cuda else batch["segment_ids"]
        attn_masks = batch["attn_masks"].cuda(master_gpu_id) if use_cuda else batch["attn_masks"]
        labels = batch["labels"].cuda(master_gpu_id) if use_cuda else batch["labels"]
        with torch.no_grad():
            probs, layer_idxes, uncertain_infos = model(tokens, token_type_ids=segment_ids, attention_mask=attn_masks,
                    inference=True, inference_speed=inference_speed)
        _, top_index = probs.topk(1)

        correct_sum += (top_index.view(-1) == labels).sum().item()
        cnt += 1
        if cnt == 1:
            stime = time.time()
        if dump_info_file != None:
            for label, pred, prob, layer_i, text in zip(labels, top_index.view(-1), probs, [layer_idxes], texts):
                infos.append((label.item(), pred.item(), prob.cpu().numpy(), layer_i, text))
        if debug_break and step > 50:
            break
    
    time_per = (time.time() - stime)/(cnt - 1)
    time_all = time.time() - stime_all
    acc = format(correct_sum / num_sample, "0.4f")
    logging.info("speed_arg:%s, time_per_record:%s, acc:%s, total_time:%s", 
                    inference_speed, format(time_per, '0.4f'), acc, format(time_all, '0.4f'))
    if dump_info_file != None and len(dump_info_file) != 0:
        with open(dump_info_file, 'w') as fw:
            for label, pred, prob, layer_i, text in infos:
                #fw.write('\t'.join([str(label), str(pred), str(prob), str(layer_i), text])+'\n')
                fw.write('\t'.join([str(label), str(pred), str(layer_i), text])+'\n')

    labels_pr = [info[0] for info in infos]
    preds_pr = [info[1] for info in infos]
    precise, recall = eval_pr(labels_pr, preds_pr)
    logging.info("precise:%s, recall:%s", format(precise, '0.4f'), format(recall, '0.4f'))



def main(args):
    config = load_json_config(args.model_config_file)
    logging.info(json.dumps(config, indent=2, sort_keys=True))
    logging.info("Load HyperParameters Done")

    #---------------------MODEL GRAPH INIT--------------------------#
    bert_config = BertConfig.from_json_file(config.get("bert_config_path"))
    model = FastBertModel(bert_config, config)
    load_saved_model(model, args.save_model_path)
        
    logging.info(model)
    logging.info("Initialize Model Done".center(60, "="))

    #-----------GPU SETTING, INFER Only Support Max 1 GPU-----------#
    use_cuda = args.gpu_ids != '-1'
    if len(args.gpu_ids) == 1 and use_cuda:
        master_gpu_id = int(args.gpu_ids)
        model = model.cuda(int(args.gpu_ids)) if use_cuda else model
    elif not use_cuda:
        master_gpu_id = None
    else:
        raise RuntimeError("GPU Mode not support, INFER Only Support Max 1 GPU: " + args.gpu_ids)

    #-----------------------Dataset Init---------------------------#
    infer_dataset = PrepareDataset(vocab_file=config.get("vocab_file"),
                         max_seq_len=config.get("max_seq_len"),
                         num_class=config.get("num_class"),
                         data_file=args.infer_data)
    logging.info("Load INFER Dataset Done, Total eval line: %s", infer_dataset.__len__())

    #-----------------------Running Mode Start, Batch Size Only Support 1--------------------------------#
    infer_model(master_gpu_id, model, infer_dataset, 
               use_cuda=use_cuda, num_workers=args.data_load_num_workers, 
               inference_speed=args.inference_speed, dump_info_file=args.dump_info_file) 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Textclassification training script arguments.")
    parser.add_argument("--model_config_file", dest="model_config_file", action="store",
                          help="The path of configuration json file.")

    parser.add_argument("--save_model_path", dest="save_model_path", action="store",
                          help="The path of trained checkpoint model.")

    parser.add_argument("--infer_data", dest="infer_data", action="store", help="")
    parser.add_argument("--dump_info_file", dest="dump_info_file", action="store", help="")

    parser.add_argument("--inference_speed", dest="inference_speed", action="store", 
                            type=float, default=1.0, help="")

    # -1 for NO GPU
    parser.add_argument("--gpu_ids", dest="gpu_ids", action="store", default="0", 
                                help="Device ids of used gpus, split by ',' , IF -1 then no gpu")

    parser.add_argument("--data_load_num_workers", dest="data_load_num_workers", action="store",type=int, default=1, help="")
    parser.add_argument("--debug_break", dest="debug_break", action="store", type=int, default=0,
                        help="Running debug_break, 0 or 1.")

    parsed_args = parser.parse_args()
    debug_break = (parsed_args.debug_break == 1)
    main(parsed_args)
