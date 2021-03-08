# coding: utf-8

import torch.nn as nn
import torch
import os
import torch
import numpy as np
import random
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from DataLoader import MyDataset, collate_fn
import time
import json
from transformers import *
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
import argparse
# from transformers.modeling_roberta import RobertaModel
from transformers.modeling_bert import BertModel,BertForMultipleChoice

from transformers.modeling_xlnet import XLNetForMultipleChoice,XLNetModel,XLNetForQuestionAnswering
from transformers.tokenization_xlnet import XLNetTokenizer


def parse_args():

    args = argparse.ArgumentParser()
    args.add_argument("--pretrain_path", default=r"E:\PythonProgram\NLPCode\PretrainModel\chinese_xlnet_base_pytorch")
    # args.add_argument("--pretrain_path",default=r"E:\PythonProgram\NLPCode\PretrainModel\chinese_bert_base")
    args.add_argument("--fold_num",default=5,help="几折交叉验证")
    args.add_argument("--seed",default=101)
    args.add_argument("--context_max_len",default=800)
    args.add_argument("--choice_max_len",default=120)
    args.add_argument("--epochs",default=8)
    args.add_argument("--train_bs",default=2)
    args.add_argument("--valid_bs",default=12)
    args.add_argument("--lr",default=2e-5)
    args.add_argument("--accum_iter",default=2)
    args.add_argument("--weight_decay",default=1e-4)
    args.add_argument("--filename",default=r"data-public/train_v2.json")
    args.add_argument("--num_workers",default=0)
    return args.parse_args()


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




class AverageMeter: #为了tqdm实时显示loss和acc
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def test_model(model, val_loader,device,criterion):  # 验证
    model.eval()

    losses = AverageMeter()
    accs = AverageMeter()

    y_truth, y_pred = [], []

    with torch.no_grad():
        tk = tqdm(val_loader, total=len(val_loader), position=0, leave=True)
        for idx, (input_ids, attention_mask, token_type_ids, y) in enumerate(tk):
            input_ids, attention_mask, token_type_ids, y = input_ids.to(device), attention_mask.to(
                device), token_type_ids.to(device), y.to(device).long()

            output = model(input_ids, attention_mask, token_type_ids).logits

            y_truth.extend(y.cpu().numpy())
            y_pred.extend(output.argmax(1).cpu().numpy())

            loss = criterion(output, y)

            acc = (output.argmax(1) == y).sum().item() / y.size(0)

            losses.update(loss.item(), y.size(0))
            accs.update(acc, y.size(0))

            tk.set_postfix(loss=losses.avg, acc=accs.avg)

    return losses.avg, accs.avg

def train_model(model, train_loader,optimizer,criterion,scaler,scheduler,
                device,args):  # 训练一个epoch
    model.train()

    losses = AverageMeter()
    accs = AverageMeter()

    optimizer.zero_grad()

    tk = tqdm(train_loader, total=len(train_loader), position=0, leave=True)

    for step, (input_ids, attention_mask, token_type_ids, y) in enumerate(tk):
        input_ids, attention_mask, token_type_ids, y = input_ids.to(device), attention_mask.to(
            device), token_type_ids.to(device), y.to(device).long()

        with autocast():  # 使用半精度训练
            outputs = model(input_ids,token_type_ids, attention_mask)
            output = outputs[0]

            loss = criterion(output, y)
            scaler.scale(loss).backward()
            # loss.backward()

            if ((step + 1) % args.accum_iter == 0) or ((step + 1) == len(train_loader)):  # 梯度累加
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

        acc = (output.argmax(1) == y).sum().item() / y.size(0)

        losses.update(loss.item() * args.accum_iter, y.size(0))
        accs.update(acc, y.size(0))

        tk.set_postfix(loss=losses.avg, acc=accs.avg)

    return losses.avg, accs.avg

class MultipleChoice(BertForMultipleChoice):
    def __init__(self, config):
        super().__init__(config)
        self.classifier = nn.Linear(config.hidden_size, 4)


def main(args):
    device = torch.device("cuda")
    seed_everything(args.seed)
    with open(args.filename, 'r', encoding='utf-8') as f:
        data = f.read()

    lines = json.loads(data)
    data = []
    labels = []
    for i in range(len(lines)):
        line = lines[i]
        content = line['Content']
        questions = line['Questions']
        for question in questions:
            question['Content'] = content
            data.append(question)
            labels.append(question['Answer'])


    folds = StratifiedKFold(n_splits=args.fold_num, shuffle=True, random_state=args.seed) \
        .split(np.arange(len(data)), np.array(labels))  # 五折交叉验证


    cv = []  # 保存每折的最佳准确率

    for fold, (trn_idx, val_idx) in enumerate(folds):

        train = [data[i] for i in trn_idx]
        val = [data[i] for i in val_idx]

        train_set = MyDataset(train,args)
        val_set = MyDataset(val,args)

        train_loader = DataLoader(train_set, batch_size=args.train_bs, collate_fn=collate_fn, shuffle=True,
                                  num_workers=args.num_workers)
        val_loader = DataLoader(val_set, batch_size=args.valid_bs, collate_fn=collate_fn, shuffle=False,
                                num_workers=args.num_workers)

        best_acc = 0

        model = XLNetForMultipleChoice.from_pretrained(args.pretrain_path)
        # model = BertForMultipleChoice.from_pretrained(args.pretrain_path)
        model.to(device)# 模型

        scaler = GradScaler()
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)  # AdamW优化器
        criterion = nn.CrossEntropyLoss()
        scheduler = get_cosine_schedule_with_warmup(optimizer, len(train_loader) // args.accum_iter,
                                                    args.epochs * len(train_loader) // args.accum_iter)
        # get_cosine_schedule_with_warmup策略，学习率先warmup一个epoch，然后cos式下降

        for epoch in range(args.epochs):
            print('epoch:', epoch)
            time.sleep(0.2)

            train_loss, train_acc = train_model(model, train_loader,optimizer,criterion,scaler,scheduler,device,args)
            val_loss, val_acc = test_model(model, val_loader,device,criterion)

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), '{}_fold_{}.pt'.format(args['model'].split('/')[-1], fold))

        cv.append(best_acc)

if __name__ == '__main__':
    args = parse_args()
    main(args)
