import json
import os, sys

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import sys, os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import argparse, time, pickle

from yaml import parse

# from dataset.dataset import LEMMA, collate_func
from dataset.hcrn_dataset import LEMMA, collate_func
from utils.utils import ReasongingTypeAccCalculator
import model.HCRN.model.HCRN as HCRN


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--basedir", type=str, default='data/',
                        help='where to store ckpts and logs')
    parser.add_argument("--name", type=str, default='hcrn_logs',
                        help='where to store ckpts and logs')
    
    parser.add_argument("--train_data_file_path", type=str, 
                        default='{}/formatted_train_qas_encode.json', 
                        )
    parser.add_argument("--test_data_file_path", type=str, 
                        default='{}/formatted_test_qas_encode.json', 
                        )
    parser.add_argument("--val_data_file_path", type=str, 
                        default='{}/formatted_val_qas_encode.json', 
                        )
    parser.add_argument("--feature_base", type=str,
                        default="data/egotasqa/hcrn_data")
    parser.add_argument('--answer_set_path', type=str, default='{}/answer_set.txt')

    parser.add_argument("--batch_size", type=int, default=32, )
    parser.add_argument("--nepoch", type=int, default=60,  
                        help='num of total epoches')
    parser.add_argument("--lr", type=float, default=1e-4,  
                        help='')
    
    parser.add_argument("--i_val",   type=int, default=20000, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_test",   type=int, default=4000, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_print",   type=int, default=60, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_weight", type=int, default=4000, 
                        help='frequency of weight ckpt saving')

    parser.add_argument('--output_dim', type=int, default=1)

    parser.add_argument('--test_only', default=0, type=int)
    parser.add_argument('--reload_model_path', default='', type=str, help='model_path')

    parser.add_argument('--question_pt_path', type=str, default='{}/glove.pt')
    parser.add_argument('--without_visual', type=int, default=0)

    parser.add_argument('--base_data_dir', type=str, default='data')
    args = parser.parse_args()
    return args

def train(args):
    device = args.device
    train_dataset = LEMMA(args.train_data_file_path.format(args.base_data_dir), 'train', 
                    app_feature_h5=args.app_feat_path,
                    motion_feature_h5=args.motion_feat_path)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_func)
    
    val_dataset = LEMMA(args.val_data_file_path.format(args.base_data_dir), 'val', 
                    app_feature_h5=args.app_feat_path,
                    motion_feature_h5=args.motion_feat_path)
    val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=True, collate_fn=collate_func)

    test_dataset = LEMMA(args.test_data_file_path.format(args.base_data_dir), 'test', 
                    app_feature_h5=args.app_feat_path,
                    motion_feature_h5=args.motion_feat_path)
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=True, collate_fn=collate_func)
    
    with open(args.answer_set_path.format(args.base_data_dir), 'r') as ansf:
        answers = ansf.readlines()
        args.output_dim = len(answers) # # output_dim == len(answers)

    args.vision_dim = 2048
    args.module_dim = 512
    args.word_dim = 300
    args.k_max_frame_level = 16
    args.k_max_clip_level = 8
    args.spl_resolution = 1
    vocab_dct = json.load(open('{}/lemma-qa_vocab.json'.format(args.base_data_dir), 'r'))
    args.question_type = 'none'

    model_kwargs = {
            'vision_dim': args.vision_dim, ## 2048
            'module_dim': args.module_dim, ## 512
            'word_dim': args.word_dim, ## 300
            'k_max_frame_level': args.k_max_frame_level, ## 16
            'k_max_clip_level': args.k_max_clip_level, ## 8
            'spl_resolution': args.spl_resolution, ## 1
            'vocab': vocab_dct, # # shape should be the same as glove_matrix
            'question_type': args.question_type ## 'none'
    }

    # glove_matrix = torch.rand(201, 300).to(device)
    with open(args.question_pt_path.format(args.base_data_dir), 'rb') as f:
        obj = pickle.load(f)
        glove_matrix = obj['glove']
    glove_matrix = torch.FloatTensor(glove_matrix).to(device)

    model = HCRN.HCRNNetwork(**model_kwargs).to(device)
    with torch.no_grad():
        model.linguistic_input_unit.encoder_embed.weight.set_(glove_matrix)
    
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    reload_step = 0
    if args.reload_model_path != '':
        print('reloading model from', args.reload_model_path)
        reload_step = reload(model=model, optimizer=optimizer, path=args.reload_model_path)
    
    with open('{}/all_reasoning_types.txt'.format(args.base_data_dir), 'r') as reasonf:
        all_reasoning_types = reasonf.readlines()
        all_reasoning_types = [item.strip() for item in all_reasoning_types]
    train_acc_calculator = ReasongingTypeAccCalculator(reasoning_types=all_reasoning_types)
    test_acc_calculator = ReasongingTypeAccCalculator(reasoning_types=all_reasoning_types)

    global_step = reload_step
    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    args.basedir = os.path.join(args.basedir, args.name)
    log_dir = os.path.join(args.basedir, 'events', TIMESTAMP)
    os.makedirs(log_dir)
    with open(os.path.join(log_dir, 'argument.txt'), 'w') as f:
        for key, value in vars(args).items():
            f.write('%s:%s\n'%(key, value))
            print(key, value)

    log_file = open(os.path.join(log_dir, 'log.txt'), 'w')
    writer = SummaryWriter(log_dir=log_dir)

    os.makedirs(os.path.join(args.basedir, 'ckpts'), exist_ok=True)
    pbar = tqdm(total=args.nepoch * len(train_dataloader))
    for epoch in range(args.nepoch):
        model.train()
        train_acc_calculator.reset()
        for i, (answer_encode, app_feat, motion_feat, question_encode, question_len_lst, reasoning_type_lst) in enumerate(train_dataloader):
            B = answer_encode.shape[0]
            question_len = torch.from_numpy(np.array(question_len_lst))
            answer_encode, app_feat, motion_feat, question_encode, question_len = answer_encode.to(device), app_feat.to(device), motion_feat.to(device), question_encode.to(device), question_len.to(device)

            ans_candidates = torch.rand(B, 5).to(device)
            ans_candidates_len = torch.rand(B, 5).to(device)
            # app_feat = torch.rand(B, 8, 16, 2048).to(device)
            # motion_feat = torch.rand(B, 8, 2048).to(device)
            # question = torch.ones(B, 44).long().to(device)
            # question_len = torch.ones(B).long().to(device)
            if args.without_visual:
                app_feat = torch.randn(B, 8, 16, 2048).to(device)
                motion_feat = torch.randn(B, 8, 2048).to(device)

            logits = model(ans_candidates, ans_candidates_len,
                    app_feat, motion_feat, question_encode, question_len)

            loss = criterion(logits, answer_encode.long())

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=12)
            optimizer.step()

            pred = torch.argmax(logits, dim=1)
            train_acc = sum(pred == answer_encode) / B

            train_acc_calculator.update(reasoning_type_lst, pred, answer_encode)
            
            writer.add_scalar('train/loss', loss.item(), global_step)
            writer.add_scalar('learning rates', optimizer.param_groups[0]['lr'], global_step)
            writer.add_scalar('train/acc', train_acc, global_step)

            pbar.update(1)
            global_step += 1
        
            if global_step % args.i_print  == 0:
                print(f"global_step:{global_step}, train_loss:{loss.item()}, train_acc:{train_acc}")

            if (global_step) % args.i_val == 0:
                test_acc_calculator.reset()
                val_loss, val_acc = validate(model, val_dataloader, epoch, args, acc_calculator=test_acc_calculator)
                writer.add_scalar('val/loss', val_loss.item(), global_step)
                writer.add_scalar('val/acc', val_acc, global_step)
                acc_dct = test_acc_calculator.get_acc()
                for key, value in acc_dct.items():
                    writer.add_scalar(f'val/reasoning_{key}', value, global_step)
                log_file.write(f'[VAL]: epoch: {epoch}, global_step: {global_step}\n')
                log_file.write(f'true count dct: {test_acc_calculator.true_count_dct}\nall count dct: {test_acc_calculator.all_count_dct}\n\n')


            if (global_step) % args.i_test == 0:
                test_acc_calculator.reset()
                test_loss, test_acc = validate( model, test_dataloader, epoch, args, acc_calculator=test_acc_calculator)
                writer.add_scalar('test/loss', test_loss.item(), global_step)
                writer.add_scalar('test/acc', test_acc, global_step)
                acc_dct = test_acc_calculator.get_acc()
                for key, value in acc_dct.items():
                    writer.add_scalar(f'test/reasoning_{key}', value, global_step)
                log_file.write(f'[TEST]: epoch: {epoch}, global_step: {global_step}\n')
                log_file.write(f'true count dct: {test_acc_calculator.true_count_dct}\nall count dct: {test_acc_calculator.all_count_dct}\n\n')


            if (global_step) % args.i_weight == 0 and global_step >= 20000:
                torch.save({
                    'hcrn_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    'global_step': global_step,
                }, os.path.join(args.basedir, 'ckpts', f"model_{global_step}.tar"))

        acc_dct = train_acc_calculator.get_acc()
        for key, value in acc_dct.items():
            writer.add_scalar(f'train/reasoning_{key}', value, global_step)
        log_file.write(f'[TRAIN]: epoch: {epoch}, global_step: {global_step}\n')
        log_file.write(f'true count dct: {train_acc_calculator.true_count_dct}\nall count dct: {train_acc_calculator.all_count_dct}\n\n')
        log_file.flush()


def test(args):
    device = args.device

    test_dataset = LEMMA(args.test_data_file_path.format(args.base_data_dir), 'test', 
                    app_feature_h5=args.app_feat_path,
                    motion_feature_h5=args.motion_feat_path)
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=True, collate_fn=collate_func)
    
    with open(args.answer_set_path.format(args.base_data_dir), 'r') as ansf:
        answers = ansf.readlines()
        args.output_dim = len(answers) # # output_dim == len(answers)

    args.vision_dim = 2048
    args.module_dim = 512
    args.word_dim = 300
    args.k_max_frame_level = 16
    args.k_max_clip_level = 8
    args.spl_resolution = 1
    vocab_dct = json.load(open('{}/lemma-qa_vocab.json'.format(args.base_data_dir), 'r'))
    args.question_type = 'none'

    model_kwargs = {
            'vision_dim': args.vision_dim, ## 2048
            'module_dim': args.module_dim, ## 512
            'word_dim': args.word_dim, ## 300
            'k_max_frame_level': args.k_max_frame_level, ## 16
            'k_max_clip_level': args.k_max_clip_level, ## 8
            'spl_resolution': args.spl_resolution, ## 1
            'vocab': vocab_dct, # # shape should be the same as glove_matrix
            'question_type': args.question_type ## 'none'
    }

    # glove_matrix = torch.rand(201, 300).to(device)
    with open(args.question_pt_path.format(args.base_data_dir), 'rb') as f:
        obj = pickle.load(f)
        glove_matrix = obj['glove']
    glove_matrix = torch.FloatTensor(glove_matrix).to(device)

    model = HCRN.HCRNNetwork(**model_kwargs).to(device)
    with torch.no_grad():
        model.linguistic_input_unit.encoder_embed.weight.set_(glove_matrix)
    
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    reload_step = 0
    if args.reload_model_path != '':
        print('reloading model from', args.reload_model_path)
        reload_step = reload(model=model, optimizer=optimizer, path=args.reload_model_path)
    
    with open('{}/all_reasoning_types.txt'.format(args.base_data_dir), 'r') as reasonf:
        all_reasoning_types = reasonf.readlines()
        all_reasoning_types = [item.strip() for item in all_reasoning_types]

    test_acc_calculator = ReasongingTypeAccCalculator(reasoning_types=all_reasoning_types)
    test_acc_calculator.reset()
    test_loss, test_acc = validate(model=model, val_loader=test_dataloader, epoch=0, args=args, acc_calculator=test_acc_calculator)
    acc_dct = test_acc_calculator.get_acc()
    for key, value in acc_dct.items():
        print(f"{key} acc:{value}")
    print('TOTAL TEST ACC:', test_acc.item())


def validate( model, val_loader, epoch, args, acc_calculator):
    model.eval()
    all_acc = 0
    all_loss = 0
    batch_size = args.batch_size
    acc_calculator.reset()

    starttime = time.time()
    with torch.no_grad():
        for i, (answer_encode, app_feat, motion_feat, question_encode, question_len_lst, reasoning_type_lst) in enumerate(tqdm(val_loader)):
            B = answer_encode.shape[0]
            question_len = torch.from_numpy(np.array(question_len_lst))
            answer_encode, app_feat, motion_feat, question_encode, question_len = answer_encode.to(device), app_feat.to(device), motion_feat.to(device), question_encode.to(device), question_len.to(device)

            ans_candidates = torch.rand(B, 5).to(device)
            ans_candidates_len = torch.rand(B, 5).to(device)

            if args.without_visual:
                app_feat = torch.randn(B, 8, 16, 2048).to(device)
                motion_feat = torch.randn(B, 8, 2048).to(device)

            logits = model(ans_candidates, ans_candidates_len,
                    app_feat, motion_feat, question_encode, question_len)

            all_loss += nn.CrossEntropyLoss().to(device)(logits, answer_encode.long())

            pred = torch.argmax(logits, dim=1)
            test_acc = sum(pred == answer_encode) / B
            all_acc += test_acc

            acc_calculator.update(reasoning_type_lst, pred, answer_encode)

    print('validating cost', time.time() - starttime, 's')
    all_loss /= len(val_loader)
    all_acc /= len(val_loader)
    model.train()
    return all_loss, all_acc


def reload( model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['hcrn_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    global_step = checkpoint['global_step']
    # model.eval()
    return global_step


if __name__ =='__main__':
    args = parse_args()

    device_id = 0
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    args.device = device
    args.app_feat_path = os.path.join(args.feature_base, "lemma-qa_appearance_feat.h5")
    args.motion_feat_path = os.path.join(args.feature_base, "lemma-qa_motion_feat.h5")
    # set random seed
    torch.manual_seed(666)
    np.random.seed(666)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(666)

    if args.test_only:
        print('test only!')
        print('loading model from', args.reload_model_path)
        test(args)
    else:
        print('start training...')
        train(args)
