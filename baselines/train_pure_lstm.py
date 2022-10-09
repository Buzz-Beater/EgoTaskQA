from pyexpat import model
from statistics import mode
from unittest import TestLoader
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import sys, os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import argparse, time

from dataset.dataset import LEMMA, collate_func
import model.pure_lstm as pure_lstm
from utils.utils import ReasongingTypeAccCalculator

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--basedir", type=str, default='data/',
                        help='where to store ckpts and logs')
    parser.add_argument("--name", type=str, default='pure_lstm_logs',
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
    parser.add_argument('--answer_set_path', type=str, default='{}/answer_set.txt')

    parser.add_argument("--batch_size", type=int, default=64, )
    parser.add_argument("--nepoch", type=int, default=200,  
                        help='num of total epoches')
    parser.add_argument("--lr", type=float, default=1e-4,  
                        help='')
    
    parser.add_argument("--i_val",   type=int, default=20000, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_test",   type=int, default=6000, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_print",   type=int, default=6, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_weight", type=int, default=6000, 
                        help='frequency of weight ckpt saving')

    parser.add_argument('--img_size', default=(224, 224))
    parser.add_argument('--num_frames_per_video', type=int, default=20)
    parser.add_argument('--cnn_modelname', type=str, default='resnet101')
    parser.add_argument('--cnn_pretrained', type=bool, default=True)
    parser.add_argument('--output_dim', type=int, default=1)
    parser.add_argument('--use_preprocessed_features', type=int, default=1)
    parser.add_argument('--feature_base_path', type=str, default='/scratch/generalvision/LEMMA/video_features')

    parser.add_argument('--test_only', default=0, type=int)
    parser.add_argument('--reload_model_path', default='', type=str, help='model_path')

    parser.add_argument('--base_data_dir', type=str, default='data')
    args = parser.parse_args()
    return args

def train(args):
    device = args.device

    train_dataset = LEMMA(args.train_data_file_path.format(args.base_data_dir), args.img_size, 'train', args.num_frames_per_video, args.use_preprocessed_features,
                         all_qa_interval_path='{}/vid_intervals.json'.format(args.base_data_dir), feature_base_path=args.feature_base_path )
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,drop_last=True, collate_fn=collate_func, pin_memory=True)
    
    val_dataset = LEMMA(args.val_data_file_path.format(args.base_data_dir), args.img_size, 'val', args.num_frames_per_video, args.use_preprocessed_features, 
                        all_qa_interval_path='{}/vid_intervals.json'.format(args.base_data_dir), feature_base_path=args.feature_base_path)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True,drop_last=True, collate_fn=collate_func)

    test_dataset = LEMMA(args.test_data_file_path.format(args.base_data_dir), args.img_size, 'test', args.num_frames_per_video, args.use_preprocessed_features,
                        all_qa_interval_path='{}/vid_intervals.json'.format(args.base_data_dir), feature_base_path=args.feature_base_path)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True,drop_last=True, collate_fn=collate_func)

    with open(args.answer_set_path.format(args.base_data_dir), 'r') as ansf:
        answers = ansf.readlines()
        args.output_dim = len(answers) # # output_dim == len(answers)

    # cnn = cnn_lstm.build_resnet(args.cnn_modelname, pretrained=args.cnn_pretrained).to(device=args.device)
    # cnn.eval() # TODO ?

    lstm = pure_lstm.pure_lstm(
        input_size=300, hidden_size=256, num_layers=6, 
        num_classes=args.output_dim,
        question_pt_path='{}/glove.pt'.format(args.base_data_dir)).to(args.device) # # vocab_size = glove_matrix.shape[0]

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(lstm.parameters(), lr=args.lr)

    with open('{}/all_reasoning_types.txt'.format(args.base_data_dir), 'r') as reasonf:
        all_reasoning_types = reasonf.readlines()
        all_reasoning_types = [item.strip() for item in all_reasoning_types]
    train_acc_calculator = ReasongingTypeAccCalculator(reasoning_types=all_reasoning_types)
    test_acc_calculator = ReasongingTypeAccCalculator(reasoning_types=all_reasoning_types)

    reload_step = 0
    if args.reload_model_path != '':
        print('reloading model from', args.reload_model_path)
        reload_step = reload(model=model, optimizer=optimizer, path=args.reload_model_path)
    
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
        lstm.train()
        train_acc_calculator.reset()
        for i, (frame_rgbs, question_encode, answer_encode, frame_features, _, question, reasoning_type_lst) in enumerate(train_dataloader):
            B, num_frame_per_video, C, H, W = frame_rgbs.shape
            frame_rgbs, question_encode, answer_encode = frame_rgbs.to(device), question_encode.to(device), answer_encode.to(device)
            if args.use_preprocessed_features:
                frame_features = frame_features.to(device)
            else:
                frame_features = torch.ones(B, num_frame_per_video, 2048)
                # frame_features = cnn(frame_rgbs.reshape(-1, C, H, W))
                # frame_features = frame_features.reshape(B, num_frame_per_video, -1)
            
            logits = lstm(question_encode)

            loss = criterion(logits, answer_encode.long())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = torch.argmax(logits, dim=1)
            train_acc = sum(pred == answer_encode) / B

            train_acc_calculator.update(reasoning_type_lst, pred, answer_encode)
            
            writer.add_scalar('train/loss', loss.item(), global_step)
            writer.add_scalar('learning rates', optimizer.param_groups[0]['lr'], global_step)
            writer.add_scalar('train/acc', train_acc, global_step)

            pbar.update(1)
            

            if global_step % args.i_print  == 0:
                print(f"global_step:{global_step}, train_loss:{loss.item()}, train_acc:{train_acc}")

            if (global_step) % args.i_val == 0:
                test_acc_calculator.reset()
                val_loss, val_acc = validate( lstm, val_dataloader, epoch, args, acc_calculator=test_acc_calculator)
                writer.add_scalar('val/loss', val_loss.item(), global_step)
                writer.add_scalar('val/acc', val_acc, global_step)
                acc_dct = test_acc_calculator.get_acc()
                for key, value in acc_dct.items():
                    writer.add_scalar(f'val/reasoning_{key}', value, global_step)
                log_file.write(f'[VAL]: epoch: {epoch}, global_step: {global_step}\n')
                log_file.write(f'true count dct: {test_acc_calculator.true_count_dct}\nall count dct: {test_acc_calculator.all_count_dct}\n\n')
                

            if (global_step) % args.i_test == 0:
                test_acc_calculator.reset()
                test_loss, test_acc = validate( lstm, test_dataloader, epoch, args, acc_calculator=test_acc_calculator)
                writer.add_scalar('test/loss', test_loss.item(), global_step)
                writer.add_scalar('test/acc', test_acc, global_step)
                acc_dct = test_acc_calculator.get_acc()
                for key, value in acc_dct.items():
                    writer.add_scalar(f'test/reasoning_{key}', value, global_step)
                log_file.write(f'[TEST]: epoch: {epoch}, global_step: {global_step}\n')
                log_file.write(f'true count dct: {test_acc_calculator.true_count_dct}\nall count dct: {test_acc_calculator.all_count_dct}\n\n')


            if (global_step) % args.i_weight == 0 and global_step >= 17000:
                torch.save({
                    'pure_lstm_state_dict': lstm.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    'global_step': global_step,
                }, os.path.join(args.basedir, 'ckpts', f"model_{global_step}.tar"))
            global_step += 1

        acc_dct = train_acc_calculator.get_acc()
        for key, value in acc_dct.items():
            writer.add_scalar(f'train/reasoning_{key}', value, global_step)
        log_file.write(f'[TRAIN]: epoch: {epoch}, global_step: {global_step}\n')
        log_file.write(f'true count dct: {train_acc_calculator.true_count_dct}\nall count dct: {train_acc_calculator.all_count_dct}\n\n')
        log_file.flush()


def test(args):
    device = args.device

    
    with open(args.answer_set_path.format(args.base_data_dir), 'r') as ansf:
        answers = ansf.readlines()
        args.output_dim = len(answers) # # output_dim == len(answers)

    # cnn = cnn_lstm.build_resnet(args.cnn_modelname, pretrained=args.cnn_pretrained).to(device=args.device)
    # cnn.eval() # TODO ?

    lstm = pure_lstm.pure_lstm(
        input_size=300, hidden_size=256, num_layers=6, 
        num_classes=args.output_dim,
        question_pt_path='{}/glove.pt'.format(args.base_data_dir)).to(args.device) # # vocab_size = glove_matrix.shape[0]

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(lstm.parameters(), lr=args.lr)

    with open('{}/all_reasoning_types.txt'.format(args.base_data_dir), 'r') as reasonf:
        all_reasoning_types = reasonf.readlines()
        all_reasoning_types = [item.strip() for item in all_reasoning_types]
    test_acc_calculator = ReasongingTypeAccCalculator(reasoning_types=all_reasoning_types)

    reload_step = 0
    if args.reload_model_path != '':
        print('reloading model from', args.reload_model_path)
        reload_step = reload(model=lstm, optimizer=optimizer, path=args.reload_model_path)

    test_dataset = LEMMA(args.test_data_file_path.format(args.base_data_dir), args.img_size, 'test', args.num_frames_per_video, args.use_preprocessed_features,
                        all_qa_interval_path='{}/vid_intervals.json'.format(args.base_data_dir), feature_base_path=args.feature_base_path)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True,drop_last=True, collate_fn=collate_func)

    test_loss, test_acc = validate(lstm=lstm, val_loader=test_dataloader, epoch=0, args=args, acc_calculator=test_acc_calculator)
    acc_dct = test_acc_calculator.get_acc()
    for key, value in acc_dct.items():
        print(f"{key} acc:{value}")
    print('test acc:', test_acc)


def validate(lstm, val_loader, epoch, args, acc_calculator):
    lstm.eval()
    all_acc = 0
    all_loss = 0
    batch_size = args.batch_size
    acc_calculator.reset()
    # starttime = time.time()
    print('validating ... ')
    with torch.no_grad():
        starttime = time.time()
        for i, (frame_rgbs, question_encode, answer_encode, frame_features, _, question, reasoning_type_lst) in enumerate(val_loader):
            
            B, num_frame_per_video, C, H, W = frame_rgbs.shape
            frame_rgbs, question_encode, answer_encode = frame_rgbs.to(args.device), question_encode.to(args.device), answer_encode.to(args.device)
            if args.use_preprocessed_features:
                frame_features = frame_features.to(device)
            else:
                frame_features = torch.ones(B, num_frame_per_video, 2048)
                # frame_features = cnn(frame_rgbs.reshape(-1, C, H, W))
                # frame_features = frame_features.reshape(B, num_frame_per_video, -1)
            
            logits = lstm(question_encode)

            all_loss += nn.CrossEntropyLoss().to(device)(logits, answer_encode.long())
            # print('validate finish in', (time.time() - starttime) * (len(val_loader) - i), 's')
            # starttime = time.time()
            pred = torch.argmax(logits, dim=1)
            test_acc = sum(pred == answer_encode) / B
            all_acc += test_acc

            acc_calculator.update(reasoning_type_lst, pred, answer_encode)

    print('validate cost:', time.time() - starttime, 's')
    all_loss /= len(val_loader)
    all_acc /= len(val_loader)
    lstm.train()
    return all_loss, all_acc


def reload(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['pure_lstm_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    global_step = checkpoint['global_step']
    return global_step

if __name__ =='__main__':
    args = parse_args()

    device_id = 0
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    args.device = device

    if args.test_only:
        print('test only!')
        print('loading model from', args.reload_model_path)
        test(args)
    else:
        print('start training...')
        train(args)