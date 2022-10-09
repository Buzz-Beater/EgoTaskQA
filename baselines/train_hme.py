from pyexpat import model
from statistics import mode
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

from dataset.hme_dataset import LEMMA, collate_func
from model.HME.attention_module_lite import *
from utils.utils import ReasongingTypeAccCalculator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--basedir", type=str, default='data/',
                        help='where to store ckpts and logs')
    parser.add_argument("--name", type=str, default='hme_logs',
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

    parser.add_argument("--batch_size", type=int, default=32, )
    parser.add_argument("--nepoch", type=int, default=33,  
                        help='num of total epoches')
    parser.add_argument("--lr", type=float, default=1e-3,  
                        help='')
    
    parser.add_argument("--i_val",   type=int, default=10000, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_test",   type=int, default=4000, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_print",   type=int, default=6, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_weight", type=int, default=4000, 
                        help='frequency of weight ckpt saving')

    parser.add_argument('--img_size', default=(224, 224))
    parser.add_argument('--num_frames_per_video', type=int, default=20)
    parser.add_argument('--cnn_modelname', type=str, default='resnet101')
    parser.add_argument('--cnn_pretrained', type=bool, default=True)
    parser.add_argument('--output_dim', type=int, default=1)
    parser.add_argument('--video_feature_path', type=str, default='data/video_feature_20.h5')

    parser.add_argument('--test_only', default=0, type=int)
    parser.add_argument('--reload_model_path', default='', type=str, help='model_path')
    parser.add_argument('--question_pt_path', type=str, default='{}/glove.pt')
   
    # parser.add_argument('--max_sequence_length', default=20, type=int)
    parser.add_argument('--base_data_dir', type=str, default='data')
    args = parser.parse_args()
    return args

def train(args):
    device = args.device

    train_dataset = LEMMA(args.train_data_file_path.format(args.base_data_dir), args.img_size, 'train', args.num_frames_per_video, args.video_feature_path)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_func)
    
    val_dataset = LEMMA(args.val_data_file_path.format(args.base_data_dir), args.img_size, 'val', args.num_frames_per_video, args.video_feature_path)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True, collate_fn=collate_func)

    test_dataset = LEMMA(args.test_data_file_path.format(args.base_data_dir), args.img_size, 'test', args.num_frames_per_video, args.video_feature_path)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True, collate_fn=collate_func)
    
    with open(args.answer_set_path.format(args.base_data_dir), 'r') as ansf:
        answers = ansf.readlines()
        answer_vocab_size = len(answers) # # output_dim == len(answers)

    with open(args.question_pt_path.format(args.base_data_dir), 'rb') as f:
        obj = pickle.load(f)
        glove_matrix = obj['glove']

    feat_channel = 2048
    feat_dim = 1
    text_embed_size = 300
    hidden_size = 512

    word_matrix = glove_matrix
    voc_len = word_matrix.shape[0]
    num_layers= 2
    max_sequence_length = args.num_frames_per_video # #  args.video_feature_num = 20, fixed

    my_rnn = AttentionTwoStream(feat_channel, feat_dim, text_embed_size, hidden_size,
                         voc_len, num_layers, word_matrix, answer_vocab_size = answer_vocab_size,
                         max_len=max_sequence_length).to(device)

    criterion = nn.CrossEntropyLoss(size_average=True).to(device)
    optimizer = optim.Adam(my_rnn.parameters(), lr=args.lr)

    reload_step = 0
    if args.reload_model_path != '':
        print('reloading model from', args.reload_model_path)
        reload_step = reload(model=my_rnn, optimizer=optimizer, path=args.reload_model_path)
    
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
        my_rnn.train()
        train_acc_calculator.reset()
        for i, (question_encode, answer_encode, vgg, c3d, question_length_lst, reasoning_type_lst) in enumerate(train_dataloader):
            B, q_len = question_encode.shape
            question_encode, answer_encode, vgg, c3d = question_encode.to(device), answer_encode.to(device), vgg.to(device), c3d.to(device)
            
            video_features = torch.cat([c3d,vgg],dim=2)
            video_features = video_features.view(video_features.size(0),video_features.size(1),1,1,video_features.size(2))
            video_lengths = [max_sequence_length] * B # # fixed

            data_dict = {'video_features': video_features,
                'video_lengths': video_lengths,
                'question_words': question_encode,
                'answers': answer_encode,
                'question_lengths': question_length_lst}

            outputs, predictions = my_rnn(data_dict)

            targets = data_dict['answers']

            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc = my_rnn.accuracy(predictions, targets)
            
            train_acc_calculator.update(reasoning_type_lst, predictions, targets)
            
            writer.add_scalar('train/loss', loss.item(), global_step)
            writer.add_scalar('learning rates', optimizer.param_groups[0]['lr'], global_step)
            writer.add_scalar('train/acc', acc, global_step)

            if global_step % args.i_print  == 0:
                print(f"global_step:{global_step}, train_loss:{loss.item()}, train_acc:{acc}")

            if (global_step) % args.i_val == 0 and global_step >= 400:
                test_acc_calculator.reset()
                val_loss, val_acc = validate(my_rnn, val_dataloader, epoch, args, acc_calculator=test_acc_calculator)
                writer.add_scalar('val/loss', val_loss.item(), global_step)
                writer.add_scalar('val/acc', val_acc, global_step)
                acc_dct = test_acc_calculator.get_acc()
                for key, value in acc_dct.items():
                    writer.add_scalar(f'val/reasoning_{key}', value, global_step)
                log_file.write(f'[VAL]: epoch: {epoch}, global_step: {global_step}\n')
                log_file.write(f'true count dct: {test_acc_calculator.true_count_dct}\nall count dct: {test_acc_calculator.all_count_dct}\n\n')

            if (global_step) % args.i_test == 0 and global_step >= 4000:
                test_acc_calculator.reset()
                test_loss, test_acc = validate(my_rnn, test_dataloader, epoch, args, acc_calculator=test_acc_calculator)
                writer.add_scalar('test/loss', test_loss.item(), global_step)
                writer.add_scalar('test/acc', test_acc, global_step)
                acc_dct = test_acc_calculator.get_acc()
                for key, value in acc_dct.items():
                    writer.add_scalar(f'test/reasoning_{key}', value, global_step)
                log_file.write(f'[TEST]: epoch: {epoch}, global_step: {global_step}\n')
                log_file.write(f'true count dct: {test_acc_calculator.true_count_dct}\nall count dct: {test_acc_calculator.all_count_dct}\n\n')


            if (global_step) % args.i_weight == 0 and global_step >= 8000:
                torch.save({
                    'my_rnn_state_dict': my_rnn.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    'global_step': global_step,
                }, os.path.join(args.basedir, 'ckpts', f"model_{global_step}.tar"))
            pbar.update(1)
            global_step += 1

        acc_dct = train_acc_calculator.get_acc()
        for key, value in acc_dct.items():
            writer.add_scalar(f'train/reasoning_{key}', value, global_step)
        log_file.write(f'[TRAIN]: epoch: {epoch}, global_step: {global_step}\n')
        log_file.write(f'true count dct: {train_acc_calculator.true_count_dct}\nall count dct: {train_acc_calculator.all_count_dct}\n\n')
        log_file.flush()


def test(args):
    device = args.device

    test_dataset = LEMMA(args.test_data_file_path.format(args.base_data_dir), args.img_size, 'test', args.num_frames_per_video, args.video_feature_path)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True, collate_fn=collate_func, drop_last=True)
    
    with open(args.answer_set_path.format(args.base_data_dir), 'r') as ansf:
        answers = ansf.readlines()
        answer_vocab_size = len(answers) # # output_dim == len(answers)

    with open(args.question_pt_path.format(args.base_data_dir), 'rb') as f:
        obj = pickle.load(f)
        glove_matrix = obj['glove']

    feat_channel = 2048
    feat_dim = 1
    text_embed_size = 300
    hidden_size = 512

    word_matrix = glove_matrix
    voc_len = word_matrix.shape[0]
    num_layers= 2
    max_sequence_length = args.num_frames_per_video # #  args.video_feature_num = 20, fixed

    my_rnn = AttentionTwoStream(feat_channel, feat_dim, text_embed_size, hidden_size,
                         voc_len, num_layers, word_matrix, answer_vocab_size = answer_vocab_size,
                         max_len=max_sequence_length).to(device)

    criterion = nn.CrossEntropyLoss(size_average=True).to(device)
    optimizer = optim.Adam(my_rnn.parameters(), lr=args.lr)

    reload_step = 0
    if args.reload_model_path != '':
        print('reloading model from', args.reload_model_path)
        reload_step = reload(model=my_rnn, optimizer=optimizer, path=args.reload_model_path)
    
    with open('{}/all_reasoning_types.txt'.format(args.base_data_dir), 'r') as reasonf:
        all_reasoning_types = reasonf.readlines()
        all_reasoning_types = [item.strip() for item in all_reasoning_types]

    test_acc_calculator = ReasongingTypeAccCalculator(reasoning_types=all_reasoning_types)
    testloss, testacc = validate(my_rnn=my_rnn, val_loader=test_dataloader, epoch=0, args=args, acc_calculator=test_acc_calculator)
    acc_dct = test_acc_calculator.get_acc()
    for key, value in acc_dct.items():
        print(f"{key} acc:{value}")
    print('test acc', testacc)


def validate(my_rnn, val_loader, epoch, args, acc_calculator):
    my_rnn.eval()
    all_acc = 0
    all_loss = 0
    batch_size = args.batch_size
    acc_calculator.reset()
    starttime = time.time()
    print('validating...')
    with torch.no_grad():
        for i, (question_encode, answer_encode, vgg, c3d, question_length_lst, reasoning_type_lst) in enumerate(tqdm(val_loader)):
            B, q_len = question_encode.shape
            question_encode, answer_encode, vgg, c3d = question_encode.to(device), answer_encode.to(device), vgg.to(device), c3d.to(device)
            
            video_features = torch.cat([c3d,vgg],dim=2)
            video_features = video_features.view(video_features.size(0),video_features.size(1),1,1,video_features.size(2))
            video_lengths = [args.num_frames_per_video] * B # # fixed

            data_dict = {'video_features': video_features,
                'video_lengths': video_lengths,
                'question_words': question_encode,
                'answers': answer_encode,
                'question_lengths': question_length_lst}

            outputs, predictions = my_rnn(data_dict)

            targets = data_dict['answers']

            loss = nn.CrossEntropyLoss(size_average=True).to(device)(outputs, targets)
            
            acc = my_rnn.accuracy(predictions, targets)
            all_loss += loss
            all_acc += acc

            acc_calculator.update(reasoning_type_lst, predictions, targets)

    all_loss /= len(val_loader)
    all_acc /= len(val_loader)
    my_rnn.train()
    return all_loss, all_acc


def reload(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['my_rnn_state_dict'])
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