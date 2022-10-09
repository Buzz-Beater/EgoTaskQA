import argparse
from regex import B
import torch.nn.functional as F
import random
import h5py
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

from yaml import parse

from model.HGA.warmup_scheduler import GradualWarmupScheduler
seed = 999

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True


def _init_fn(worker_id):
    np.random.seed(seed)


from model.HGA.models.lstm_cross_cycle_gcn_dropout import LSTMCrossCycleGCNDropout
from dataset.hme_dataset import LEMMA, collate_func
from utils.utils import ReasongingTypeAccCalculator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--basedir", type=str, default='data/',
                        help='where to store ckpts and logs')
    parser.add_argument("--name", type=str, default='hga_logs',
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

    parser.add_argument("--nepoch", type=int, default=150,  
                        help='num of total epoches')
    
    parser.add_argument("--i_val",   type=int, default=10000, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_test",   type=int, default=3000, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_print",   type=int, default=1000,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_weight", type=int, default=3000, 
                        help='frequency of weight ckpt saving')

    parser.add_argument('--img_size', default=(224, 224))
    parser.add_argument('--num_frames_per_video', type=int, default=20)

    parser.add_argument('--video_feature_path', type=str, default='data/video_feature_20.h5')

    parser.add_argument('--test_only', default=0, type=int)
    parser.add_argument('--reload_model_path', default='', type=str, help='model_path')
    parser.add_argument('--question_pt_path', type=str, default='{}/glove.pt')
   
    # parser.add_argument('--max_sequence_length', default=20, type=int)
    # # from HGA/main.py parser
    parser.add_argument(
        '--rnn_layers', type=int, default=1, help='number of layers in lstm')
    parser.add_argument(
        '--birnn', type=int, default=0, help='bidirectional rnn or not')
    parser.add_argument(
        '--gcn_layers',
        type=int,
        default=2,
        help='number of layers in gcn (+1)')
    parser.add_argument(
        '--tf_layers',
        type=int,
        default=1,
        help='number of layers in transformer')
    parser.add_argument("--batch_size", type=int, default=64, )
    parser.add_argument('--max_n_videos', type=int, default=100000)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lr_list', type=list, default=[10, 20, 30, 40])
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--cycle_beta', type=float, default=0.01)
    parser.add_argument('--two_loss', type=int, default=0)
    parser.add_argument(
        '--change_lr', type=str, default='none', help='0 False, 1 True')
    parser.add_argument(
        '--weight_decay', type=float, default=0, help='weight_decay')
    parser.add_argument('--fusion_type', type=str, default='coattn')
    parser.add_argument('--ablation', type=str, default='none')

    parser.add_argument(
        '--hidden_size',
        type=int,
        default=512,
        help='dimension of lstm hidden states')
    parser.add_argument(
        '--prefetch',
        type=str,
        default='none',
        help='prefetch function [nvidia, background]')
    parser.add_argument(
        '--task',
        type=str,
        default='FrameQA',
        help='[Count, Action, FrameQA, Trans]')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--q_max_length', type=int, default=35, help='limit of question pos_embedding')
    parser.add_argument('--v_max_length', type=int, default=80, help='limit of video pos_embedding')

    parser.add_argument('--base_data_dir', type=str, default='data')
    args = parser.parse_args()
    return args

def test(args):
    device = args.device

    test_dataset = LEMMA(args.test_data_file_path.format(args.base_data_dir), args.img_size, 'test', args.num_frames_per_video, args.video_feature_path)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, drop_last=True, shuffle=True, collate_fn=collate_func)
    
    with open(args.answer_set_path.format(args.base_data_dir), 'r') as ansf:
        answers = ansf.readlines()
        answer_vocab_size = len(answers) # # output_dim == len(answers)

    with open(args.question_pt_path.format(args.base_data_dir), 'rb') as f:
        obj = pickle.load(f)
        glove_matrix = obj['glove']

    if args.two_loss > 0:
        args.two_loss = True
    else:
        args.two_loss = False

    if args.birnn > 0:
        args.birnn = True
    else:
        args.birnn = False

    assert args.ablation in ['none', 'gcn', 'global', 'local', 'only_local']
    assert args.fusion_type in [
        'none', 'coattn', 'single_visual', 'single_semantic', 'coconcat',
        'cosiamese'
    ]
    # # mymodel
    args.word_matrix = glove_matrix
    args.voc_len = args.word_matrix.shape[0]
    args.resnet_input_size = 4096
    args.c3d_input_size = 4096
    args.answer_vocab_size = answer_vocab_size

    model = LSTMCrossCycleGCNDropout(
        args.voc_len,
        args.rnn_layers,
        args.birnn,
        'gru',
        args.word_matrix,
        args.resnet_input_size,
        args.c3d_input_size,
        args.rnn_layers,
        args.birnn,
        'gru',
        args.hidden_size,
        dropout_p=args.dropout,
        gcn_layers=args.gcn_layers,
        num_heads=8,
        answer_vocab_size=args.answer_vocab_size,
        q_max_len=args.q_max_length,
        v_max_len=args.v_max_length,
        tf_layers=args.tf_layers,
        two_loss=args.two_loss,
        fusion_type=args.fusion_type,
        ablation=args.ablation)
    model.to(device)

    reload_step = 0
    if args.reload_model_path != '':
        print('reloading model from', args.reload_model_path)
        model = torch.load(args.reload_model_path, )
        # reload_step = reload(model=model, optimizer=optimizer, path=args.reload_model_path)
    

    criterion = nn.CrossEntropyLoss(size_average=True).to(device)
    if args.change_lr == 'none':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.change_lr == 'acc':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr / 5., weight_decay=args.weight_decay)
        # val plateau scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.1, patience=3, verbose=True)
        # target lr = args.lr * multiplier
        scheduler_warmup = GradualWarmupScheduler(
            optimizer, multiplier=5, total_epoch=5, after_scheduler=scheduler)
    elif args.change_lr == 'loss':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr / 5., weight_decay=args.weight_decay)
        # val plateau scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=3, verbose=True)
        # target lr = args.lr * multiplier
        scheduler_warmup = GradualWarmupScheduler(
            optimizer, multiplier=5, total_epoch=5, after_scheduler=scheduler)
    elif args.change_lr == 'cos':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr / 5., weight_decay=args.weight_decay)
        # consine annealing
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.max_epoch)
        # target lr = args.lr * multiplier
        scheduler_warmup = GradualWarmupScheduler(
            optimizer, multiplier=5, total_epoch=5, after_scheduler=scheduler)
    elif args.change_lr == 'step':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args.lr_list, gamma=0.1)

    with open('{}/all_reasoning_types.txt'.format(args.base_data_dir), 'r') as reasonf:
        all_reasoning_types = reasonf.readlines()
        all_reasoning_types = [item.strip() for item in all_reasoning_types]

    test_acc_calculator = ReasongingTypeAccCalculator(reasoning_types=all_reasoning_types)

    testloss, testacc = validate(model=model, val_loader=test_dataloader, epoch=0, args=args, criterion=criterion, acc_calculator=test_acc_calculator)
    acc_dct = test_acc_calculator.get_acc()
    for key, value in acc_dct.items():
        print(f"{key} acc:{value}")
    print('test acc:', testacc)


def train(args):
    device = args.device

    train_dataset = LEMMA(args.train_data_file_path.format(args.base_data_dir), args.img_size, 'train', args.num_frames_per_video, args.video_feature_path)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, shuffle=True, collate_fn=collate_func)
    
    val_dataset = LEMMA(args.val_data_file_path.format(args.base_data_dir), args.img_size, 'val', args.num_frames_per_video, args.video_feature_path)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, drop_last=True, shuffle=True, collate_fn=collate_func)

    test_dataset = LEMMA(args.test_data_file_path.format(args.base_data_dir), args.img_size, 'test', args.num_frames_per_video, args.video_feature_path)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, drop_last=True, shuffle=True, collate_fn=collate_func)
    
    with open(args.answer_set_path.format(args.base_data_dir), 'r') as ansf:
        answers = ansf.readlines()
        answer_vocab_size = len(answers) # # output_dim == len(answers)

    with open(args.question_pt_path.format(args.base_data_dir), 'rb') as f:
        obj = pickle.load(f)
        glove_matrix = obj['glove']

    if args.two_loss > 0:
        args.two_loss = True
    else:
        args.two_loss = False

    if args.birnn > 0:
        args.birnn = True
    else:
        args.birnn = False

    assert args.ablation in ['none', 'gcn', 'global', 'local', 'only_local']
    assert args.fusion_type in [
        'none', 'coattn', 'single_visual', 'single_semantic', 'coconcat',
        'cosiamese'
    ]
    # # mymodel
    args.word_matrix = glove_matrix
    args.voc_len = args.word_matrix.shape[0]
    args.resnet_input_size = 4096
    args.c3d_input_size = 4096
    args.answer_vocab_size = answer_vocab_size

    model = LSTMCrossCycleGCNDropout(
        args.voc_len,
        args.rnn_layers,
        args.birnn,
        'gru',
        args.word_matrix,
        args.resnet_input_size,
        args.c3d_input_size,
        args.rnn_layers,
        args.birnn,
        'gru',
        args.hidden_size,
        dropout_p=args.dropout,
        gcn_layers=args.gcn_layers,
        num_heads=8,
        answer_vocab_size=args.answer_vocab_size,
        q_max_len=args.q_max_length,
        v_max_len=args.v_max_length,
        tf_layers=args.tf_layers,
        two_loss=args.two_loss,
        fusion_type=args.fusion_type,
        ablation=args.ablation)
    model.to(device)

    reload_step = 0
    if args.reload_model_path != '':
        print('reloading model from', args.reload_model_path)
        model = torch.load(args.reload_model_path, )
        # reload_step = reload(model=model, optimizer=optimizer, path=args.reload_model_path)
    

    criterion = nn.CrossEntropyLoss(size_average=True).to(device)
    if args.change_lr == 'none':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.change_lr == 'acc':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr / 5., weight_decay=args.weight_decay)
        # val plateau scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.1, patience=3, verbose=True)
        # target lr = args.lr * multiplier
        scheduler_warmup = GradualWarmupScheduler(
            optimizer, multiplier=5, total_epoch=5, after_scheduler=scheduler)
    elif args.change_lr == 'loss':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr / 5., weight_decay=args.weight_decay)
        # val plateau scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=3, verbose=True)
        # target lr = args.lr * multiplier
        scheduler_warmup = GradualWarmupScheduler(
            optimizer, multiplier=5, total_epoch=5, after_scheduler=scheduler)
    elif args.change_lr == 'cos':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr / 5., weight_decay=args.weight_decay)
        # consine annealing
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.max_epoch)
        # target lr = args.lr * multiplier
        scheduler_warmup = GradualWarmupScheduler(
            optimizer, multiplier=5, total_epoch=5, after_scheduler=scheduler)
    elif args.change_lr == 'step':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args.lr_list, gamma=0.1)

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
        print('Start Training Epoch: {}'.format(epoch))

        model.train()
        train_acc_calculator.reset()

        loss_list = []
        prediction_list = []
        correct_answer_list = []

        if args.change_lr == 'cos':
            # consine annealing
            scheduler_warmup.step(epoch=epoch)

        for i, (question_encode, answer_encode, vgg, c3d, question_length_lst, reasoning_type_lst) in enumerate(train_dataloader):
            B, q_len = question_encode.shape
            B, v_len, _ = vgg.shape
            question_encode, answer_encode, vgg, c3d = question_encode.to(device), answer_encode.to(device), vgg.to(device), c3d.to(device)
            
            video_lengths = torch.from_numpy(np.array([v_len] * B)) # # fixed
            question_lengths = torch.from_numpy(np.array(question_length_lst))
            answer_type = 'open'

            optimizer.zero_grad()
            
            out, predictions, answers, _ = model(args.task, vgg, c3d, video_lengths, question_encode, question_lengths, answer_encode, answer_type )
            loss = criterion(out, answers)
            loss.backward()
            optimizer.step()
            
            correct_answer_list.append(answers)
            loss_list.append(loss.item())
            prediction_list.append(predictions.detach())

            acc = (torch.sum(answers == predictions).cpu().numpy()) / B
            train_acc_calculator.update(reasoning_type_lst, predictions, answers)
            
            writer.add_scalar('train/loss', loss.item(), global_step)
            writer.add_scalar('learning rates', optimizer.param_groups[0]['lr'], global_step)
            writer.add_scalar('train/acc', acc, global_step)

            pbar.update(1)

            if global_step % args.i_print  == 0:
                print(f"global_step:{global_step}, train_loss:{loss.item()}, train_acc:{acc}")

            if (global_step) % args.i_val == 0:
                test_acc_calculator.reset()
                val_loss, val_acc = validate(model, val_dataloader, epoch, args, criterion, acc_calculator=test_acc_calculator)
                writer.add_scalar('val/loss', val_loss.item(), global_step)
                writer.add_scalar('val/acc', val_acc, global_step)
                acc_dct = test_acc_calculator.get_acc()
                for key, value in acc_dct.items():
                    writer.add_scalar(f'val/reasoning_{key}', value, global_step)
                log_file.write(f'[VAL]: epoch: {epoch}, global_step: {global_step}\n')
                log_file.write(f'true count dct: {test_acc_calculator.true_count_dct}\nall count dct: {test_acc_calculator.all_count_dct}\n\n')

            if (global_step) % args.i_test == 0:
                test_acc_calculator.reset()
                test_loss, test_acc = validate(model, test_dataloader, epoch, args, criterion, acc_calculator=test_acc_calculator)
                writer.add_scalar('test/loss', test_loss.item(), global_step)
                writer.add_scalar('test/acc', test_acc, global_step)
                acc_dct = test_acc_calculator.get_acc()
                for key, value in acc_dct.items():
                    writer.add_scalar(f'test/reasoning_{key}', value, global_step)
                log_file.write(f'[TEST]: epoch: {epoch}, global_step: {global_step}\n')
                log_file.write(f'true count dct: {test_acc_calculator.true_count_dct}\nall count dct: {test_acc_calculator.all_count_dct}\n\n')


            if (global_step) % args.i_weight == 0 and global_step >= 30000:
                torch.save(model, os.path.join(args.basedir, 'ckpts', f"model_{global_step}.pth"))
            
            global_step += 1
        
        acc_dct = train_acc_calculator.get_acc()
        for key, value in acc_dct.items():
            writer.add_scalar(f'train/reasoning_{key}', value, global_step)
        log_file.write(f'[TRAIN]: epoch: {epoch}, global_step: {global_step}\n')
        log_file.write(f'true count dct: {train_acc_calculator.true_count_dct}\nall count dct: {train_acc_calculator.all_count_dct}\n\n')
        log_file.flush()

        
        train_loss = np.mean(loss_list)
        correct_answer = torch.cat(correct_answer_list, dim=0).long()
        predict_answer = torch.cat(prediction_list, dim=0).long()
        assert correct_answer.shape == predict_answer.shape
        current_num = torch.sum(predict_answer == correct_answer).cpu().numpy()
        acc = current_num / len(correct_answer) * 100.

        if args.change_lr == 'acc':
            scheduler_warmup.step(epoch, val_acc)
        elif args.change_lr == 'loss':
            scheduler_warmup.step(epoch, val_loss)
        elif args.change_lr == 'step':
            scheduler.step()

        print(
            "Train|Epoch: {}, Acc : {:.3f}={}/{}, Train Loss: {:.3f}".format(
                epoch, acc, current_num, len(correct_answer), train_loss))


@torch.no_grad()
def validate(model, val_loader, epoch, args, criterion, acc_calculator):
    model.eval()
    print('validating ... ')
    
    loss_list = []
    prediction_list = []
    correct_answer_list = []
    acc_calculator.reset()

    starttime = time.time()
    for i, (question_encode, answer_encode, vgg, c3d, question_length_lst, reasoning_type_lst) in enumerate(val_loader):
        B, q_len = question_encode.shape
        B, v_len, _ = vgg.shape
        question_encode, answer_encode, vgg, c3d = question_encode.to(device), answer_encode.to(device), vgg.to(device), c3d.to(device)
        
        video_lengths = torch.from_numpy(np.array([v_len] * B)) # # fixed
        question_lengths = torch.from_numpy(np.array(question_length_lst))
        answer_type = 'open'

        out, predictions, answers, _ = model(args.task, vgg, c3d, video_lengths, question_encode, question_lengths, answer_encode, answer_type )
        loss = criterion(out, answers)
        
        correct_answer_list.append(answers)
        loss_list.append(loss.item())
        prediction_list.append(predictions.detach())
        # print('validate finish in', (time.time() - starttime) * (len(val_loader) - i), 's')
        # starttime = time.time()
        acc_calculator.update(reasoning_type_lst, predictions, answers)

    print('validate cost:', time.time() - starttime, 's')
    val_loss = np.mean(loss_list)
    correct_answer = torch.cat(correct_answer_list, dim=0).long()
    predict_answer = torch.cat(prediction_list, dim=0).long()
    assert correct_answer.shape == predict_answer.shape

    current_num = torch.sum(predict_answer == correct_answer).cpu().numpy()

    acc = current_num / len(correct_answer) * 100.
    print(
        "VAL/TEST |Epoch: {}, Acc: {:3f}={}/{}, Val Loss: {:3f}".format(
            epoch, acc, current_num, len(correct_answer), val_loss))
    
    model.train()
    return val_loss, acc


# def reload( model, optimizer, path):
#     checkpoint = torch.load(path)
#     model.load_state_dict(checkpoint['hga_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     global_step = checkpoint['global_step']
#     # model.eval()
#     return global_step



if __name__ =='__main__':
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    if args.test_only:
        print('test only!')
        print('loading model from', args.reload_model_path)
        test(args)
    else:
        print('start training...')
        train(args)