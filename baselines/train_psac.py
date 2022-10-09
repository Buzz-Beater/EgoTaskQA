import argparse
from pyexpat import model
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import os, pickle, time
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from model.PSAC.models import FrameQA_model
from dataset.dataset import LEMMA, collate_func
from utils.utils import ReasongingTypeAccCalculator

def build_resnet(model_name, pretrained=False):
    cnn = getattr(torchvision.models, model_name)(pretrained=pretrained)
    model = torch.nn.Sequential(*list(cnn.children())[:-1])
    return model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--task', type=str, default='FrameQA',help='FrameQA, Count, Action, Trans')
    parser.add_argument('--num_hid', type=int, default=512)
    parser.add_argument('--model', type=str, default='temporalAtt', help='temporalAtt')
    parser.add_argument('--max_len',type=int, default=50) # # Lq = 50, defined in code_file/models/language_model.py, = args.max_len
    parser.add_argument('--char_max_len', type=int, default=17)
    parser.add_argument('--num_frame', type=int, default=20)
    parser.add_argument('--output', type=str, default='saved_models/%s/exp-11')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=1000, help='random seed')
    # parser.add_argument('--sentense_file_path',type=str, default='./data/dataset')
    # parser.add_argument('--glove_file_path', type=str, default='/home/leiting/scratch/hcrn-videoqa/data/glove/glove.840B.300d.txt')
    parser.add_argument('--feat_category',type=str,default='resnet')
    # parser.add_argument('--feat_path',type=str,default='/mnt/data2/lixiangpeng/dataset/tgif/features')
    # parser.add_argument('--Multi_Choice',type=int, default=5)
    parser.add_argument('--vid_enc_layers', type=int, default=1)
    # parser.add_argument('--test_phase', type=bool, default=False)

    # #
    parser.add_argument("--basedir", type=str, default='data/',
                        help='where to store ckpts and logs')
    parser.add_argument("--name", type=str, default='psac_logs',
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

    parser.add_argument("--nepoch", type=int, default=70,  
                        help='num of total epoches')
    parser.add_argument("--lr", type=float, default=1e-3,  
                        help='')
    
    parser.add_argument("--i_val",   type=int, default=20000, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_test",   type=int, default=4000, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_print",   type=int, default=6, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_weight", type=int, default=4000, 
                        help='frequency of weight ckpt saving')

    parser.add_argument('--question_pt_path', type=str, default='{}/glove.pt')
    parser.add_argument('--ntoken_c', type=int, default=40, help='num of chars')
    parser.add_argument('--c_emb_dim', type=int, default=64, help='dim of char_embedding')

    parser.add_argument('--img_size', default=(224, 224))
    parser.add_argument('--num_frames_per_video', type=int, default=20)
    parser.add_argument('--cnn_modelname', type=str, default='resnet101')
    parser.add_argument('--cnn_pretrained', type=bool, default=True)

    parser.add_argument('--test_only', default=0, type=int)
    parser.add_argument('--reload_model_path', default='', type=str, help='model_path')
    parser.add_argument('--use_preprocessed_features', type=int, default=1)
    parser.add_argument('--feature_base_path', type=str, default='/scratch/generalvision/LEMMA/video_features')

    parser.add_argument('--base_data_dir', type=str, default='data')
    
    args = parser.parse_args()
    return args


def train(args):
    torch.backends.cudnn.enabled = False
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    print('parameters:', args)
    print('task:',args.task,'model:', args.model)
    device_id = 0
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    args.device = device
    
    batch_size = args.batch_size

    with open(args.question_pt_path.format(args.base_data_dir), 'rb') as f:
        obj = pickle.load(f)
        glove_matrix = obj['glove']

    word_mat = torch.from_numpy(glove_matrix)
    char_mat = torch.from_numpy(np.random.normal(loc=0.0, scale=1, size=(args.ntoken_c, args.c_emb_dim)))

    with open(args.answer_set_path.format(args.base_data_dir), 'r') as ansf:
        answers = ansf.readlines()
        num_ans_candidates = len(answers) # # output_dim == len(answers)


    # word_mat = torch.rand(13, 300) # # defined 300
    # char_mat = torch.rand(40, 64) # # defined 64
    # num_ans_candidates = 2
    cnn = build_resnet(args.cnn_modelname, pretrained=args.cnn_pretrained).to(device=args.device)
    cnn.eval() # TODO ?

    my_model = FrameQA_model.build_my_model(args.task, args.vid_enc_layers, num_ans_candidates=num_ans_candidates,
                             num_hid=args.num_hid, word_mat=word_mat, char_mat=char_mat).to(device)

    
    train_dataset = LEMMA(args.train_data_file_path.format(args.base_data_dir), args.img_size, 'train', args.num_frames_per_video, args.use_preprocessed_features,
                         all_qa_interval_path='{}/vid_intervals.json'.format(args.base_data_dir), feature_base_path=args.feature_base_path )
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,drop_last=True, collate_fn=collate_func, pin_memory=True)
    
    val_dataset = LEMMA(args.val_data_file_path.format(args.base_data_dir), args.img_size, 'val', args.num_frames_per_video, args.use_preprocessed_features, 
                        all_qa_interval_path='{}/vid_intervals.json'.format(args.base_data_dir), feature_base_path=args.feature_base_path)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True,drop_last=True, collate_fn=collate_func)

    test_dataset = LEMMA(args.test_data_file_path.format(args.base_data_dir), args.img_size, 'test', args.num_frames_per_video, args.use_preprocessed_features,
                        all_qa_interval_path='{}/vid_intervals.json'.format(args.base_data_dir), feature_base_path=args.feature_base_path)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True,drop_last=True, collate_fn=collate_func)

    criterion = nn.CrossEntropyLoss().to(device)
    # optimizer = torch.optim.Adam(my_model.parameters(), lr=args.lr)
    optimizer = torch.optim.Adamax(my_model.parameters())  

    reload_step = 0
    if args.reload_model_path != '':
        print('reloading model from', args.reload_model_path)
        reload_step = reload(cnn, model=my_model, optimizer=optimizer, path=args.reload_model_path)
    
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

    print('========start train========')
    # torch.autograd.set_detect_anomaly(True)
    for epoch in range(args.nepoch):
        my_model.train()
        train_acc_calculator.reset()
        for i, (frame_rgbs, question_encode, answer_encode, frame_features, question_char_encode, question, reasoning_type_lst) in enumerate(train_dataloader):
            B, num_frame_per_video, C, H, W = frame_rgbs.shape
            frame_rgbs, question_encode, answer_encode, question_char_encode = frame_rgbs.to(device), question_encode.to(device), answer_encode.to(device), question_char_encode.to(device)
            if args.use_preprocessed_features:
                frame_features = frame_features.to(device)
            else:
                frame_features = cnn(frame_rgbs.reshape(-1, C, H, W))
                frame_features = frame_features.reshape(B, num_frame_per_video, -1) # # B x 36 x 2048
            
            # B = 4
            # sentence_len = 25 # # Lq = 25, defined in code_file/models/language_model.py, = args.max_len
            # word_len = 17
            # num_of_sampled_frames = 36 # # Lc = 36, defined in code_file/models/language_model.py
            # v, q_w, q_c = torch.rand(B, num_of_sampled_frames, 2048).cuda(), torch.ones(B, sentence_len).long().cuda(), torch.ones(B, sentence_len, word_len).long().cuda()
            # a = torch.rand(1).cuda()
            # output = my_model(v, q_w, q_c, a)
            # print(output.shape)

            padded_question_encode = torch.zeros(B, args.max_len).long().cuda()
            padded_question_encode[:, :question_encode.shape[1]] = question_encode.clone()

            logits = my_model(frame_features, padded_question_encode, question_char_encode, answer_encode)

            answer_encode = answer_encode.long()

            loss = criterion(logits, answer_encode)
            loss.backward()
            nn.utils.clip_grad_norm_(my_model.parameters(), 0.25)
            optimizer.step()
            optimizer.zero_grad()

            pred = logits.argmax(dim=1)
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
                val_loss, val_acc = validate(cnn, my_model, val_dataloader, epoch, args, acc_calculator=test_acc_calculator)
                writer.add_scalar('val/loss', val_loss.item(), global_step)
                writer.add_scalar('val/acc', val_acc, global_step)
                acc_dct = test_acc_calculator.get_acc()
                for key, value in acc_dct.items():
                    writer.add_scalar(f'val/reasoning_{key}', value, global_step)
                log_file.write(f'[VAL]: epoch: {epoch}, global_step: {global_step}\n')
                log_file.write(f'true count dct: {test_acc_calculator.true_count_dct}\nall count dct: {test_acc_calculator.all_count_dct}\n\n')

            
            if (global_step) % args.i_test == 0:
                test_acc_calculator.reset()
                test_loss, test_acc = validate(cnn, my_model, test_dataloader, epoch, args, acc_calculator=test_acc_calculator)
                writer.add_scalar('test/loss', test_loss.item(), global_step)
                writer.add_scalar('test/acc', test_acc, global_step)
                acc_dct = test_acc_calculator.get_acc()
                for key, value in acc_dct.items():
                    writer.add_scalar(f'test/reasoning_{key}', value, global_step)
                log_file.write(f'[TEST]: epoch: {epoch}, global_step: {global_step}\n')
                log_file.write(f'true count dct: {test_acc_calculator.true_count_dct}\nall count dct: {test_acc_calculator.all_count_dct}\n\n')


            if (global_step) % args.i_weight == 0 and global_step >= 32000:
                torch.save({
                    'cnn_state_dict': cnn.state_dict(),
                    'psac_state_dict': my_model.state_dict(),
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

def validate(cnn, psac, val_loader, epoch, args, acc_calculator):
    psac.eval()
    all_acc = 0
    all_loss = 0
    batch_size = args.batch_size
    acc_calculator.reset()
    
    starttime = time.time()
    with torch.no_grad():
        for i, (frame_rgbs, question_encode, answer_encode, frame_features, question_char_encode, question, reasoning_type_lst) in enumerate(val_loader):
            B, num_frame_per_video, C, H, W = frame_rgbs.shape
            frame_rgbs, question_encode, answer_encode, question_char_encode = frame_rgbs.to(device), question_encode.to(device), answer_encode.to(device), question_char_encode.to(device)
            if args.use_preprocessed_features:
                frame_features = frame_features.to(device)
            else:
                frame_features = cnn(frame_rgbs.reshape(-1, C, H, W))
                frame_features = frame_features.reshape(B, num_frame_per_video, -1) # # B x 36 x 2048
            
            padded_question_encode = torch.zeros(B, args.max_len).long().cuda()
            padded_question_encode[:, :question_encode.shape[1]] = question_encode.clone()

            logits = psac(frame_features, padded_question_encode, question_char_encode, answer_encode)

            answer_encode = answer_encode.long()

            loss = nn.CrossEntropyLoss().to(device)(logits, answer_encode)
            
            pred = logits.argmax(dim=1)
            train_acc = sum(pred == answer_encode) / B

            all_loss += loss
            all_acc += train_acc
            # print('validate finish in', (time.time() - starttime) * (len(val_loader) - i), 's')
            # starttime = time.time()
            acc_calculator.update(reasoning_type_lst, pred, answer_encode)
    print('validate cost', time.time() - starttime, 's')
    psac.train()
    return all_loss / len(val_loader), all_acc / len(val_loader)

def test(args):
    torch.backends.cudnn.enabled = False
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    print('parameters:', args)
    print('task:',args.task,'model:', args.model)
    device_id = 0
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    args.device = device
    
    batch_size = args.batch_size

    with open(args.question_pt_path.format(args.base_data_dir), 'rb') as f:
        obj = pickle.load(f)
        glove_matrix = obj['glove']

    word_mat = torch.from_numpy(glove_matrix)
    char_mat = torch.from_numpy(np.random.normal(loc=0.0, scale=1, size=(args.ntoken_c, args.c_emb_dim)))

    with open(args.answer_set_path.format(args.base_data_dir), 'r') as ansf:
        answers = ansf.readlines()
        num_ans_candidates = len(answers) # # output_dim == len(answers)


    # word_mat = torch.rand(13, 300) # # defined 300
    # char_mat = torch.rand(40, 64) # # defined 64
    # num_ans_candidates = 2
    cnn = build_resnet(args.cnn_modelname, pretrained=args.cnn_pretrained).to(device=args.device)
    cnn.eval() # TODO ?

    my_model = FrameQA_model.build_my_model(args.task, args.vid_enc_layers, num_ans_candidates=num_ans_candidates,
                             num_hid=args.num_hid, word_mat=word_mat, char_mat=char_mat).to(device)

    test_dataset = LEMMA(args.test_data_file_path.format(args.base_data_dir), args.img_size, 'test', args.num_frames_per_video, args.use_preprocessed_features,
                        all_qa_interval_path='{}/vid_intervals.json'.format(args.base_data_dir), feature_base_path=args.feature_base_path)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True,drop_last=True, collate_fn=collate_func)

    criterion = nn.CrossEntropyLoss().to(device)
    # optimizer = torch.optim.Adam(my_model.parameters(), lr=args.lr)
    optimizer = torch.optim.Adamax(my_model.parameters())  

    reload_step = 0
    if args.reload_model_path != '':
        print('reloading model from', args.reload_model_path)
        reload_step = reload(cnn, model=my_model, optimizer=optimizer, path=args.reload_model_path)
    
    with open('{}/all_reasoning_types.txt'.format(args.base_data_dir), 'r') as reasonf:
        all_reasoning_types = reasonf.readlines()
        all_reasoning_types = [item.strip() for item in all_reasoning_types]
    test_acc_calculator = ReasongingTypeAccCalculator(reasoning_types=all_reasoning_types)

    testloss, testacc = validate(cnn=cnn, psac=my_model, val_loader=test_dataloader, epoch=0, args=args, acc_calculator=test_acc_calculator)
    acc_dct = test_acc_calculator.get_acc()
    for key, value in acc_dct.items():
        print(f"{key} acc:{value}")
    print('test acc:', testacc)
    
def reload(cnn, model, optimizer, path):
    checkpoint = torch.load(path)
    cnn.load_state_dict(checkpoint['cnn_state_dict'])
    model.load_state_dict(checkpoint['psac_state_dict'])
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