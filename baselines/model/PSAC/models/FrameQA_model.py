import torch.nn as nn
from model.PSAC.models.language_model import *
from model.PSAC.models.classifier import SimpleClassifier
from model.PSAC.models.fc import FCNet
import torch
from torch.autograd import Variable
from model.PSAC.models.attention import NewAttention
from model.PSAC.models.attention import *
from torch.nn.utils.weight_norm import weight_norm
import time
import numpy as np
from model.PSAC.models.model_utils import *
import torch.nn.functional as F


class FrameQAModel(nn.Module):
    def __init__(self, model_name, vid_encoder, ques_encoder,classifier,
                 n_layer=6, n_head=8, d_k=64, d_v=64, v_len=35, v_emb_dim=300,
                 d_model=512, d_inner_hid=512, dropout=0.1, conv_num=4, num_choice=1):
        super(FrameQAModel, self).__init__()

        self.model_name= model_name
        self.num_choice = num_choice
        self.vid_encoder = vid_encoder
        self.ques_encoder = ques_encoder
        self.classifier = classifier


    def forward(self, v, q_w, q_c, labels, return_attns=False):
        
        # visual info
        vid_output = self.vid_encoder(v)  # v : batch_size x v_len x ctx_dim  # batch_size x v_len x d_v
        vid_output = v + vid_output
        # vid_output = v
        fus_output = self.ques_encoder(vid_output, q_w, q_c)
        
        logits = self.classifier(fus_output)
        out = F.log_softmax(logits, dim=1)
        return out

    def evaluate(self, dataloader):
        score = 0
        num_data = 0
        for v, q_w, q_c, a in iter(dataloader):
            v = Variable(v).cuda()
            q_w = Variable(q_w).cuda()
            q_c = Variable(q_c).cuda()
            pred = self.forward(v, q_w, q_c, None)
            batch_score = compute_score_with_logits(pred, a.cuda())
            score += batch_score
            num_data += pred.size(0)

        score = float(score) / len(dataloader.dataset)
        return score

    def sample(self, dataset, dataloader):
        import json
        score = 0
        num_data = 0
        j = 0
        results = []
        for v, q_w, q_c, a, ques_eng, ans_eng, idx in iter(dataloader):
            v = Variable(v).cuda()
            q_w = Variable(q_w).cuda()
            q_c = Variable(q_c).cuda()
            pred = self.forward(v, q_w, q_c, None)
            # pred_label = pred.data.cpu().max(1)[1]
            batch_score = compute_score_with_logits(pred, a.cuda())
            prediction = torch.max(pred, 1)[1]
            score += batch_score
            num_data += pred.size(0)

            # write in json
            prediction = np.array(prediction.cpu().data)
            for ques, pred, ans, idx in zip(ques_eng, list(prediction), ans_eng, idx):
                ins = {}
                ins['index'] = j
                # ins['gif_name'] = gif_name
                # ins['key'] = key.data
                ins['id'] = int(idx)
                ins['question'] = ques
                ins['prediction'] = dataset.label2ans[pred]
                ins['answer'] = ans
                ins['prediction_success'] = bool(ans == dataset.label2ans[pred])
                results.append(ins)
                j += 1
        with open('data/prediction/FrameQA_prediction.json', 'w') as f:
            json.dump(results, f)
        score = float(score) / len(dataloader.dataset)
        return score

def build_temporalAtt(task_name, n_layer, dataset, num_hid, dictionary, glove_file):
    # w_emb = WordEmbedding(dataset.dictionary.ntoken, len(dataset.dictionary.idx2char), 300, 64, 0.0)
    vid_encoder = Encoder(n_layer=n_layer, n_head=8, d_k=256, d_v=256, v_len=36, v_emb_dim=300,
                               d_model=2048, d_inner_hid=512, dropout=0.1)
    w = WordEmbedding(dictionary.ntoken, dictionary.c_ntoken, 300, 64, 0.1)
    word_mat, char_mat = w.init_embedding(dictionary, glove_file, task_name)
    
    ques_encoder = Ques_Encoder(word_mat, char_mat)
    classifier = SimpleClassifier(
        num_hid, num_hid * 2, dataset.num_ans_candidates, 0.5)
    # classifier = weight_norm(nn.Linear(num_hid, dataset.num_ans_candidates), dim=None)
    return FrameQAModel(task_name, vid_encoder, ques_encoder, classifier)


def build_my_model(task_name, n_layer, num_ans_candidates, num_hid, word_mat, char_mat, glove_file=None):
    # w_emb = WordEmbedding(dataset.dictionary.ntoken, len(dataset.dictionary.idx2char), 300, 64, 0.0)
    vid_encoder = Encoder(n_layer=n_layer, n_head=8, d_k=256, d_v=256, v_len=36, v_emb_dim=300,
                               d_model=2048, d_inner_hid=512, dropout=0.1)
    # w = WordEmbedding(dictionary.ntoken, dictionary.c_ntoken, 300, 64, 0.1)
    # word_mat, char_mat = w.init_embedding(dictionary, glove_file, task_name)
    
    ques_encoder = Ques_Encoder(word_mat, char_mat)
    classifier = SimpleClassifier(
        num_hid, num_hid * 2, num_ans_candidates, 0.5)
    # classifier = weight_norm(nn.Linear(num_hid, dataset.num_ans_candidates), dim=None)
    return FrameQAModel(task_name, vid_encoder, ques_encoder, classifier)


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1]
    pred_y = logits.data.cpu().numpy().squeeze()
    target_y = labels.cpu().numpy().squeeze()
    scores = sum(pred_y==target_y)
    return scores
