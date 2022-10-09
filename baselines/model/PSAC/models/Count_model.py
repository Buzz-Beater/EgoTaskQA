import torch.nn as nn
from language_model import *
from classifier import SimpleClassifier
from fc import FCNet
import torch
from torch.autograd import Variable
import numpy as np
from attention import NewAttention
from model_utils import *
import torch.nn.functional as F

class CountModel(nn.Module):
    def __init__(self, model_name, vid_encoder, ques_encoder, classifier,
                 n_layer=6, n_head=8, d_k=64, d_v=64, v_len=35, v_emb_dim=300,
                 d_model=512, d_inner_hid=512, dropout=0.1, conv_num=4, num_choice=1):
        super(CountModel, self).__init__()
        self.model_name = model_name
        self.num_choice = num_choice
        self.vid_encoder = vid_encoder
        self.ques_encoder = ques_encoder
        self.classifier = classifier

    def forward(self, v, q_w, q_c, labels):

        # visual info
        vid_output = self.vid_encoder(v)
        vid_output = vid_output + v
        # vid_output = v
        fus_output = self.ques_encoder(vid_output, q_w, q_c)
        logits = self.classifier(fus_output)
        logits = torch.squeeze(logits)

        return logits

    def evaluate(self, dataloader):
        score = 0

        for v, q_w, q_c, a,ques_eng, ans_eng, idx in iter(dataloader):
            v = Variable(v).cuda()
            q_w = Variable(q_w).cuda()
            q_c = Variable(q_c).cuda()
            logits = self.forward(v, q_w, q_c , None)
            logits = logits.data.cpu().numpy().squeeze()
            pred = np.clip(logits, a_min=1, a_max=10)
            pred = np.round(pred)
            batch_score = compute_score_with_logits(pred, a.cuda())
            score += batch_score

        score = float(score) / len(dataloader.dataset)
        return score

    def evaluate_and_sample(self, dataloader, dataset):
        score = 0
        j = 0
        for idxs, v, q_w, q_c, a in iter(dataloader):
            v = Variable(v).cuda()
            q_w = Variable(q_w).cuda()
            q_c = Variable(q_c).cuda()
            logits = self.forward(v, q_w, q_c , None)
            logits = logits.data.cpu().numpy().squeeze()
            pred = np.clip(logits, a_min=1, a_max=10)
            pred = np.round(pred)
            batch_score = compute_score_with_logits(pred, a.cuda())
            score += batch_score
            print('index:', j)
            # print('gif name:', dataset.entries[j]['gif_name'])
            print('question:',dataset.entries[j]['question'])
            print('pred answer:', pred[0])
            print('gth answer:', a.numpy()[0])

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
            logits = self.forward(v, q_w, q_c, None)
            pred = logits.data.cpu().numpy().squeeze()
            # pred = pred.data.cpu().numpy().squeeze()
            prediction = np.clip(np.round(pred), a_min=1, a_max=10)
            # pred = np.clip(logits, a_min=1, a_max=10)
            # pred = pred.data.cpu().numpy().squeeze()
            # prediction = np.clip(np.round(pred), a_min=1, a_max=10)
            # pred_label = pred.data.cpu().max(1)[1]
            # batch_score = compute_score_with_logits(pred, a.cuda())

            # prediction = torch.max(pred, 1)[1]
            # score += batch_score
            # num_data += pred.size(0)

            # write in json
            # prediction = np.array(prediction.cpu().data)
            for ques, pred, ans, idx in zip(ques_eng, list(prediction), ans_eng, idx):
                ins = {}
                ins['index'] = j
                # ins['gif_name'] = gif_name
                # ins['key'] = key.data
                ins['id'] = int(idx)
                ins['question'] = ques
                ins['prediction'] = int(pred)
                ins['answer'] = ans.item()
                ins['prediction_success'] = bool(ans.item() == pred)
                results.append(ins)
                j += 1
        with open('data/prediction/Count_prediction.json', 'w') as f:
            json.dump(results, f)
        score = float(score) / len(dataloader.dataset)
        return score


def build_temporalAtt(task_name, n_layer, dataset, num_hid, dictionary, glove_file):
    vid_encoder = Encoder(n_layer=n_layer, n_head=8, d_k=256, d_v=256, v_len=36, v_emb_dim=300,
                          d_model=2048, d_inner_hid=512, dropout=0.1)
    w = WordEmbedding(dictionary.ntoken, dictionary.c_ntoken, 300, 64, 0.1)
    word_mat, char_mat = w.init_embedding(dictionary, glove_file, task_name)
    ques_encoder = Ques_Encoder(word_mat, char_mat)
    classifier = SimpleClassifier(
        num_hid, num_hid * 2, 1, 0.5)
    # classifier = weight_norm(nn.Linear(num_hid, 1), dim=None)
    return CountModel(task_name, vid_encoder, ques_encoder, classifier)


def compute_score_with_logits(logits, labels):
    pred_y = logits
    target_y = labels.cpu().numpy().squeeze()
    score = sum(np.square(np.array(pred_y)-target_y))
    return score
