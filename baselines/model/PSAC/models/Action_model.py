import torch.nn as nn
from .language_model import *
from .classifier import SimpleClassifier
from .fc import FCNet
import torch
from torch.autograd import Variable
import numpy as np
from .attention import NewAttention
import time
from .model_utils import *
import torch.nn.functional as F


class ActionModel(nn.Module):
    def __init__(self, model_name, vid_encoder, ques_encoder, classifier, num_choice=5):
        super(ActionModel, self).__init__()
        self.model_name = model_name
        self.vid_encoder = vid_encoder
        self.ques_encoder = ques_encoder
        self.classifier = classifier
        self.num_choice = num_choice

    def forward(self, v, q_w, q_c, labels):
        """Forward

        v: [batch, multi-choice, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        vid_output = self.vid_encoder(v)
        vid_output = vid_output + v
        # vid_output = v
        fus_output = self.ques_encoder(vid_output, q_w, q_c)
        logits = self.classifier(fus_output)
        logits = logits.view(int(q_w.shape[0] / self.num_choice), self.num_choice)

        return logits

    def evaluate(self, dataloader):
        score = 0
        num_data = 0
        for v, q_w, q_c, a, ques_eng, ans_eng, idx in iter(dataloader):
            q_w = np.array(q_w)
            q_w = torch.from_numpy(q_w.reshape(-1, q_w.shape[-1]))
            q_c = np.array(q_c)
            q_c = torch.from_numpy(q_c.reshape(-1, q_c.shape[-2], q_c.shape[-1]))
            v = np.array(v)
            v = np.tile(v, [1, self.num_choice]).reshape(-1, v.shape[-2], v.shape[-1])
            v = Variable(torch.from_numpy(v).cuda())
            q_w = Variable(q_w.cuda())
            q_c = Variable(q_c.cuda())
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
        results =[]
        for v, q_w, q_c, a, ques_eng, ans_eng, idx in iter(dataloader):
            q_w = np.array(q_w)
            q_w = torch.from_numpy(q_w.reshape(-1, q_w.shape[-1]))
            q_c = np.array(q_c)
            q_c = torch.from_numpy(q_c.reshape(-1, q_c.shape[-2], q_c.shape[-1]))
            v = np.array(v)
            v = np.tile(v, [1, self.num_choice]).reshape(-1, v.shape[-2], v.shape[-1])
            v = Variable(torch.from_numpy(v).cuda())
            q_w = Variable(q_w.cuda())
            q_c = Variable(q_c.cuda())
            pred = self.forward(v, q_w, q_c, None)
            # pred_label = pred.data.cpu().max(1)[1]
            batch_score = compute_score_with_logits(pred, a.cuda())
            prediction = torch.max(pred, 1)[1]
            score += batch_score
            num_data += pred.size(0)

            # write in json
            prediction = np.array(prediction.cpu().data)
            for ques, pred, ans ,idx in zip(ques_eng, list(prediction), ans_eng, idx):
                ins = {}
                ins['index'] = j
                # ins['gif_name'] = gif_name
                #ins['key'] = key.data
                ins['id'] = int(idx)
                ins['question'] = ques
                ins['prediction'] = int(pred)
                ins['answer'] = ans.item()
                ins['prediction_success'] = bool(ans.item() == pred)
                results.append(ins)
                j += 1
        with open('data/prediction/Action_prediction.json', 'w') as f:
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
    return ActionModel(task_name, vid_encoder, ques_encoder, classifier)


def compute_score_with_logits(logits, labels):
    # logits = logits.view(labels.size(0), 5)
    logits = torch.max(logits, 1)[1]
    pred_y = logits.data.cpu().numpy().squeeze()
    target_y = labels.cpu().numpy().squeeze()
    scores = sum(pred_y==target_y)
    return scores
