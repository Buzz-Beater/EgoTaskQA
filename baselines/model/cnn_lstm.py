import torch
import torch.nn as nn
import torchvision
import math
import numpy as np
import pickle


d_model = 300

device_id = 0
device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")


def build_resnet(model_name, pretrained=False):
    cnn = getattr(torchvision.models, model_name)(pretrained=pretrained)
    model = torch.nn.Sequential(*list(cnn.children())[:-1])
    return model

class Embedding(nn.Module):
    def __init__(self, feature_size=2048, vocab_size=10,glove=None):
        super(Embedding, self).__init__()
        if glove is not None:
            self.tok_embed = nn.Embedding.from_pretrained(torch.from_numpy(glove), freeze=True)
        else:
            self.tok_embed = nn.Embedding(vocab_size, d_model)  # token embedding
        # self.tok_embed = nn.Embedding(vocab_size, d_model)  # token embedding
        # self.pos_embed = nn.Embedding(maxlen, d_model)  # position embedding
        # self.seg_embed = nn.Embedding(n_segments, d_model)  # segment(token type) embedding
        self.feature_dim2dmodel = nn.Linear(feature_size, d_model)
        self.norm = nn.LayerNorm(d_model)

        # if glove is not None:
        #     self.tok_embed.weight = nn.Parameter(torch.from_numpy(glove))

    def forward(self, x, frames_feature):
        # seq_len = seg.size(1)
        mapped_feature = self.feature_dim2dmodel(frames_feature)

        # pos = torch.arange(seq_len, dtype=torch.long)
        # pos = pos.unsqueeze(0).expand_as(seg).to(device)  # (seq_len,) -> (batch_size, seq_len)

        embedding = torch.cat( (mapped_feature, self.tok_embed(x)), dim=1)
        return self.norm(embedding)


class base_lstm(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(base_lstm, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        # self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x): # # lstm_input: B x seq x 2048
        # Set initial states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        h0, c0 = h0.to(device),c0.to(device)
        # Forward propagate RNN
        out, _ = self.lstm(x, (h0, c0))

        # Decode hidden state of last time step
        out = self.fc(out[:, -1, :])
        # out = self.softmax(out)
        return out

class lstm(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes,
                question_pt_path='data/glove.pt', feature_size=2048,):
        super(lstm, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = base_lstm(input_size, hidden_size, num_layers, num_classes)
        
        with open(question_pt_path, 'rb') as f:
            obj = pickle.load(f)
            self.glove_matrix = obj['glove']
        
        self.vocab_size = self.glove_matrix.shape[0]
        assert input_size == self.glove_matrix.shape[1]

        self.embedding = Embedding(feature_size, self.vocab_size, self.glove_matrix)

    def forward(self, x, frames_feature):
        
        embedded_x = self.embedding(x, frames_feature)
        # Forward propagate RNN
        
        logits = self.lstm(embedded_x)
        return logits

if __name__ == '__main__':
    B, num = 2, 3

    feature = torch.rand(B, num, 2048)
    sentences = torch.ones(B, 12, dtype=torch.long)

    lstm = lstm(300, 256, 1, num_classes=10)
    output = lstm(sentences, feature)
    print(output.shape) # # B x num_classses