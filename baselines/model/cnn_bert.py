from turtle import pd
import torch
import torch.nn as nn
import torchvision
import math
import numpy as np
import pickle

# BERT Parameters
# maxlen = 100 # vocab.json:maxlen 
n_layers = 6
n_heads = 12
d_model = 300
d_ff = d_model*4 # 4*d_model, FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
# n_segments = 3 


# vocab_size = 26 # todo -- size of question_set

device_id = 0
device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")


def build_resnet(model_name, pretrained=False):
    cnn = getattr(torchvision.models, model_name)(pretrained=pretrained)
    model = torch.nn.Sequential(*list(cnn.children())[:-1])
    return model

def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k(=len_q), one is masking
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k


def gelu(x):
    "Implementation of the gelu activation function by Hugging Face"
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class Embedding(nn.Module):
    def __init__(self, feature_size=2048, vocab_size=10, maxlen=100,n_segments=3, glove=None):
        super(Embedding, self).__init__()
        if glove is not None:
            self.tok_embed = nn.Embedding.from_pretrained(torch.from_numpy(glove), freeze=False)
        else:
            self.tok_embed = nn.Embedding(vocab_size, d_model)  # token embedding
        self.pos_embed = nn.Embedding(maxlen, d_model)  # position embedding
        self.seg_embed = nn.Embedding(n_segments, d_model)  # segment(token type) embedding
        self.feature_dim2dmodel = nn.Linear(feature_size, d_model)
        self.norm = nn.LayerNorm(d_model)

        

    def forward(self, x, seg, frames_feature):
        seq_len = seg.size(1)
        mapped_feature = self.feature_dim2dmodel(frames_feature)

        pos = torch.arange(seq_len, dtype=torch.long)
        pos = pos.unsqueeze(0).expand_as(seg).to(device)  # (seq_len,) -> (batch_size, seq_len)

        embedding = torch.cat( (mapped_feature, self.tok_embed(x)), dim=1) + self.pos_embed(pos) + self.seg_embed(seg)
        return self.norm(embedding)


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.W_O = nn.Linear(n_heads * d_v, d_model)
        self.LayerNorm = nn.LayerNorm(d_model)
    def forward(self, Q, K, V, attn_mask):
        # q: [batch_size x len_q x d_model], k: [batch_size x len_k x d_model], v: [batch_size x len_k x d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # v_s: [batch_size x n_heads x len_k x d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size x n_heads x len_q x len_k]

        # context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v) # context: [batch_size x len_q x n_heads * d_v]
        output = self.W_O(context)
        return self.LayerNorm(output + residual), attn # output: [batch_size x len_q x d_model]

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # (batch_size, len_seq, d_model) -> (batch_size, len_seq, d_ff) -> (batch_size, len_seq, d_model)
        return self.fc2(gelu(self.fc1(x)))

class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size x len_q x d_model]
        return enc_outputs, attn

class BERT(nn.Module):
    def __init__(self, output_dim, question_pt_path='data/glove.pt', feature_size=2048, maxlen=100,n_segments=3):
        super(BERT, self).__init__()
        with open(question_pt_path, 'rb') as f:
            obj = pickle.load(f)
            self.glove_matrix = obj['glove']
        
        self.vocab_size = self.glove_matrix.shape[0]

        self.embedding = Embedding(feature_size=feature_size, vocab_size=self.vocab_size, maxlen=maxlen, n_segments=n_segments, glove=self.glove_matrix)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(0.5),
            nn.Tanh(),
        )
        self.classifier = nn.Linear(d_model, output_dim)
        
        

    def forward(self, input_ids, segment_ids, frame_features):
        output = self.embedding(input_ids, segment_ids, frame_features)
        enc_self_attn_mask = get_attn_pad_mask(segment_ids, segment_ids)
        for layer in self.layers:
            output, enc_self_attn = layer(output, enc_self_attn_mask)
        # output : [batch_size, len, d_model], attn : [batch_size, n_heads, d_mode, d_model]
        # it will be decided by first token(CLS)
        h_pooled = self.fc(output[:, 0]) # [batch_size, d_model]
        logits_clsf = self.classifier(h_pooled) # [batch_size, output_dim]

        return logits_clsf


if __name__ == '__main__':
    B, num = 2, 3
    # img = torch.randn(B, num, 3, 224, 224)
    # img = img.reshape(-1, 3, 224, 224)
    # model = build_resnet('resnet101')
    # out = model(img)
    # out = out.reshape(B, num, -1)
    # print(out.shape)
    
    # # input sentence: <cls> will he pick up the apple
    # # input visual: frame1 frame10 frame20
    # # embedded input: <cls> mapped_frame1 mapped_frame10 mapped_Frame20 will he pick up the apple

    feature = torch.rand(B, num, 2048)
    sentences = torch.ones(B, 12, dtype=torch.long)
    visual_segments = torch.ones(B, num, dtype=torch.long)
    question_segments = torch.ones(B, 12, dtype=torch.long)
    question_segments[:, -1] = 0
    
    segments = torch.cat((question_segments[:, 0:1], visual_segments, question_segments[:, 1:,]), dim=1)

    model = BERT(2)
    # # sentences: B x 12
    # # segments: B x (num+12) 
    # # feature B x num x 2048
    logits = model(sentences, segments, feature)
    print(logits.shape) # # B x otuput_dim