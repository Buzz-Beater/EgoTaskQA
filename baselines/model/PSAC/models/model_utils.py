import torch.nn as nn
import numpy as np
import torch
import torch.nn.init as init
import torch.nn.functional as F
import math

Nh = 8
D = 512
Dchar=64
Dword=300
dropout=0.1
dropout_char = 0.01

def mask_logits(inputs, mask):
    mask = mask.type(torch.float32)
    return inputs + (-1e30) * (1 - mask)

class Initialized_Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, relu=False, stride=1, padding=0, groups=1, bias=False):
        super(Initialized_Conv1d, self).__init__()
        self.out = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups, bias=bias)
        if relu is True:
            self.relu = True
            nn.init.kaiming_normal_(self.out.weight, nonlinearity='relu')
        else:
            self.relu = False
            nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x):
        if self.relu == True:
            return F.relu(self.out(x))
        else:
            return self.out(x)


def get_attn_padding_mask(seq_q, seq_k):
    ''' Indicate the padding-related part to mask '''
    # assert seq_q.dim() == 2 and seq_k.dim() == 2
    # mb_size, len_q = seq_q.size()
    # mb_size, len_k = seq_k.size()
    # pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)   # bx1xsk
    # pad_attn_mask = pad_attn_mask.expand(mb_size, len_q, len_k) # bxsqxsk
    seq_shp = seq_q.shape
    pad_attn_mask = torch.from_numpy(np.ones([seq_shp[0], seq_shp[1]]))
    return pad_attn_mask

def position_encoding_init(v_len, v_emb_dim):

    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / v_emb_dim) for j in range(v_emb_dim)]
        if pos != 0 else np.zeros(v_emb_dim) for pos in range(v_len)
    ])
    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
    return torch.from_numpy(position_enc).type(torch.FloatTensor)

def get_v_mask(batch_size, v_len):
    ins = np.array(range(0, v_len)).reshape([1, v_len])
    ins = np.repeat(ins,batch_size, axis=0)
    return torch.from_numpy(ins).type(torch.LongTensor)

class Linear(nn.Module):
    def __init__(self, d_in, d_out, bias=True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(d_in, d_out, bias=bias)
        init.xavier_normal_(self.linear.weight)

    def forward(self, x):
        return self.linear(x)

class LayerNormalization(nn.Module):
    def __init__(self, d_hid, eps=1e-3):
        super(LayerNormalization, self).__init__()

        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(d_hid), requires_grad=True)
        self.b_2 = nn.Parameter(torch.zeros(d_hid), requires_grad=True)

    def forward(self, z):
        if z.size(1) == 1:
            return z

        mu = torch.mean(z, keepdim=True, dim=-1)
        sigma = torch.std(z, keepdim=True, dim=-1)
        ln_out = (z - mu.expand_as(z))/(sigma.expand_as(z) + self.eps)
        ln_out = ln_out * self.a_2.expand_as(ln_out) + self.b_2.expand_as(ln_out)
        return ln_out

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, attn_dropout=0.1): # 512
        super(ScaledDotProductAttention, self).__init__()
        self.temper = np.power(d_model, 0.5)
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, attn_mask=None): # mask: n_head x batch_size x v_len
        attn = torch.bmm(q, k.transpose(1, 2)) / self.temper # (n_head x mb) x len_v x len_v
        if attn_mask is not None:
            assert attn_mask.size() == attn.size(), \
                'Attention mask shape {} mismatch ' \
                'with Attention logit tensor shape ' \
                '{}.'.format(attn_mask.size(), attn.size())

            attn.data.masked_fill_(attn_mask, -float('inf'))

        attn = self.softmax(attn) # (n_head x mb) x len_v x len_v
        attn= self.dropout(attn)
        output = torch.bmm(attn, v) # (n_head x mb) x len_v x d_v
        return output, attn



class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head # 8
        self.d_k = d_k # 64
        self.d_v = d_v # 64
        self.w_qs = nn.Parameter(torch.FloatTensor(n_head, d_model, d_k)) # 8 x 512 x 64
        self.w_ks = nn.Parameter(torch.FloatTensor(n_head, d_model, d_k))
        self.w_vs = nn.Parameter(torch.FloatTensor(n_head, d_model, d_v))

        self.attention = ScaledDotProductAttention(d_model)
        self.layer_norm = LayerNormalization(d_model)
        self.proj = Linear(n_head * d_v, d_model)
        self.dropout = nn.Dropout(dropout)

        init.xavier_normal_(self.w_qs)
        init.xavier_normal_(self.w_ks)
        init.xavier_normal_(self.w_vs)

    def forward(self, q, k, v, attn_mask=None):
        d_k, d_v = self.d_k, self.d_v
        n_head = self.n_head
        residual = q # batch_size x v_len x ctx_dim

        mb_size, len_q, d_model = q.size() # batch_size , len_v, ctx_dim 2048
        mb_size, len_k, d_model = k.size()
        mb_size, len_v, d_model = v.size()

        q_s = q.repeat(n_head, 1, 1).view(n_head, -1, d_model)  # n_head x (mb_size*len_q) x ctx_dim
        k_s = k.repeat(n_head, 1, 1).view(n_head, -1, d_model)  # n_head x (mb_size*len_k) x ctx_dim
        v_s = v.repeat(n_head, 1, 1).view(n_head, -1, d_model)  # n_head x (mb_size*len_v) x ctx_dim

        # treat the result as a (n_head * mb_size) size batch
        q_s = torch.bmm(q_s, self.w_qs).view(-1, len_q, d_k)  # (n_head*mb_size) x len_q x d_k
        k_s = torch.bmm(k_s, self.w_ks).view(-1, len_k, d_k)  # (n_head*mb_size) x len_k x d_k
        v_s = torch.bmm(v_s, self.w_vs).view(-1, len_v, d_v)  # (n_head*mb_size) x len_v x d_v

        outputs, attns = self.attention(q_s, k_s, v_s, attn_mask=None) # (n_head*mb_size) x len_v x d_v
        outputs = torch.cat(torch.split(outputs, mb_size, dim=0), dim=-1) # mb x len_v x (n_head x d_v)
        outputs = self.proj(outputs)  # mb x len_v x d_model
        outputs = self.dropout(outputs)
        return self.layer_norm(outputs+residual), attns

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_hid, d_inner_hid, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Conv1d(d_hid, d_inner_hid, 1)  # position-wise
        self.w_2 = nn.Conv1d(d_inner_hid, d_hid, 1)  # position-wise
        self.layer_norm = LayerNormalization(d_hid)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x # batch_size x v_len x d_v
        output = self.relu(self.w_1(x.transpose(1, 2)))  # batch_size x d_v x v_len
        output = self.w_2(output).transpose(2, 1)  # batch_size x v_len x d_v
        output = self.dropout(output)
        return self.layer_norm(output + residual)

class SelfAttention(nn.Module):
    def __init__(self):
        super(SelfAttention, self).__init__()
        self.mem_conv = Initialized_Conv1d(D, D*2, kernel_size=1, relu=False, bias=False)
        self.query_conv = Initialized_Conv1d(D, D, kernel_size=1, relu=False, bias=False)

        bias = torch.empty(1)
        nn.init.constant_(bias, 0)
        self.bias = nn.Parameter(bias)

    def forward(self, queries, mask):
        memory = queries # batch_size x D x len_v

        memory = self.mem_conv(memory)
        query = self.query_conv(queries)
        memory = memory.transpose(1, 2)
        query = query.transpose(1, 2)

        Q = self.split_last_dim(query, Nh) # Nh: 8   # batch_size, n_head, D, head_dim
        K, V = [self.split_last_dim(tensor, Nh) for tensor in torch.split(memory, D, dim=2)] # two mat: batch_size, n_head, D, head_dim
        key_depth_per_head = D//Nh
        Q *= key_depth_per_head ** -0.5
        x = self.dot_product_attention(Q, K, V, mask =mask)
        return self.combine_last_two_dim(x.permute(0,2,1,3)).transpose(1,2)

    def dot_product_attention(self, q, k ,v, bias = False, mask = None):
        """dot-product attention.
        Args:
        q: a Tensor with shape [batch, heads, length_q, depth_k]
        k: a Tensor with shape [batch, heads, length_kv, depth_k]
        v: a Tensor with shape [batch, heads, length_kv, depth_v]
        bias: bias Tensor (see attention_bias())
        is_training: a bool of training
        scope: an optional string
        Returns:
        A Tensor.
        """
        logits = torch.matmul(q,k.permute(0,1,3,2))
        if bias:
            logits += self.bias
        if mask is not None:
            shapes = [x if x != None else -1 for x in list(logits.size())]
            mask = mask.view(shapes[0], 1, 1, shapes[-1])
            logits = mask_logits(logits, mask)
        weights = F.softmax(logits, dim=-1)
        # dropping out the attention links for each of the heads
        weights = F.dropout(weights, p=dropout, training=self.training)
        return torch.matmul(weights, v)

    def split_last_dim(self, x, n):
        """Reshape x so that the last dimension becomes two dimensions.
        The first of these two dimensions is n.
        Args:
        x: a Tensor with shape [..., m]
        n: an integer.
        Returns:
        a Tensor with shape [..., n, m/n]
        """
        old_shape = list(x.size()) # [batch_size, D, len_q]
        last = old_shape[-1] # len_q
        new_shape = old_shape[:-1] + [n] + [last // n if last else None] # batch_size, D, n_head, head_dim
        ret = x.view(new_shape)
        return ret.permute(0, 2, 1, 3)# batch_size, n_head, D, head_dim
    def combine_last_two_dim(self, x):
        """Reshape x so that the last two dimension become one.
        Args:
        x: a Tensor with shape [..., a, b]
        Returns:
        a Tensor with shape [..., ab]
        """
        old_shape = list(x.size())
        a, b = old_shape[-2:]
        new_shape = old_shape[:-2] + [a * b if a and b else None]
        ret = x.contiguous().view(new_shape)
        return ret

class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        self.conv2d = nn.Conv2d(Dchar, D, kernel_size = (1,5), padding=0, bias=True)
        nn.init.kaiming_normal_(self.conv2d.weight, nonlinearity='relu')
        self.conv1d = Initialized_Conv1d(Dword+D, D, bias=False)
        self.high = Highway(2)

    def forward(self, ch_emb, wd_emb, length):
        N = ch_emb.size()[0] # batch_size
        ch_emb = ch_emb.permute(0, 3, 1, 2) # batch_size x c_dim x seq_q x seq_c
        ch_emb = F.dropout(ch_emb, p=dropout_char, training=self.training)
        ch_emb = self.conv2d(ch_emb) # batch_size x D x seq_q x 96
        ch_emb = F.relu(ch_emb)
        ch_emb, _ = torch.max(ch_emb, dim=3)
        ch_emb = ch_emb.squeeze()  # batch_size x D x seq_q

        wd_emb = F.dropout(wd_emb, p=dropout, training=self.training)  # batch_size x seq_w x w_dim
        wd_emb = wd_emb.transpose(1, 2) # batch_size x w_dim x seq_q
        emb = torch.cat([ch_emb, wd_emb], dim=1)  # batch_size x (D+w_dim) x seq_q
        emb = self.conv1d(emb) # batch_size x D x seq_q
        emb = self.high(emb)
        return emb

class Highway(nn.Module):
    def __init__(self, layer_num, size=D):
        super(Highway, self).__init__()
        self.n = layer_num
        self.linear = nn.ModuleList([Initialized_Conv1d(size, size, relu=False, bias=True) for _ in range(self.n)])
        self.gate = nn.ModuleList([Initialized_Conv1d(size, size, bias=True) for _ in range(self.n)])

    def forward(self, x):
        #x: shape [batch_size, hidden_size, length]
        for i in range(self.n):
            gate = torch.sigmoid(self.gate[i](x))
            nonlinear = self.linear[i](x)
            nonlinear = F.dropout(nonlinear, p=dropout, training=self.training)
            x = gate * nonlinear + (1 - gate) * x
            #x = F.relu(x)
        return x


def PosEncoder(x, min_timescale=1.0, max_timescale=1.0e4):
    x = x.transpose(1, 2)# batch_size x len x D
    length = x.size()[1]
    channels = x.size()[2]
    signal = get_timing_signal(length, channels, min_timescale, max_timescale)
    return (x + signal.cuda()).transpose(1, 2) # batch_size x D x len

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, k, bias=True):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=in_ch, kernel_size=k, groups=in_ch, padding=k // 2, bias=False)
        self.pointwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0, bias=bias)
    def forward(self, x):
        return F.relu(self.pointwise_conv(self.depthwise_conv(x)))

def get_timing_signal(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    position = torch.arange(length).type(torch.float32)
    num_timescales = channels // 2
    log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) / (float(num_timescales)-1))
    inv_timescales = min_timescale * torch.exp(
            torch.arange(num_timescales).type(torch.float32) * -log_timescale_increment)
    scaled_time = position.unsqueeze(1) * inv_timescales.unsqueeze(0) # 48
    signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim = 1)
    m = nn.ZeroPad2d((0, (channels % 2), 0, 0)) # 35 x 96
    signal = m(signal)
    signal = signal.view(1, length, channels)
    return signal

class EncoderBlock(nn.Module):
    def __init__(self, conv_num, ch_num, k): # k=7
        super(EncoderBlock, self).__init__()
        self.convs = nn.ModuleList([DepthwiseSeparableConv(ch_num, ch_num, k) for _ in range(conv_num)])
        self.self_att = SelfAttention()
        self.FFN_1 = Initialized_Conv1d(ch_num, ch_num, relu=True, bias=True)
        self.FFN_2 = Initialized_Conv1d(ch_num, ch_num, bias=True)
        self.norm_C = nn.ModuleList([nn.LayerNorm(D) for _ in range(conv_num)])
        self.norm_1 = nn.LayerNorm(D)
        self.norm_2 = nn.LayerNorm(D)
        self.conv_num = conv_num
    def forward(self, x, mask, l, blks): #x : batch_size x D x len_v     blks:1
        total_layers = (self.conv_num+1)*blks  # 5
        out = PosEncoder(x) # batch_size x D x len_v
        for i, conv in enumerate(self.convs):
            res = out
            out = self.norm_C[i](out.transpose(1,2)).transpose(1,2)
            if (i) % 2 == 0:
                out = F.dropout(out, p=dropout, training=self.training)
            out = conv(out)
            out = self.layer_dropout(out, res, dropout*float(l)/total_layers)
            l += 1
        res = out
        out = self.norm_1(out.transpose(1,2)).transpose(1,2)
        out = F.dropout(out, p=dropout, training=self.training)
        out = self.self_att(out, mask) # batch_size x D x len_v
        out = self.layer_dropout(out, res, dropout*float(l)/total_layers)
        l += 1
        res = out

        out = self.norm_2(out.transpose(1,2)).transpose(1, 2)
        out = F.dropout(out, p=dropout, training=self.training)
        out = self.FFN_1(out)
        out = self.FFN_2(out)
        out = self.layer_dropout(out, res, dropout*float(l)/total_layers)
        return out

    def layer_dropout(self, inputs, residual, dropout):
        if self.training == True:
            pred = torch.empty(1).uniform_(0,1) < dropout
            if pred:
                return residual
            else:
                return F.dropout(inputs, dropout, training=self.training) + residual
        else:
            return inputs + residual


