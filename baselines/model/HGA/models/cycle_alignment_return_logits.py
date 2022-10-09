import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .q_v_transformer import padding_mask_embedded
# torch.set_printoptions(threshold=np.inf)


def padding_mask_k(seq_q, seq_k):
    """ seq_k of shape (batch, k_len, k_feat) and seq_q (batch, q_len, q_feat). q and k are padded with 0. pad_mask is (batch, q_len, k_len).
    In batch 0:
    [[x x x 0]     [[0 0 0 1]
     [x x x 0]->    [0 0 0 1]
     [x x x 0]]     [0 0 0 1]] uint8
    """
    fake_q = torch.ones_like(seq_q)
    pad_mask = torch.bmm(fake_q, seq_k.transpose(1, 2))
    pad_mask = pad_mask.eq(0)
    return pad_mask


def padding_mask_q(seq_q, seq_k):
    """ seq_k of shape (batch, k_len, k_feat) and seq_q (batch, q_len, q_feat). q and k are padded with 0. pad_mask is (batch, q_len, k_len).
    In batch 0:
    [[x x x x]      [[0 0 0 0]
     [x x x x]  ->   [0 0 0 0]
     [0 0 0 0]]      [1 1 1 1]] uint8
    """
    fake_k = torch.ones_like(seq_k)
    pad_mask = torch.bmm(seq_q, fake_k.transpose(1, 2))
    pad_mask = pad_mask.eq(0)
    return pad_mask


def classification_loss(logits, labels):
    """Loss function based on classifying the correct indices.
  In the paper, this is called Cycle-back Classification.
  Args:
    logits: Tensor, Pre-softmax scores used for classification loss. These are
      similarity scores after cycling back to the starting sequence.
    labels: Tensor, One hot labels containing the ground truth. The index where
      the cycle started is 1.
    label_smoothing: Float, label smoothing factor which can be used to
      determine how hard the alignment should be.
  Returns:
    loss: Tensor, A scalar classification loss calculated using standard softmax
      cross-entropy loss.
  """
    return F.cross_entropy(logits, labels)


def regression_loss(
        logits, labels, seq_len, loss_type, normalize_indices, var_lambda):
    """Loss function based on regressing to the correct indices.
  In the paper, this is called Cycle-back Regression. There are 3 variants
  of this loss:
  i) regression_mse: MSE of the predicted indices and ground truth indices.
  ii) regression_mse_var: MSE of the predicted indices that takes into account
  the variance of the similarities. This is important when the rate at which
  sequences go through different phases changes a lot. The variance scaling
  allows dynamic weighting of the MSE loss based on the similarities.
  iii) regression_huber: Huber loss between the predicted indices and ground
  truth indices.
  Args:
    logits: Tensor, Pre-softmax similarity scores after cycling back to the
      starting sequence.
    labels: Tensor, One hot labels containing the ground truth. The index where
      the cycle started is 1.
    num_steps: Integer, Number of steps in the sequence embeddings.
    steps: Tensor, step indices/frame indices of the embeddings of the shape
      [N, T] where N is the batch size, T is the number of the timesteps.
    seq_lens: Tensor, Lengths of the sequences from which the sampling was done.
      This can provide additional temporal information to the alignment loss.
    loss_type: String, This specifies the kind of regression loss function.
      Currently supported loss functions: regression_mse, regression_mse_var,
      regression_huber.
    normalize_indices: Boolean, If True, normalizes indices by sequence lengths.
      Useful for ensuring numerical instabilities don't arise as sequence
      indices can be large numbers.
    variance_lambda: Float, Weight of the variance of the similarity
      predictions while cycling back. If this is high then the low variance
      similarities are preferred by the loss while making this term low results
      in high variance of the similarities (more uniform/random matching).
    huber_delta: float, Huber delta described in tf.keras.losses.huber_loss.
  Returns:
     loss: Tensor, A scalar loss calculated using a variant of regression.
  """
    # logits (bs, seq_len, seq_len), labels (bs, seq_len)

    # steps of shape (bs, seq_len, seq_len) are indexes
    steps = torch.arange(seq_len)[None, None, :].expand_as(logits)
    steps = steps.float()

    beta = F.softmax(logits, dim=2)
    true_time = labels
    # pred_time of shape (bs, seq_len)
    pred_time = torch.sum(steps * beta, 2)

    if loss_type in ['regression_mse', 'regression_mse_var']:
        if 'var' in loss_type:
            tiled_pred_time = pred_time.unsqueeze(1).expand(-1, seq_len, -1)
            # Variance aware regression.
            # pred_time_variance of shape (batch_size, seq_len)
            pred_time_variance = torch.sum(
                torch.pow(steps - tiled_pred_time, 2) * beta, 2)

            # Using log of variance as it is numerically stabler.
            pred_time_log_var = torch.log(pred_time_variance)
            squared_error = torch.pow(true_time - pred_time, 2)
            return torch.mean(
                torch.exp(-pred_time_log_var) * squared_error +
                var_lambda * pred_time_log_var)

        else:
            return F.mse_loss(pred_time, true_time)
    elif loss_type == 'regression_huber':
        return F.smooth_l1_loss(pred_time, true_time)
    else:
        raise ValueError(
            'Unsupported regression loss %s. Supported losses are: '
            'regression_mse, regresstion_mse_var and regression_huber.' %
            loss_type)


# ! correct?
def pairwise_l2_distance_1(embs1, embs2):
    """Computes pairwise distances between all rows of embs1 and embs2."""
    norm1 = torch.sum(torch.pow(embs1, 2), 1)
    norm1 = norm1.view(-1, 1)
    norm2 = torch.sum(torch.pow(embs1, 2), 1)
    norm2 = norm1.view(-1, 1)

    dist = norm1 + norm2 - 2.0 * torch.mm(embs1, embs2.transpose(0, 1))
    # Max to ensure matmul doesn't produce anything negative due to floating
    # point approximations.
    dist = torch.max(dist, torch.zeros_like(dist))

    return dist


def pairwise_l2_distance_2(X1, X2):
    """Computes pairwise distances between all rows of embs1 and embs2."""
    m, d = X1.size()
    n = X2.size()[0]
    # X1 is of shape m*d.
    X1 = torch.unsqueeze(X1, dim=1).expand(m, n, d)
    # X2 is of shape n*d.
    X2 = torch.unsqueeze(X2, dim=0).expand(m, n, d)
    # dist is of shape m*n, where dist[i][j] = sqrt(|X1[i, :] - X[j, :]|^2)
    dist = torch.sum((X1 - X2)**2, dim=2)

    return dist


def get_scaled_similarity(
        embs1, embs2, similarity_type='cosine', temperature=1.0):
    """Returns similarity between each all rows of embs1 and all rows of embs2.
  The similarity is scaled by the number of channels/embedding size and
  temperature.
  Args:
    embs1: Tensor, Embeddings of the shape [M, D] where M is the number of
      embeddings and D is the embedding size.
    embs2: Tensor, Embeddings of the shape [N, D] where N is the number of
      embeddings and D is the embedding size.
    similarity_type: String, Either one of 'l2' or 'cosine'.
    temperature: Float, Temperature used in scaling logits before softmax.
  Returns:
    similarity: Tensor, [M, N] tensor denoting similarity between embs1 and
      embs2.
  """
    # channels = torch.tensor(embs1.size()[1], dtype=torch.float).to(embs1.device)
    # Go for embs1 to embs2.
    if similarity_type == 'cosine':
        embs2 = torch.transpose(embs2, 0, 1)
        # similarity of shape (M, N)
        similarity = torch.mm(embs1, embs2)
    elif similarity_type == 'l2':
        similarity = -1.0 * pairwise_l2_distance_2(embs1, embs2)
    else:
        raise ValueError('similarity_type can either be l2 or cosine.')

    # Scale the distance  by number of channels. This normalization helps with
    # optimization.
    # similarity /= channels
    # Scale the distance by a temperature that helps with how soft/hard the
    # alignment should be.
    similarity /= temperature

    return similarity


def align_pair_of_sequences(
        embs1, embs2, embs1_length, embs2_length, similarity_type, temperature):
    """Align a given pair embedding sequences.
  Args:
    embs1: Tensor, Embeddings of the shape [M, D] where M is the number of
      embeddings and D is the embedding size.
    embs2: Tensor, Embeddings of the shape [N, D] where N is the number of
      embeddings and D is the embedding size.
    similarity_type: String, Either one of 'l2' or 'cosine'.
    temperature: Float, Temperature used in scaling logits before softmax.
  Returns:
     logits: Tensor, Pre-softmax similarity scores after cycling back to the
      starting sequence.
    labels: Tensor, One hot labels containing the ground truth. The index where
      the cycle started is 1.
  """
    # no pad embs1 and embs2
    embs1 = embs1[:embs1_length]
    embs2 = embs2[:embs2_length]

    max_num_steps = torch.tensor(embs1.size()[0])

    # Find distances between embs1 and embs2. sim_12 of shape (M, N)
    sim_12 = get_scaled_similarity(embs1, embs2, similarity_type, temperature)
    # # mask sim_12
    # mask12 = torch.ones_like(sim_12)
    # mask12[:q_length, :v_length] = 0
    # mask12 = mask12.byte()
    # sim_12.masked_fill_(mask12, -np.inf)

    # Softmax the distance.
    softmaxed_sim_12 = F.softmax(sim_12, 1)

    # Calculate soft-nearest neighbors. nn_embs of shape (M, D)
    nn_embs = torch.mm(softmaxed_sim_12, embs2)

    # Find distances between nn_embs and embs1. sim_21 of shape (M, M)
    sim_21 = get_scaled_similarity(nn_embs, embs1, similarity_type, temperature)

    # # mask sim_21
    # mask21 = torch.ones_like(sim_21)
    # mask21[:q_length, :q_length] = 0
    # mask21 = mask21.byte()
    # sim_21.masked_fill_(mask21, -np.inf)

    # logits of shape (M, M)
    logits = sim_21
    # labels of shape (M, )
    labels = torch.arange(max_num_steps).to(embs1.device)

    return logits, labels


def compute_deterministic_alignment_loss(
        question,
        video,
        q_length,
        v_length,
        loss_type='classification',
        similarity_type='cosine',
        temperature=1.0,
        variance_lambda=0.1,
        normalize_indices=None):
    """Compute cycle-consistency loss for all steps in each sequence.
  This aligns each pair of videos in the batch except with itself.
  When aligning it also matters which video is the starting video. So for N
  videos in the batch, we have N * (N-1) alignments happening.
  For example, a batch of size 3 has 6 pairs of sequence alignments.
  Args:
    embs: Tensor, sequential embeddings of the shape [N, T, D] where N is the
      batch size, T is the number of timesteps in the sequence, D is the size
      of the embeddings.
    steps: Tensor, step indices/frame indices of the embeddings of the shape
      [N, T] where N is the batch size, T is the number of the timesteps.
    seq_lens: Tensor, Lengths of the sequences from which the sampling was
    done. This can provide additional information to the alignment loss.
    num_steps: Integer/Tensor, Number of timesteps in the embeddings.
    batch_size: Integer, Size of the batch.
    loss_type: String, This specifies the kind of loss function to use.
      Currently supported loss functions: 'classification', 'regression_mse',
      'regression_mse_var', 'regression_huber'.
    similarity_type: String, Currently supported similarity metrics: 'l2' ,
      'cosine' .
    temperature: Float, temperature scaling used to scale the similarity
      distributions calculated using the softmax function.
    label_smoothing: Float, Label smoothing argument used in
      tf.keras.losses.categorical_crossentropy function and described in this
      paper https://arxiv.org/pdf/1701.06548.pdf.
    variance_lambda: Float, Weight of the variance of the similarity
      predictions while cycling back. If this is high then the low variance
      similarities are preferred by the loss while making this term low
      results in high variance of the similarities (more uniform/random
      matching).
    huber_delta: float, Huber delta described in tf.keras.losses.huber_loss.
    normalize_indices: Boolean, If True, normalizes indices by sequence
      lengths. Useful for ensuring numerical instabilities doesn't arise as
      sequence indices can be large numbers.
  Returns:
    loss: Tensor, Scalar loss tensor that imposes the chosen variant of the
        cycle-consistency loss.
  """

    ### get masked q and v
    # q_v_mask = padding_mask_embedded(question, video)
    # q_q_mask = padding_mask_embedded(question, question)

    labels_list = []
    logits_list = []
    loss_list = []

    batch_size = question.size()[0]

    for i in range(batch_size):
        # logits of shape (q_seq_len, q_seq_len) and labels of shape (q_seq_len, )
        logits, labels = align_pair_of_sequences(
            question[i], video[i], q_length[i], v_length[i], similarity_type,
            temperature)

        if loss_type == 'classification':
            loss = classification_loss(logits, labels)
        elif 'regression' in loss_type:
            # 35
            seq_len = question.size()[1]
            loss = regression_loss(
                logits, labels, seq_len, loss_type, normalize_indices,
                variance_lambda, huber_delta)
        else:
            raise NotImplementedError

        loss_list.append(loss)
        logits_list.append(logits)
        labels_list.append(labels)

    # # logits of shape (bs, q_seq_len, q_seq_len)
    # logits = torch.cat(logits_list, dim=0)
    # # labels of shape (bs, q_seq_len), [[0,1,2,3,4,...],...]
    # labels = torch.cat(labels_list, 0)
    all_loss = torch.mean(torch.tensor(loss_list))

    return all_loss


def get_scaled_similarity_batch(
        embs1, embs2, similarity_type='cosine', temperature=1.0):
    """Returns similarity between each all rows of embs1 and all rows of embs2.
  The similarity is scaled by the number of channels/embedding size and
  temperature.
  """
    # channels = torch.tensor(embs1.size()[2], dtype=torch.float).to(embs1.device)
    # Go for embs1 to embs2.
    if similarity_type == 'cosine':
        embs2 = torch.transpose(embs2, 1, 2)
        # similarity of shape (M, N)
        similarity = torch.bmm(embs1, embs2)
    elif similarity_type == 'l2':
        similarity = -1.0 * pairwise_l2_distance_2(embs1, embs2)
    else:
        raise ValueError('similarity_type can either be l2 or cosine.')

    # Scale the distance  by number of channels. This normalization helps with optimization.
    # similarity /= channels

    # Scale the distance by a temperature that helps with how soft/hard the
    # alignment should be.
    similarity /= temperature

    return similarity


def compute_alignment_loss_batch(
        question,
        video,
        q_length,
        v_length,
        loss_type='classification',
        similarity_type='cosine',
        temperature=1.0,
        variance_lambda=0.1,
        normalize_indices=None):
    """ Matrix operation of a batch data."""
    # question of shape (batch, q_seq_len=35, hidden_size)
    # video of shape (batch, v_seq_len=80, hidden_size)

    batch_size = question.size()[0]

    max_num_steps = torch.tensor(question.size()[1])

    # Find distances between embs1 and embs2. sim_12 of shape (M, N)
    sim_12 = get_scaled_similarity_batch(
        question, video, similarity_type, temperature)
    # # mask sim_12, for padded k
    mask12_k = padding_mask_k(question, video)
    sim_12.masked_fill_(mask12_k, -np.inf)
    # Softmax the distance.
    softmaxed_sim_12 = F.softmax(sim_12, 2)
    # mask for padded q
    mask12_q = padding_mask_q(question, video)
    # [[x x x 0]
    # [x x x 0]
    # [0 0 0 0]]
    softmaxed_sim_12 = softmaxed_sim_12.masked_fill(mask12_q, 0.)

    # Calculate soft-nearest neighbors. nn_embs of shape (M, D)
    nn_embs = torch.bmm(softmaxed_sim_12, video)

    # Find distances between nn_embs and embs1. sim_21 of shape (M, M)
    sim_21 = get_scaled_similarity_batch(
        nn_embs, question, similarity_type, temperature)

    # # # mask sim_21
    # mask21_k = padding_mask_k(nn_embs, question)
    # sim_21.masked_fill_(mask21_k, -1e5)
    # # # mask
    # # mask21_q = padding_mask_q(nn_embs, question)
    # # sim_21.masked_fill_(mask21_q, 0.)

    # logits of shape (bs, q_seq_len, q_seq_len), only [:q_length, :q_length] is not zero, others are zeros.
    logits = sim_21
    # labels of shape (bs, q_seq_len)
    labels = torch.arange(max_num_steps).repeat(batch_size,
                                                1).to(torch.device("cuda"))

    # if loss_type == 'classification':
    #     loss_all = None
    #     for i in range(batch_size):
    #         loss = F.cross_entropy(
    #             logits[i, :q_length[i], :q_length[i]], labels[i, :q_length[i]])
    #         loss_all = loss_all + loss if loss_all is not None else loss
    #     loss = loss_all / float(batch_size)
    # elif 'regression' in loss_type:
    #     # 35
    #     seq_len = question.size()[1]
    #     loss = regression_loss(
    #         logits, labels, seq_len, loss_type, normalize_indices,
    #         variance_lambda, huber_delta)
    # else:
    #     raise NotImplementedError

    return logits, labels