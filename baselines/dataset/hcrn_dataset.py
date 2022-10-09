import json
from turtle import pd
from cv2 import MOTION_AFFINE
import torch
from torch.utils.data import DataLoader, Dataset
import glob
import PIL
import numpy as np
import os
import torchvision.transforms as transforms
import h5py, pickle


class LEMMA(Dataset):
    def __init__(self, tagged_qas_path,  mode, app_feature_h5, motion_feature_h5) -> None:
        super().__init__()
        with open(tagged_qas_path, 'r') as f:
            self.tagged_qas = json.load(f)
        self.mode = mode
        
        print('loading appearance feature from %s' % (app_feature_h5))
        with h5py.File(app_feature_h5, 'r') as app_features_file:
            app_video_ids = app_features_file['ids'][()]
        self.app_feat_id_to_index = {str(id): i for i, id in enumerate(app_video_ids)}
        print('loading motion feature from %s' % (motion_feature_h5))
        with h5py.File(motion_feature_h5, 'r') as motion_features_file:
            motion_video_ids = motion_features_file['ids'][()]
        self.motion_feat_id_to_index = {str(id): i for i, id in enumerate(motion_video_ids)}
        
        self.app_feature_h5 = app_feature_h5
        self.motion_feature_h5 =motion_feature_h5
    
    def __len__(self):
        return len(self.tagged_qas)

    def __getitem__(self, index):
        item = self.tagged_qas[index]
        question = item['question']
        reasoning_type = item['reasoning_type'].split('$') # # list of string
        question_encode = item['question_encode']

        question_encode = torch.from_numpy(np.array(question_encode)).long()
        answer_encode = torch.tensor(int(item['answer_encode'])).float()
        
        video_idx = item['video_id']
        app_index = self.app_feat_id_to_index[str(video_idx)]
        motion_index = self.motion_feat_id_to_index[str(video_idx)]

        with h5py.File(self.app_feature_h5, 'r') as f_app:
            appearance_feat = f_app['resnet_features'][app_index]  # (8, 16, 2048)
        with h5py.File(self.motion_feature_h5, 'r') as f_motion:
            motion_feat = f_motion['resnext_features'][motion_index]  # (8, 2048)


        appearance_feat = torch.from_numpy(appearance_feat)
        motion_feat = torch.from_numpy(motion_feat)

        # # torch.Size([4]) torch.Size([4, 8, 16, 2048]) torch.Size([4, 8, 2048]) torch.Size([4, 28])
        return answer_encode, appearance_feat, motion_feat, question_encode, reasoning_type


def collate_func(batch):
    answer_encode_lst,  appearance_feat_lst, motion_feat_lst, question_encode_lst = [], [], [], []
    reasoning_type_lst = []
    question_len_lst = []

    for i, (answer_encode, appearance_feat, motion_feat, question_encode, reasoning_type) in enumerate(batch):
        question_encode_lst.append(question_encode)
        answer_encode_lst.append(answer_encode)
        reasoning_type_lst.append(reasoning_type)
        appearance_feat_lst.append(appearance_feat)
        motion_feat_lst.append(motion_feat)
        question_len_lst.append(len(question_encode))

    question_encode_lst = torch.nn.utils.rnn.pad_sequence(question_encode_lst, batch_first=True, padding_value=0)
    answer_encode_lst = torch.tensor(answer_encode_lst)
    appearance_feat_lst = torch.stack(appearance_feat_lst, dim=0)
    motion_feat_lst = torch.stack(motion_feat_lst, dim=0)

    # # 
    return answer_encode_lst, appearance_feat_lst, motion_feat_lst, question_encode_lst, question_len_lst, reasoning_type_lst


if __name__ == '__main__':
    dataset = LEMMA('/home/leiting/scratch/lemma_simple_model/data/formatted_test_qas_encode.json', 
                    mode='train',
                    app_feature_h5='data/hcrn_data/lemma-qa_appearance_feat.h5',
                    motion_feature_h5='data/hcrn_data/lemma-qa_motion_feat.h5')

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_func)
    for i, (answer_encode, appearance_feat, motion_feat, question_encode, question_len_lst, reasoning_type_lst) in enumerate(dataloader):
        print(i, answer_encode.shape, appearance_feat.shape, motion_feat.shape, question_encode.shape)
        print(len(question_len_lst))
        break