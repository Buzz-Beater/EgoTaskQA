import json
import torch
from torch.utils.data import DataLoader, Dataset
import glob
import PIL
import numpy as np
import os
import torchvision.transforms as transforms
import h5py, pickle


class LEMMA(Dataset):
    def __init__(self, tagged_qas_path, img_size=(224, 224), mode='train', 
                num_of_sampled_frames=20, video_feature_path='/home/leiting/scratch/lemma_simple_model/data/video_feature_20.h5') -> None:
        super().__init__()
        with open(tagged_qas_path, 'r') as f:
            self.tagged_qas = json.load(f)
        self.img_size = img_size
        self.mode = mode
        self.num_of_sampled_frames = num_of_sampled_frames
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize(img_size)
        
        self.video_feature = h5py.File(video_feature_path, 'r')
 

    def __len__(self):
        return len(self.tagged_qas)

    def __getitem__(self, index):
        item = self.tagged_qas[index]
        question_encode = item['question_encode']
        reasoning_type = item['reasoning_type'].split('$') # # list of string
        question_encode = torch.from_numpy(np.array(question_encode)).long()
        answer_encode = torch.tensor(int(item['answer_encode'])).long()
        
        vgg = torch.from_numpy(self.video_feature['vgg_features'][item['video_id']])
        c3d = torch.from_numpy(self.video_feature['c3d_features'][item['video_id']])
        
        return question_encode, answer_encode, vgg, c3d, reasoning_type


def collate_func(batch):
    question_encode_lst, answer_encode_lst, vgg_lst, c3d_lst = [], [], [], []
    question_length_lst = []
    reasoning_type_lst = []
    for i, (question_encode, answer_encode, vgg, c3d, reasoning_type) in enumerate(batch):
        question_encode_lst.append(question_encode)
        answer_encode_lst.append(answer_encode)
        vgg_lst.append(vgg)
        c3d_lst.append(c3d)
        question_length_lst.append(len(question_encode))
        reasoning_type_lst.append(reasoning_type)

    question_encode_lst = torch.nn.utils.rnn.pad_sequence(question_encode_lst, batch_first=True, padding_value=0)
    answer_encode_lst = torch.tensor(answer_encode_lst)
    vgg_lst = torch.stack(vgg_lst, dim=0)
    c3d_lst = torch.stack(c3d_lst, dim=0)

    # # torch.Size([4, 12]) torch.Size([4]) torch.Size([4, 20, 4096]) torch.Size([4, 20, 4096])
    return question_encode_lst, answer_encode_lst, vgg_lst, c3d_lst, question_length_lst, reasoning_type_lst


if __name__ == '__main__':
    dataset = LEMMA('/home/leiting/scratch/lemma_simple_model/data/formatted_test_qas_encode.json', 
                    (224, 224), 'train', 20, 
                    video_feature_path='/home/leiting/scratch/lemma_simple_model/data/video_feature_20.h5')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_func)
    for i, (question_encode, answer_encode, vgg, c3d, question_length_lst, reasoning_type_lst) in enumerate(dataloader):
        print(i, question_encode.shape, answer_encode.shape, vgg.shape, c3d.shape)
        print('question_length_lst length:', len(question_length_lst))
        break
