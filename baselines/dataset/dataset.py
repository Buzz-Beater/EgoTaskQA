import json
import torch
from torch.utils.data import DataLoader, Dataset
import glob
import PIL
import numpy as np
import os
import torchvision.transforms as transforms
import h5py


class LEMMA(Dataset):
    def __init__(self, tagged_qas_path, img_size=(224, 224), mode='train',
                 num_of_sampled_frames=20, use_preprocessed_features=True,
                all_qa_interval_path='data/vid_intervals.json', feature_base_path='/scratch/generalvision/LEMMA/video_features') -> None:
        super().__init__()
        with open(tagged_qas_path, 'r') as f:
            self.tagged_qas = json.load(f)
        self.img_size = img_size
        self.mode = mode
        self.num_of_sampled_frames = num_of_sampled_frames
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize(img_size)
        self.use_preprocessed_features = use_preprocessed_features
        self.feature_base_path = feature_base_path
        self.all_features = []
        self.preload_visual_features(all_qa_interval_path=all_qa_interval_path)
        print('preload all features for', len(self.all_features), 'videos')
        

    def load_frame_paths_from_interval(self, interval):
        video_name, fpv, start, end = interval.split('|') # # [start, end)
        start, end = int(start) + 1 , int(end) + 1 # # 
        start, end = str(start), str(end)
        start, end = start.rjust(5, '0'), end.rjust(5, '0')
        video_path = os.path.join('/scratch/generalvision/LEMMA/videos', video_name, f'fpv{fpv[-1]}', 'img_*.jpg')
        all_video_frames = sorted(glob.glob(video_path))
        start_frame = os.path.join('/scratch/generalvision/LEMMA/videos', video_name, f'fpv{fpv[-1]}', f'img_{start}.jpg')
        end_frame = os.path.join('/scratch/generalvision/LEMMA/videos', video_name, f'fpv{fpv[-1]}', f'img_{end}.jpg')
        start_idx = all_video_frames.index(start_frame)
        end_idx = all_video_frames.index(end_frame)
        return all_video_frames[start_idx:end_idx]
    
    def sample_frame_paths(self, frame_paths, num_of_sampled_frames):
        # # uniformly sample frames, input: L x 3 x H x W, output: num_of_sampled_frames x 3 x H x W
        res_frame_paths = []
        for i in range(0, len(frame_paths), len(frame_paths) // num_of_sampled_frames):
            res_frame_paths.append(frame_paths[i])
        return res_frame_paths[:num_of_sampled_frames]
    
    def sample_frame_features(self, interval, num_of_sampled_frames):
        video_name, fpv, start, end = interval.split('|') # # [start, end)
        start_idx, end_idx = int(start) , int(end) # # 
        fpv = f'fpv{fpv[-1]}'

        filename = os.path.join(self.feature_base_path, video_name, fpv, 'resnet101_feat.h5')

        # filename_lst = frame_paths[0].split('/')[:-1]
        # filename_lst[filename_lst.index('videos')] = 'video_features'

        # start_idx = int(frame_paths[0].split('/')[-1][4:9]) - 1
        # end_idx = int(frame_paths[-1].split('/')[-1][4:9])
        
        # filename = '/'.join(filename_lst) + '/resnet101_feat.h5'

        res = []
        with h5py.File(filename, "r") as f:
            # list all groups
            # print("Keys: %s" % f.keys())
            a_group_key = list(f.keys())[0]
            # Get the data
            data = list(f[a_group_key])
            for i in range(start_idx, end_idx, (end_idx - start_idx) // num_of_sampled_frames):
                res.append(data[i])
        return res[:num_of_sampled_frames]

    def __len__(self):
        return len(self.tagged_qas)

    def __getitem__(self, index):
        item = self.tagged_qas[index]
        question = item['question']
        reasoning_type = item['reasoning_type'].split('$') # # list of string
        question_encode = item['question_encode']

        question_encode = torch.from_numpy(np.array(question_encode)).long()
        answer_encode = torch.tensor(int(item['answer_encode'])).float()
        question_char_encode = torch.from_numpy(np.array(item['question_char_encode'])).long()
        
        if self.use_preprocessed_features == False:
            frame_paths = self.sample_frame_paths(self.load_frame_paths_from_interval(item['interval']), self.num_of_sampled_frames)
            frame_rgbs = []
            for frame_path in frame_paths:
                frame_rgb = PIL.Image.open(frame_path).convert('RGB')
                frame_rgb = self.resize(self.to_tensor(frame_rgb))
                frame_rgbs.append(frame_rgb)
            frame_rgbs = torch.stack(frame_rgbs, dim=0)

            frame_features = torch.zeros(3)
        else:
            # frame_features = self.sample_frame_features(self.load_frame_paths_from_interval(item['interval']), self.num_of_sampled_frames)
            # frame_features = torch.from_numpy(np.stack(frame_features, axis=0))
            frame_features = self.all_features[item['video_id']]
            frame_rgbs = torch.zeros(self.num_of_sampled_frames, 3, self.img_size[0], self.img_size[1])
        
        return frame_rgbs, question_encode, answer_encode, frame_features, question_char_encode, question, reasoning_type
    
    def preload_visual_features(self, all_qa_interval_path):
        print('preloading all visual features...')
        with open(all_qa_interval_path, 'r') as f:
            all_qa_interval_lst = json.load(f)
        
        for interval in all_qa_interval_lst:
            frame_features = self.sample_frame_features(interval, self.num_of_sampled_frames)
            frame_features = torch.from_numpy(np.stack(frame_features, axis=0)) # # num_of_sampled_frames x 2048
            self.all_features.append(frame_features)

        

def collate_func(batch):
    frame_rgbs_lst, question_encode_lst, answer_encode_lst, question_char_encode_lst = [], [], [], []
    frame_features_lst = []
    question_lst = []
    reasoning_type_lst = []
    for i, (frame_rgbs, question_encode, answer_encode, frame_features, question_char_encode, question, reasoning_type) in enumerate(batch):
        frame_rgbs_lst.append(frame_rgbs)
        question_encode_lst.append(question_encode)
        answer_encode_lst.append(answer_encode)
        frame_features_lst.append(frame_features)
        question_char_encode_lst.append(question_char_encode)
        question_lst.append(question)
        reasoning_type_lst.append(reasoning_type)

    frame_rgbs_lst = torch.stack(frame_rgbs_lst, dim=0)
    question_encode_lst = torch.nn.utils.rnn.pad_sequence(question_encode_lst, batch_first=True, padding_value=0)
    answer_encode_lst = torch.tensor(answer_encode_lst)
    frame_features_lst = torch.stack(frame_features_lst, dim=0)
    question_char_encode_lst = torch.stack(question_char_encode_lst, dim=0)

    # # torch.Size([4, 20, 3, 224, 224]) torch.Size([4, 12]) torch.Size([4]) torch.Size([4, 20, 2048]) torch.Size([4, 25, 15])
    return frame_rgbs_lst, question_encode_lst, answer_encode_lst, frame_features_lst, question_char_encode_lst, question_lst, reasoning_type_lst


if __name__ == '__main__':
    dataset = LEMMA('/home/leiting/scratch/lemma_simple_model/data/formatted_test_qas_encode.json', (224, 224), 'train', 20, True)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_func)
    for i, (frame_rgbs, question_encode, answer_encode, frame_features, question_char_encode, question, reasonging_type) in enumerate(dataloader):
        print(i, frame_rgbs.shape, question_encode.shape, answer_encode.shape, frame_features.shape, question_char_encode.shape)
        print(len(question))
        break