import os
import sys
import json
import h5py

from six import iteritems
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset


class VQA_Dataset(Dataset):

    @staticmethod
    def extend_args(parser):
        parser.add_argument_group('Dataset related arguments')
        parser.add_argument('--output_h5', type=str, default='Data/data_prepro.hdf5', help='Output hdf5 dataset')
        parser.add_argument('--output_json', type=str, default='Data/data_prepro.json', help='Output json dataset')
        parser.add_argument('--img_norm', type=int, default=1, choices=[0, 1], help='Normalize image or not')

        return parser

    def right_align(self, seq, lengths):
        v = np.zeros(np.shape(seq))
        N = np.shape(seq)[1]
        for i in range(np.shape(seq)[0]):
            v[i][N - lengths[i]:N - 1] = seq[i][0:lengths[i] - 1]
            return v

    def __init__(self, args, dtype):
        super().__init__()

        self.args = args
        self.dtype = dtype
        print ('Loading hdf5 and json files ..........')

        self.data_json = {}
        self.data = {}

        with open(self.args['output_json']) as f:
            self.data_file = json.load(f)

        for key in self.data_file.keys():
            self.data_json[key] = self.data_file[key]

        with h5py.File(self.args['output_h5'], 'r') as f:
            self.data['questions'] = np.asarray(f.get('encoded_ques_' + self.dtype))
            # ques_len_train
            self.data['ques_len_' + self.dtype] = np.asarray(f.get('ques_len_' + self.dtype))
            self.data['img_pos_' + self.dtype] = np.asarray(f.get('img_pos_' + self.dtype))

            if self.dtype == 'train':
                self.data['answers'] = np.asarray(f.get('encoded_ans'))

        print('question aligning')
        self.data['questions'] = self.right_align(self.data['questions'], self.data['ques_len_' + self.dtype])

        self.num_data_points[self.dtype] = self.data['questions'].shape[0]
        self.vocab_size = len(self.data_json['idx2word'])
        print ('Vocabulary size : {}'.format(self.vocab_size))

    def __len__(self):

        return len(self.data['questions'].shape[0])

    def __getitem__(self, idx):

        dtype = self.dtype
        item = {'index': idx}
        item['img'] = self.data_json['unique_img_' + dtype][self.data['img_pos_' + dtype][idx]]
        item['question'] = self.data['questions'][idx]
        item['answer'] = self.data['answers'][idx]

        return item
