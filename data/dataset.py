import json
import os
from math import ceil

from torch.utils.data import Dataset
import torch
import torch.nn as nn

from .vocab import Vocab

class LogDataset(Dataset):
    def __init__(self, data_dir, window_size, mode = 'train'):
        self.data_dir = data_dir
        self.mode = mode
        self.window_size = window_size

        if mode == 'train':
            self.data = json.load(open(os.path.join(data_dir, 'train.json')))
        else:
            test_normal = json.load(open(os.path.join(data_dir, 'test_normal.json')))
            test_abnormal = json.load(open(os.path.join(data_dir, 'test_abnormal.json')))
            self.data = test_normal + test_abnormal
            self.data = sorted(self.data, key = lambda x: x['Timestamp'][0])

        self.length = len(self.data)

        if self.mode == 'train':
            print('Splitting data into windows...')
            self.data = self.window()

        if self.mode == 'train':
            self.vocab = Vocab([sample['EventTemplate'] for sample in self.data])
            Vocab.dump_vocab(os.path.join(data_dir, 'vocab.pkl'), self.vocab)
        
        else:
            if os.path.exists(os.path.join(data_dir, 'vocab.pkl')):
                self.vocab = Vocab.load_vocab(os.path.join(data_dir, 'vocab.pkl'))
            else:
                raise FileNotFoundError('Vocab file not found')

       
    
    def window(self):
        new_data = []
        for sample in self.data:
            for i in range(max(len(sample['EventTemplate']) - self.window_size, 0) + 1):
                if len(sample['EventTemplate'][i : i + self.window_size]) < 2:
                    continue
                
                new_data.append({
                    'BlockId': sample['BlockId'],
                    'EventTemplate': sample['EventTemplate'][i : i + self.window_size]
                })
               
        
        return new_data


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sample = self.data[index]
        event_ids = [self.vocab.word_to_id(word) for word in sample['EventTemplate']]
        if self.mode == 'train':
            return torch.tensor(event_ids)
        
        else:
            return torch.tensor(event_ids), max(sample['Label']), sample['BlockId']
    
    def collate_fn(self, batch):
        if self.mode == 'train':
            event_sequences = [item[:-1] for item in batch]
            next_event_ids = [item[-1] for item in batch]
            event_sequences = nn.utils.rnn.pad_sequence(event_sequences, batch_first=True, padding_value = self.vocab.word_to_id('<pad>'))
            next_event_ids = torch.tensor(next_event_ids)
            
            return event_sequences, next_event_ids
        
       
        
    

