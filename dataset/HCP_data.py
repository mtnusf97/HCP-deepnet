import os
import torch
from torch.utils.data import Dataset
import pickle


class HCPData(Dataset):
    def __init__(self, config):
        self.config = config
        self.idx2names_path = config.dataset.idx2names_path
        self.data_folder_path = config.dataset.data_folder_path
        with open(self.idx2names_path, 'rb') as f:
            idx2names = pickle.load(f)
        self.idx2names = idx2names
        if config.dataset.transform:
            self.transform = eval(config.dataset.transform)()
        if config.dataset.target_transform:
            self.target_transform = eval(config.dataset.target_transform)()

    def __len__(self):
        return len(self.idx2names)

    def __getitem__(self, idx):
        subject_path = os.path.join(self.data_folder_path, self.idx2names[idx])
        with open(subject_path, 'rb') as f:
            data = pickle.load(f)
        ts_data = data['data']
        label = data['target']
        if self.transform:
            ts_data = self.transform(ts_data)
        if self.target_transform:
            label = self.target_transform(label)
        return ts_data, label


class HCPTransform(object):

    def __call__(self, sample):
        sample = torch.Tensor(sample)
        sample = sample.float()
        sample = torch.transpose(sample, 0, 1)

        return sample


class ToTensor(object):

    def __call__(self, sample):
        sample = torch.tensor([sample])
        return sample.float()


# idx2names_path = '/home/matin/school/Amir_Omidvarnia/data_idx2name_test.pkl'
# data_folder_path = '/home/matin/school/Amir_Omidvarnia/gender_data'
