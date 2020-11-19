import os
import torch.utils.data as data
import numpy as np
from torchvision import transforms as T
from data_objects.transforms import Normalize, generate_test_sequence
import pandas as pd
from pathlib import Path


def get_eval_paths():
    data_dir = Path("/media/mdt/WD/zalo/speech/data/Train-Test-Data/feature/public-test/test")
    path_list = data_dir.glob("*.npy")
    path_list = [i for i in path_list]
    return path_list


def get_test_paths(pairs_path, db_dir):
    def convert_folder_name(path):
        basename = os.path.splitext(path)[0]
        items = basename.split('/')
        speaker_dir = items[0]
        fname = '{}_{}.npy'.format(items[1], items[2])
        p = os.path.join(speaker_dir, fname)
        return p

    pairs = [line.strip().split() for line in open(pairs_path, 'r').readlines()]
    nrof_skipped_pairs = 0
    path_list = []
    issame_list = []

    for pair in pairs:
        if pair[0] == '1':
            issame = True
        else:
            issame = False

        path0 = db_dir.joinpath(pair[1][:-4] + ".npy")
        path1 = db_dir.joinpath(pair[2][:-4] + ".npy")

        if os.path.exists(path0) and os.path.exists(path1):    # Only add the pair if both paths exist
            path_list.append((path0,path1,issame))
            issame_list.append(issame)
        else:
            nrof_skipped_pairs += 1
    if nrof_skipped_pairs>0:
        print('Skipped %d image pairs' % nrof_skipped_pairs)

    return path_list


class VoxcelebTestset(data.Dataset):
    def __init__(self, data_dir, partial_n_frames):
        super(VoxcelebTestset, self).__init__()
        self.data_dir = data_dir
        self.root = data_dir.joinpath('feature', 'test')
        # self.test_pair_txt_fpath = data_dir.joinpath('veri_test.txt')
        # self.test_pairs = get_test_paths(self.test_pair_txt_fpath, self.root)
        self.test_paths = list(self.root.glob("**/*.npy"))
        self.partial_n_frames = partial_n_frames
        mean = np.load(self.data_dir.joinpath('mean.npy'))
        std = np.load(self.data_dir.joinpath('std.npy'))
        self.transform = T.Compose([
            Normalize(mean, std)
        ])

    def load_feature(self, feature_path):
        feature = np.load(feature_path)
        test_sequence = generate_test_sequence(feature, self.partial_n_frames)
        return test_sequence

    def __getitem__(self, index):
        path_1 = self.test_paths[index]

        feature1 = self.load_feature(path_1)
        # feature2 = self.load_feature(path_2)

        if self.transform is not None:
            feature1 = self.transform(feature1)
            # feature2 = self.transform(feature2)
        return feature1, str(path_1)

    def __len__(self):
        return len(self.test_paths)



class VoxcelebTestsetZalo(data.Dataset):
    def __init__(self, data_dir, partial_n_frames):
        super(VoxcelebTestsetZalo, self).__init__()
        self.data_dir = data_dir
        self.root = data_dir.joinpath('feature', 'test')
        # self.test_pair_txt_fpath = data_dir.joinpath('veri_test.txt')
        # self.test_pairs = get_test_paths(self.test_pair_txt_fpath, self.root)
        self.test_pairs = get_eval_paths()
        self.partial_n_frames = partial_n_frames
        mean = np.load(self.data_dir.joinpath('mean.npy'))
        std = np.load(self.data_dir.joinpath('std.npy'))
        self.transform = T.Compose([
            Normalize(mean, std)
        ])

    def load_feature(self, feature_path):
        feature = np.load(feature_path)
        test_sequence = generate_test_sequence(feature, self.partial_n_frames)
        return test_sequence

    def __getitem__(self, index):
        path_1 = self.test_pairs[index]

        feature1 = self.load_feature(path_1)
        # feature2 = self.load_feature(path_2)

        if self.transform is not None:
            feature1 = self.transform(feature1)
            # feature2 = self.transform(feature2)
        return feature1, str(path_1)

    def __len__(self):
        return len(self.test_pairs)