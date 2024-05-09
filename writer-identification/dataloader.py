import os
import pickle
import numpy as np
from scipy import misc
import torch.utils.data as data
import torch
from torchvision.transforms import Compose, ToTensor, Lambda, Resize, Grayscale, GaussianBlur, ColorJitter
from torchvision import transforms
import random
import cv2
import lmdb

from data_augmenter import transform


class DatasetFromLMDB(data.Dataset):
    def __init__(self, lmdb_path, labels_path, augment=False):
        super(DatasetFromLMDB, self).__init__()

        self.augment = augment
        self.lmdb_path = lmdb_path
        self.labels_path = labels_path
        self.savename = "writer_save"

        self.lmdb_txn = lmdb.open(lmdb_path, readonly=True).begin()

        self.image_to_writer = {}
        with open(labels_path, encoding='utf-8') as f:
            self.labels = np.array([line.rstrip().split(" ", 2) for line in f])

        for label in self.labels:
            self.image_to_writer[label[0]] = label[1]

        self.idlist = self._get_all_identity()
        self.idx_tab = self._convert_identity2index(self.savename)
        self.num_writer = len(self.idx_tab)

        # convert to idx for neural network
    def _convert_identity2index(self, savename):
        if os.path.exists(savename):
            with open(savename,'rb') as fp:
                    identity_idx = pickle.load(fp)
        else:
            #'''
            identity_idx = {}
            for idx,ids in enumerate(self.idlist):
                    identity_idx[ids] = idx
            
            with open(savename,'wb') as fp:
                    pickle.dump(identity_idx,fp)
            #'''
                
        return identity_idx
                        
    # get all writer identity
    def _get_all_identity(self):
        writer_list = []
        for label in self.labels:
            img = label[0]
            writerId = self._get_identity(img)
            writer_list.append(writerId)
        writer_list=list(set(writer_list))
        return writer_list
    
    def _get_identity(self,fname):
        return self.image_to_writer[fname]
        
    def augment_transform(self):
        return Compose([
            ToTensor(),
            Lambda(lambda x: (torch.tile(x, (3,1)))),
            # Lambda(lambda x: transform(x)),
            GaussianBlur(3, sigma=(0.1, 2.0)),
            ColorJitter(brightness=0.2, contrast=0.3, saturation=0.3, hue=0.1),
            Grayscale(3),
            Resize((224, 224)),
        ])

    def not_augment_transform(self):
        return Compose([
            ToTensor(),
            Lambda(lambda x: (torch.tile(x, (3,1)))),
            Grayscale(3),
            Resize((224, 224)),
        ])


    def __getitem__(self, index):
        image_name = self.labels[index][0]

        image = np.frombuffer(self.lmdb_txn.get(image_name.encode()), dtype=np.uint8)
        image = cv2.imdecode(image, 1)
        if self.augment:
            image = self.augment_transform()(image)
        else:
            image = self.not_augment_transform()(image)
    
        writer = self.idx_tab[self._get_identity(image_name)]

        return image, writer

    def __len__(self):
        return len(self.labels)
