import torchvision
import numpy as np
import torch
import torchvision.transforms as transforms
import os
import PIL
import json


class ImbalanceCIFAR10JSON(torch.utils.data.Dataset):
    def __init__(self, fname="lt.json", fname_syns=None, transform=None,
                 calc_ACC=False, calc_CAS=False, add_embed=False, xflip=False, random_seed=0):
        """
            Loads data given a path (root_dir) and preprocess them (transforms, blur)
        :param root_dir:
        :param transform:
        :param blur:
        """
        self.fname = fname
        self.sysn_fname = fname_syns
        self.xflip = xflip
        self.random_seed = random_seed
        self.transform = transform

        if not calc_CAS:

            print(f"\nLoading filenames from '{fname}' file...")
            self.data = self.read_from_json(fname, add_embed, embedding_val=1)

            self.class_dist = self.get_class_dist(self.data)
            self.classes = set(self.class_dist)
            self.cls_num = len(self.classes)
            self.img_max = self.class_dist[max(self.class_dist, key=self.class_dist.get)]
            cls_num_list = self.get_cls_num_list()

            print(f"original data cls num: {cls_num_list}")

            if calc_ACC:
                gap_num_per_cls = [int(self.img_max - i) for i in cls_num_list]

                print(f"\nLoading syns images from '{fname_syns}' to augment {fname}...")
                print(f"Syns data cls num: {gap_num_per_cls}")
                syns_data = self.read_from_json(fname_syns, add_embed, embedding_val=0, gap_num_per_cls=gap_num_per_cls)
                self.data.extend(syns_data)

        else:

            print(f"\nLoading syns images from '{fname_syns}' file...")
            self.data = self.read_from_json(fname_syns, add_embed, embedding_val=0)
            self.class_dist = self.get_class_dist(self.data)
            cls_num_list = self.get_cls_num_list()
            print(f"Syns data cls num: {cls_num_list}")

        if self.xflip:
            raise NotImplementedError

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        fname, *targets = self.data[idx]

        img_name = os.path.join(fname)
        image = PIL.Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return (image, *targets)

    def __len__(self):
        return len(self.data)

    @staticmethod
    def get_class_dist(data):
        """

        :param data:
        :return:
        """
        class_dist = dict()
        for d in data:
            if d[1] in class_dist:
                class_dist[d[1]] += 1
            else:
                class_dist[d[1]] = 1
        return class_dist

    def get_cls_num_list(self):
        cls_num_list = []
        for k, v in self.class_dist.items():
            cls_num_list.append(v)
        return cls_num_list

    def read_from_json(self, fname, add_embed, embedding_val, gap_num_per_cls=None):
        with open(fname) as f:
            data_ = json.load(f)

        if data_["labels"]:
            data = np.array(data_["labels"], dtype=object)
        else:
            raise RuntimeError

        if gap_num_per_cls:
            data = self.select_gap_imgs(data, gap_num_per_cls)

        dirname = os.path.dirname(fname)
        data = [[f'{dirname}/{x[0]}', x[1]] for x in data]

        if add_embed:
            data = [[*x, embedding_val] for x in data]

        return data

    def select_gap_imgs(self, data, gap_num_per_cls):
        selected_data = []
        for class_idx, count in zip(self.classes, gap_num_per_cls):
            idx = np.where(data[:, 1] == class_idx)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:count]
            selected_data.append(data[selec_idx, ...])
        return np.vstack(selected_data)
