import torch
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import random

class TUBerlinData(Dataset):
    def __init__(self, root_path, crop_size=225, mode='std', **kwargs):

        self.photo_root = os.path.join(root_path, 'photos')
        self.sketch_root = os.path.join(root_path, 'sketches')
        self.crop_size = crop_size
        self.seed = kwargs.get('seed', None)

        # load train/test cates
        with open(os.path.join(root_path, 'cate_sep.txt'), 'r') as f:
            cates_raw = list(filter(lambda x: x and (not x.startswith('.')), f.read().split('\n')))
        # load wordvec
        self.wordvec = torch.from_numpy(np.load(os.path.join(root_path, 'wordvec.npy'))).float() # shape: 250x100

        cates = {}
        self.cate2idx = {}
        for i,c in enumerate(cates_raw):
            m, c = c.split('/')
            cates[m] = cates.get(m, [])+ [c]
            self.cate2idx[c] = i        

		# rearrange cate according to mode
        assert mode in ('std', 'fewshot-train', 'fewshot-finetune', 'zeroshot')
        self.mode = mode
        if mode == 'std':
            cates['source'] += cates.get('target', [])
            cates['target'] = cates['source']
        elif mode == 'zeroshot':
            # 在家不方便大改。。凑合一下
            valid_test_cates = [p for p in cates['source'] if len(os.listdir(os.path.join(self.photo_root, p)))> 400]
            random.seed(self.seed)
            test_cates = random.sample(valid_test_cates, 30)
            cates['target'] = test_cates
            cates['source'] = list(set(cates['source']).difference(set(test_cates)))
        # print(cates)
        self.num_train_cates = len(cates['source'])

        # prepare files
        self.build(cates)
        self.set_phase('train')

        # transforms
        # note that the sizes of raw photo/sketch are 227/256
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize(crop_size)
        self.randomflip = transforms.RandomHorizontalFlip()
        self.resizedcrop = transforms.RandomResizedCrop(size=crop_size, scale=(0.8, 1))
        self.randomcrop = transforms.RandomCrop(crop_size)
        self.centercrop = transforms.CenterCrop(crop_size)

    def build(self, cates):
        self.cate_num = {}
        print('Building dataset ...', end='\r')
        self.source_train_phos = {}
        self.source_train_skts = {}
        self.source_valid_skts = {}

        self.target_train_phos = {}
        self.target_train_skts = {}
        self.target_test_skts = {}

        phos, skts = {}, {}
        for c in set(cates['source'] + cates['target']):
            flst = filter(lambda x:x and (not x.startswith('.')), os.listdir(os.path.join(self.photo_root, c)))
            phos[c] = list(map(lambda x: os.path.join(self.photo_root, c, x), flst))
            flst = filter(lambda x:x and (not x.startswith('.')), os.listdir(os.path.join(self.sketch_root, c)))
            skts[c] = list(map(lambda x: os.path.join(self.sketch_root, c, x), flst))

        if self.mode == 'std':
            for c in cates['source']:
                self.source_train_phos[c] = phos[c]
                self.cate_num[self.cate2idx[c]] = len(phos[c])
                random.seed(self.seed)
                random.shuffle(skts[c])
                self.source_train_skts[c] = skts[c][:-11]
                self.source_valid_skts[c] = skts[c][-11:-10]
                self.target_test_skts[c] = skts[c][-10:]

        elif self.mode == 'zeroshot':
            for c in cates['source']:
                self.source_train_phos[c] = phos[c]
                self.cate_num[self.cate2idx[c]] = len(phos[c])
                random.shuffle(skts[c])
                self.source_train_skts[c] = skts[c][:-1]
                self.source_valid_skts[c] = skts[c][-1:]

            for c in cates['target']:
                self.target_train_phos[c] = phos[c]
                self.cate_num[self.cate2idx[c]] = len(phos[c])
                self.target_test_skts[c] = skts[c]

        elif self.mode.startswith('fewshot'):
            shot = eval(self.mode.split('_')[1])
            for c in cates['source']:
                self.source_train_phos[c] = phos[c]
                self.cate_num[self.cate2idx[c]] = len(phos[c])
                random.shuffle(skts[c])
                self.source_train_skts[c] = skts[c][:-1]
                self.source_valid_skts[c] = skts[c][-1:]

            for c in cates['target']:
                self.target_train_phos[c] = phos[c]
                self.cate_num[self.cate2idx[c]] = len(phos[c])
                self.target_train_skts[c] = skts[c][:-shot]
                self.target_test_skts[c] = skts[c][-shot:]
        else:
            raise ValueError("mode should be 'std' or 'fewshot_x'(x is the number of shots) or 'zeroshot'")

        random.seed(None)
        print('Finish building dataset!')

    def __len__(self):
        return len(self.phos)

    def set_phase(self, phase='train'):
        if hasattr(self, 'phase') and phase==self.phase:
            return
        self.phase = phase

        if self.mode == 'std':
            assert phase in ('train', 'valid', 'test'), "phase must be train/valid/test"
            self.phos = []
            for c, clst in self.source_train_phos.items():
                idx = self.cate2idx[c]
                for f in clst:
                    self.phos.append((f, idx))
            random.shuffle(self.phos)

            self.skts = []
            for c, clst in getattr(self, 'source_%s_skts'%phase).items():
                idx = self.cate2idx[c]
                for f in clst:
                    self.skts.append((f, idx))
            random.shuffle(self.skts)

        elif self.mode == 'zeroshot':
            assert phase in ('train', 'valid', 'test'), "phase must be train/valid/test"
            flag = 'target' if phase == 'test' else 'source'

            self.phos = []
            for c, clst in getattr(self, '%s_train_phos' % flag).items():
                idx = self.cate2idx[c]
                for f in clst:
                    self.phos.append((f, idx))
            random.shuffle(self.phos)

            self.skts = []
            for c, clst in getattr(self, '%s_%s_skts' %(flag, phase)).items():
                idx = self.cate2idx[c]
                for f in clst:
                    self.skts.append((f, idx))
            random.shuffle(self.skts)

        elif self.mode.startswith('fewshot'):
            raise NotImplementedError

        else:
            raise ValueError("mode should be 'std' or 'fewshot_x'(x is the number of shots) or 'zeroshot'")
        print("Phase of dataset is set to '%s'."%phase)


    def __getitem__(self, index):
        (fpho, cpho) = self.phos[index]
        pho = self.to_tensor(self.randomflip(self.resize(Image.open(fpho))))

        if self.phase == 'train':
            (fskt, cskt) = random.choice(self.skts)
            skt = self.to_tensor(self.randomflip(self.randomcrop(Image.open(fskt)))).expand(3, self.crop_size, self.crop_size)
        else:
            return pho, torch.LongTensor([cpho])

        cates = torch.LongTensor([cskt, cpho])
        return skt, pho, cates

    def get_skts(self):
        # use __getitem__ to get photos?
        if not hasattr(self, '%sskt'%self.phase):
            skts, cs = [], []
            for fskt, c in self.skts:
                skts.append(self.to_tensor(self.centercrop(Image.open(fskt))))
                cs.append(c)

            skts = torch.stack(skts).repeat(1,3,1,1)
            cs = torch.LongTensor(cs)
            if self.phase == 'valid':
                self.validskt = (skts, cs)
            elif self.phase == 'test':
                self.testskt = (skts, cs)

        return getattr(self, '%sskt'%self.phase)

    def get_loader(self, **kwargs):
        return DataLoader(self, **kwargs)


if __name__ == '__main__':
    data_root = "D:\\datasets\\TU-Berlin"
    data = TUBerlinData(data_root, seed=1)
    print(data.source_valid_skts)

