import torch
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import random

class SketchyData(Dataset):
    def __init__(self, root_path, crop_size=225, mode='std', **kwargs):

        self.photo_root = os.path.join(root_path, '256x256', 'photo', 'tx_000100000000')
        self.sketch_root = os.path.join(root_path, '256x256', 'sketch', 'tx_000100000000')
        self.crop_size = crop_size
        self.seed = kwargs.get('seed', None)

        # get invalid sketches
        info_root = os.path.join(root_path, 'info')
        invalid_files = [os.path.join(info_root, f) for f in os.listdir(info_root) if f.startswith('invalid')]

        for file in invalid_files:
            invalid_skts = []
            with open(file, 'r') as f:
                invalid_skts += f.read().split('\n')
        self.invalid_skts = set(map(lambda x:x+'.png', invalid_skts))

        # load train/test cates
        with open(os.path.join(root_path, 'cate_sep.txt'), 'r') as f:
            cates_raw = list(filter(lambda x: x and (not x.startswith('.')), f.read().split('\n')))

        cates = {}
        self.cate2idx = {}
        for i,c in enumerate(cates_raw):
            m, c = c.split('/')
            cates[m] = cates.get(m, [])+ [c]
            self.cate2idx[c] = i
        self.num_cates = len(cates_raw)

		# rearrange cate according to mode
        assert mode in ('std', 'fewshot-train', 'fewshot-finetune', 'zeroshot-train')
        self.mode = mode
        if mode == 'std':
            cates['source'] += cates.get('target', [])
            cates['target'] = cates['source']

        # prepare files
        self.build(cates)
        self.set_phase('train')

        # transforms
        # note that the sizes of raw photo/sketch are 227/256
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize((crop_size, crop_size))
        self.randomflip = transforms.RandomHorizontalFlip()
        self.resizedcrop = transforms.RandomResizedCrop(size=crop_size, scale=(0.8, 1))
        self.randomcrop = transforms.RandomCrop(crop_size)
        self.centercrop = transforms.CenterCrop(crop_size)

    def build(self, cates):
        self.cate_num = {}
        print('Building dataset ...')
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
            flst = filter(lambda x: not ((c+'/'+x) in self.invalid_skts), flst)
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
        else:
            raise NotImplementedError('Currently, only standard mode is supported.')

        self.train_phos = []
        self.train_skts = []
        for c, clst in self.source_train_phos.items():
            idx = self.cate2idx[c]
            for f in clst:
                self.train_phos.append((f, idx))
        for c, clst in self.source_train_skts.items():
            idx = self.cate2idx[c]
            for f in clst:
                self.train_skts.append((f, idx))
        random.seed(None)
        random.shuffle(self.train_phos)
        random.shuffle(self.train_skts)

        self.valid_skts = []
        for c, clst in self.source_valid_skts.items():
            idx = self.cate2idx[c]
            for f in clst:
                self.valid_skts.append((f, idx))
        random.shuffle(self.valid_skts)        

        self.test_skts = []
        for c, clst in self.target_test_skts.items():
            idx = self.cate2idx[c]
            for f in clst:
                self.test_skts.append((f, idx))    
        random.shuffle(self.test_skts)    

        print('Finish building dataset!')

    def __len__(self):
        return len(self.train_phos)

    def set_phase(self, phase='train'):
        self.phase = phase

    def __getitem__(self, index):
        (fpho, cpho) = self.train_phos[index]
        pho = self.to_tensor(self.randomflip(self.resize(Image.open(fpho))))
        pho = pho.expand(3, self.crop_size, self.crop_size)

        if self.phase == 'train':
            (fskt, cskt) = random.choice(self.train_skts)
            skt = self.to_tensor(self.randomflip(self.randomcrop(Image.open(fskt)))).expand(3, self.crop_size, self.crop_size)
        else:
            return pho, torch.LongTensor([cpho])

        cates = torch.LongTensor([cskt, cpho])
        return skt, pho, cates

    def get_valid(self):
        # use __getitem__ to get photos?
        if not hasattr(self, 'validskt'):
            skts, cs = [], []
            for fskt, c in self.valid_skts:
                skts.append(self.to_tensor(self.centercrop(Image.open(fskt))))
                cs.append(c)

            skts = torch.stack(skts)
            skts = skts.expand(skts.size(0),3,skts.size(2),skts.size(3))
            cs = torch.LongTensor(cs)
            self.validskt = (skts, cs)
        return self.validskt

    def get_test(self):
        # use __getitem__ to get photos?
        if not hasattr(self, 'testskt'):
            skts, cs = [], []
            for fskt, c in self.test_skts:
                skts.append(self.to_tensor(self.centercrop(Image.open(fskt))))
                cs.append(c)

            skts = torch.stack(skts).repeat(1,3,1,1)
            cs = torch.LongTensor(cs)
            self.validskt = (skts, cs)
        return self.validskt

    def get_loader(self, **kwargs):
        return DataLoader(self, **kwargs)


if __name__ == '__main__':
    exit(0)
    data_root = "D:\\datasets\\TU-Berlin"
    data = TUBerlinData(data_root, seed=1)
    print(data.source_valid_skts)



