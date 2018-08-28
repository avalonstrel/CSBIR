import torch
from torchvision.utils import save_image
import torch.nn.functional as F
import time
from models import losses
#from test import test
from utils import *
from tqdm import tqdm
from copy import deepcopy
try:
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (8192*2, rlimit[1]))
except ModuleNotFoundError:
    pass
except ValueError:
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

class Solver:
    def __init__(self, config, data, model):
        print(config)
        self.config = config
        self.data = data
        self.model = model

        self.build_loss()
        self.build_opts()
        self.data_loader = self.data.get_loader(batch_size=config.batch_size, num_workers=max(config.batch_size//4,1), drop_last=True)

        self.config.best_res = {}


    def build_loss(self):
        # build model according to loss
        self.loss_ratio = {}

        # sphere loss: 这个好像没有什么特殊的操作
        for loss_type in ('sphere', 'centre', 'softmax', 'attribute'):
            if loss_type in self.config.loss_type:
                self.model[loss_type] = losses[loss_type](self.config, wordvec=self.data.wordvec)
                self.loss_ratio[loss_type] = self.config.loss_ratio[self.config.loss_type.index(loss_type)]

        print('Successfully build all losses')

    def build_opts(self):
        opts = dict()
        for key, net in self.model.items():
            opts[key] = torch.optim.Adam(net.parameters(), lr=self.config.lr, weight_decay=self.config.decay)
        self.opts = opts
        print('Successfully build all opts')


    def train(self):

        # load model / model to device
        for key in self.model.keys():
            self.model[key] = self.model[key].to(self.config.device)
        if self.config.pretrained_model > 0:
            load_model(self.model, self.config)
            print('loading models successfully!\n')

        # data_loader
        data_iter = iter(self.data_loader)

        # constants
        start_time = time.time()

        # log
        log = {}
        log['nsteps'] = self.config.num_steps


        # start training!
        self.feats = None
        for i in range(self.config.num_steps):
            log['step'] = i+1

            # get data
            (skts, phos, idxs), data_iter = get_data(data_iter, self.data_loader)
            skts, phos, idxs = skts.to(self.config.device), phos.to(self.config.device), idxs.to(self.config.device)
            idxs = torch.cat(idxs.split(1, dim=1)).view(-1)

            loss = 0
            for loss_type in ('sphere', 'softmax', 'centre'):
                if loss_type in self.config.loss_type:
                    self.compute_feat(skts, phos)
                  
                    loss_ = self.model[loss_type](self.feats, idxs)

                    log['loss/%s'%loss_type] = loss_.item()
                    loss += loss_ * self.loss_ratio[loss_type]

            for up_opt in self.opts.keys():
                self.opts[up_opt].zero_grad()
            loss.backward()
            for up_opt in self.opts.keys():
                self.opts[up_opt].step()

            self.feats = None

            ### save log ###
            if (i+1) % (self.config.log_step) == 0:
                log['time_elapse'] = time.time() - start_time
                print_item = [list(set(['softmax', 'sphere', 'centre']).intersection(set(self.config.loss_type)))]
                save_log(log, self.config, print_item)

            ### validation ###
            if (i+1) % self.config.valid_step == 0 and (i+1) // self.config.valid_step > 2:
                #self.valid(log)

            #if (i+1) % self.config.model_save_step == 0:
            #    ### save models ###
                save_model(self.model, self.config, log)
                print('saving models successfully!\n')
                self.test(log, 'valid')
                # save model before validating

            ### update lr ###
            if (i+1) % self.config.model_save_step == 0 and (i+1) > (self.config.num_steps - self.config.num_steps_decay):
                lr = self.config.lr * (self.config.num_steps - i) / self.config.num_steps_decay
                for opt in self.opts.values():
                    for param in opt.param_groups:
                        param['lr'] = lr
                print('update learning rate to {:.6f}'.format(lr))


    def compute_feat(self, skts, phos):
        if self.feats is None:
            self.feats = self.model['net'](torch.cat([skts, phos]))

    def test(self, log, phase='valid'):

        # prepare
        if phase == 'test':
            # load model / model to device
            for key in ('net',):
                self.model[key] = self.model[key].to(self.config.device)
            assert self.config.pretrained_model > 0
            load_model(self.model, self.config)
            print('Start Testing ...')
        else:
            print('Start Validation ...')

        # deal with data
        data = deepcopy(self.data)
        self.model['net'].eval()
        data.set_phase(phase)
        valid_skts, valid_cates = data.get_skts()
        cate_num = data.cate_num
    
        ######### compute sketch features #########
        skts_feats = []
        valid_skts = valid_skts.split(self.config.batch_size*4)
        with torch.no_grad():
            for skts in valid_skts:
                if phase == 'valid':
                    feat = self.model['net'](skts.to(self.config.device))
                else:
                    skts = skts.to(self.config.device)
                    skts = torch.cat([skts, shuffle(skts, dim=2, inv=True)])
                    feat = torch.stack(self.model['net'](skts).split(skts.size(0)//2), dim=2)
                skts_feats.append(feat)
        skts_feats = torch.cat(skts_feats)


        ######### compute photo features #########
        # get validation photos and categories
        data_loader = data.get_loader(batch_size=self.config.batch_size*2, num_workers=4, shuffle=True)
        phos_feats, phos_cates = [], []
        for i,(phos, cs) in enumerate(data_loader):
            print('\rgetting features of photos: [{}/{}]'.format(i, len(data_loader)), end='')
            with torch.no_grad():
                phos_cates.append(cs)
                if phase == 'valid':
                    feat = self.model['net'](phos.to(self.config.device))
                else:
                    phos = phos.to(self.config.device)
                    phos = torch.cat([phos, shuffle(phos, dim=2, inv=True)])
                    feat = torch.stack(self.model['net'](phos).split(phos.size(0)//2), dim=2)
                phos_feats.append(feat)
        phos_feats = torch.cat(phos_feats)
        phos_cates = torch.cat(phos_cates).to(self.config.device)

        ######### compute metric (MAP, precision@200) #########
        APs, P200 = [], []
        print('\nvalidating', end='')
        for i in tqdm(range(len(skts_feats))):
            skt_feat = skts_feats[i].unsqueeze(0)
            c = valid_cates[i].item()
            # compute distance
            with torch.no_grad():
                if self.config.distance == 'sq':
                    if phase == 'valid':
                        dist = (phos_feats - skt_feat).pow(2).sum(dim=1)
                    else:
                        dist1 = (phos_feats - skt_feat[:,:,:1]).pow(2).sum(dim=1).cpu()
                        torch.cuda.empty_cache()
                        dist2 = (phos_feats - skt_feat[:,:,1:]).pow(2).sum(dim=1).cpu()
                        torch.cuda.empty_cache()
                        dist = torch.cat([dist1, dist2], dim=1).to(self.config.device)
                        if self.config.test_dist == 'mean':
                            dist = dist.mean(dim=1)
                        elif self.config.test_dist == 'min':
                            dist = dist.min(dim=1)[0]
                res = phos_cates[dist.sort(dim=0)[1]] == c
                res = res.float()

            # compute p@200
            if self.config.mode == 'std':
                P200.append(res[:200].mean().item())
            elif self.config.mode == 'zeroshot':
                P200.append(res[:100].mean().item())

            # compute average precision
            k, rightk, precision = 0, 0, []
            while rightk < cate_num[c]:
                r = res[k].item()
                if r:
                    precision.append(res[:k+1].mean().item())
                    rightk += 1
                k += 1
            APs.append(sum(precision)/len(precision))

        # final result
        MAP = sum(APs) / len(APs)
        P200 = sum(P200) / len(P200)

        # record
        if phase == 'valid':
            save_valid_test_result(self.config, log['step'], MAP, P200, phase)
        elif phase == 'test':
            save_valid_test_result(self.config, int(self.config.pretrained_model), MAP, P200, phase)

        print('result: MAP:{:.4f} | P200:{:4f}'.format(MAP, P200))

        # recover state
        self.data.set_phase('train')
        self.model['net'].train()

        return MAP, P200


