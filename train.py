import torch
from torchvision.utils import save_image
import torch.nn.functional as F
import time
from models import losses
#from test import test
from utils import *
from tqdm import tqdm
from copy import deepcopy
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (8192, rlimit[1]))

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
                self.model[loss_type] = losses[loss_type](self.config)
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

            if (i+1) % self.config.valid_step == 0:
                self.valid(log)

            ### save log ###
            if (i+1) % (self.config.log_step) == 0:
                log['time_elapse'] = time.time() - start_time
                print_item = [list(set(['softmax', 'sphere', 'centre']).intersection(set(self.config.loss_type)))]
                save_log(log, self.config, print_item)


            if (i+1) % self.config.model_save_step == 0:
                ### save models ###
                save_model(self.model, self.config, log)
                print('saving models successfully!\n')

            ### update lr ###
            if (i+1) % self.config.model_save_step == 0 and (i+1) > (self.config.num_steps - self.config.num_steps_decay):
                lr = self.config.lr * (self.config.num_steps - i) / self.config.num_steps_decay
                for opt in self.opts.values():
                    for param in opt.param_groups:
                        param['lr'] = lr
                print('update learning rate to {:.6f}'.format(lr))


    def compute_feat(self, skts, phos):
        if not self.feats:
            self.feats = self.model['net'](torch.cat([skts, phos]))

    def valid(self, log):
        data = deepcopy(self.data)
        self.model['net'].eval()
        valid_skts, valid_cates = self.data.get_valid()
        cate_num = self.data.cate_num

        skts_feats = []
        valid_skts = valid_skts.split(self.config.batch_size*4)
        with torch.no_grad():
            for skts in valid_skts:
                skts_feats.append(self.model['net'](skts.to(self.config.device)))
        skts_feats = torch.cat(skts_feats)

        # get validation photos and categories
        data.set_phase('valid')
        print(len(data))
        data_loader = data.get_loader(batch_size=self.config.batch_size*2, num_workers=4, shuffle=True)
        phos_feats, phos_cates = [], []
        print('getting features of photos')
        for i,(phos, cs) in enumerate(data_loader):
        #for i in tqdm(range(len(data))):
        #    (phos, cs) = data[i]
            #if i > 100:
            #    break
            print('\r{}/{}'.format(i, len(data_loader)), end='')
            with torch.no_grad():
                phos_cates.append(cs)
                phos_feats.append(self.model['net'](phos.to(self.config.device)))
        phos_feats = torch.cat(phos_feats)
        phos_cates = torch.cat(phos_cates).to(self.config.device)

        # compute MAP and precision@200
        APs, P200 = [], []
        print('validating')
        for i in tqdm(range(len(skts_feats))):
            skt_feat = skts_feats[i].unsqueeze(0)
            c = valid_cates[i].item()
            # compute distance
            with torch.no_grad():
                if self.config.distance == 'sq':
                    dist = (phos_feats - skt_feat).pow(2).sum(dim=1)
                #dist = dist.cpu()
                res = phos_cates[dist.sort(dim=0)[1]] == c
                res = res.float()
            # compute p@200
            P200.append(res[:200].mean().item())

            # compute average precision
            k, rightk, precision = 0, 0, []
            while rightk < cate_num[c]:
                precision.append(res[:k+1].mean().item())
                k, rightk = k+1, rightk + res[k].item()
            APs.append(sum(precision)/len(precision))

        MAP = sum(APs) / len(APs)
        P200 = sum(P200) / len(P200)

        # record
        log['valid/MAP'] = MAP
        log['valid/P200'] = P200

        print('####### Validation ######')
        print('MAP:{:.4f} | P200:{:4f}'.format(MAP, P200))

        self.data.set_phase('train')
        self.model['net'].train()

    def test(self, log):
        raise NotImplementedError
        accu, accu_complex = test(self.model, self.data, self.config, self.config.verbose)
        log['test/top-1'] = accu['top-1']
        log['test/top-10'] = accu['top-10']
        self.config.best_res['top-1'] = max(self.config.best_res.get('top-1',0), accu['top-1'])
        self.config.best_res['top-10'] = max(self.config.best_res.get('top-10',0), accu['top-10'])
        log['test/multi-top-1'] = accu_complex['top-1']
        log['test/multi-top-10'] = accu_complex['top-10']
        self.config.best_res['multi-top-1'] = max(self.config.best_res.get('multi-top-1',0), accu_complex['top-1'])
        self.config.best_res['multi-top-10'] = max(self.config.best_res.get('multi-top-10',0), accu_complex['top-10'])


        print('#################### validation #################')
        print('#      |       simple       |      complex      #')
        print('# ---- |  top-1  |  top-10  |  top-1  |  top-10 #')
        print('# curr | {:.5f} | {:.5f}  | {:.5f} | {:.5f} #'.format(accu['top-1'], accu['top-10'], accu_complex['top-1'], accu_complex['top-10']))
        print('# best | {:.5f} | {:.5f}  | {:.5f} | {:.5f} #'.format(self.config.best_res['top-1'], self.config.best_res['top-10'],self.config.best_res['multi-top-1'], self.config.best_res['multi-top-10']))
        print('#################################################')






