from utils import meters
from models.Q_modules.Q_core import *
import wandb

class BasicLogger:
    '''
    Training Log
    '''
    def __init__(self) -> None:
        self.batch_time = meters.AverageMeter()
        self.data_time = meters.AverageMeter()
        self.losses = meters.AverageMeter()
        self.top1 = meters.AverageMeter()
        self.top5 = meters.AverageMeter()
        self.best_prec = 0

    def reset(self):
        self.batch_time.reset()
        self.data_time.reset()
        self.losses.reset()
        self.top1.reset()
        self.top5.reset()

    def update(self):
        self.best_prec = max(self.best_prec, self.top1.avg)

    def log(self, batch, logfile=None):
        print("Batch:{} Time:{:.3f}({:.3f})\tDataTime:{:.3f}({:.3f})\tloss:{:.4f}({:.4f})\tprec@1:{:.4f}({:.4f})\tprec@5:{:.4f}({:.4f})".format(batch, 
                                  self.batch_time.val, self.batch_time.avg,
                                  self.data_time.val, self.data_time.avg,
                                  self.losses.val, self.losses.avg, 
                                  self.top1.val, self.top1.avg,
                                  self.top5.val, self.top5.avg))

class WandbLogger:
    '''
    WandbLog
    '''
    def __init__(self):
        self.cur_epoch = 0
        self.train_logger = None
        self.basic = {}
        self.bw = {
            'A':None,
            'W':None,
            'G':None,
            'bA':None
        }
        self.sparsity = {
            'A':None,
            'W':None,
            'G':None,
            'bA':None,
        }

    def update(self, epoch, q_params_list, train=True):
        self.cur_epoch = epoch
        # update mean bw
        if train:
            for datatype in ['A', 'W', 'G', 'bA']:
                total_mean_bw, layer_mean_bws = mean_bwmap(q_params_list, datatype=datatype, bwmaptype='int_bwmap')
                section = datatype + '_bw/'
                self.bw[datatype] = {section + 'mean_bw':total_mean_bw}
                cur_layer = 0
                for layer_mean_bw in layer_mean_bws:
                    self.bw[datatype].update({section+'layer'+str(cur_layer).zfill(2)+'_bw':layer_mean_bw})
                    cur_layer += 1
            # update mean sparsity
            for datatype in ['A', 'W', 'G', 'bA']:
                total_mean_sparsity, layer_mean_sparsities = mean_sparsity(q_params_list, datatype=datatype)
                section = datatype + '_sparsity/'
                self.sparsity[datatype] = {section + 'mean_sparsity':total_mean_sparsity}
                cur_layer = 0
                for layer_mean_sparsity in layer_mean_sparsities:
                    self.sparsity[datatype].update({section+'layer'+str(cur_layer).zfill(2)+'_sparsity':layer_mean_sparsity})
                    cur_layer += 1

        prefix = train and 'train_' or 'val_'

        self.basic.update({prefix+'loss':self.train_logger.losses.avg})
        self.basic.update({prefix+'prec1':self.train_logger.top1.avg})
        self.basic.update({prefix+'prec5':self.train_logger.top5.avg})
        self.basic.update({'best_prec':self.train_logger.best_prec})

    def log(self):
        for datatype in ['W', 'bA']:
            print(self.bw[datatype])
            print(self.sparsity[datatype])
            wandb.log(self.bw[datatype], step=self.cur_epoch)
            wandb.log(self.sparsity[datatype], step=self.cur_epoch)
            wandb.log(self.basic, step=self.cur_epoch)
