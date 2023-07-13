import os
import sys
import shutil
import numpy as np
from utils import utils
from tensorboardX import SummaryWriter
import torch
from datetime import datetime


class Logger(object):
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS
        time = datetime.now().strftime('%m%d-%H%M')
        # out_dir = os.path.join('..', 'R' + time)
        out_dir = os.path.join('..', 'T' + time + '-' + FLAGS.savename)

        self.img_out_dir = os.path.join(out_dir, 'Result')
        self.model_dir = os.path.join(out_dir, 'Model')
        self.log_dir = os.path.join(out_dir, 'Log')
        utils.checkpath(self.img_out_dir)
        utils.checkpath(self.model_dir)
        utils.checkpath(self.log_dir)
        shutil.copy('./utils/bin/plotboard.py', self.log_dir)
        shutil.copy('./utils/bin/run.bat', self.log_dir)

        self.writer = SummaryWriter(logdir=self.log_dir)

    def write_tensorboard(self, mode, logs, epoch):
        for key in logs:
            tag = mode + '/' + key
            self.writer.add_scalar(tag, logs[key], epoch)
        # auto flush the values
        for it in self.writer.all_writers.values():
            it.flush()
    


    def save_model(self, model, modelname, mode='model'):
        """
        mode: model, weights
        """
        if mode == 'model':
            torch.save(
                model,
                os.path.join(self.model_dir, 'Model_{}.pkl'.format(modelname)),
            )
        else:
            torch.save(
                model.state_dict(),
                os.path.join(self.model_dir, 'Weights_{}.pkl'.format(modelname)),
            )

        return None

    def load_weights(self, model, weightsfile):
        model.load_state_dict(torch.load(weightsfile))

        return None

    def load_model(self, modelfile):
        model = torch.load(modelfile)

        return model
