from __future__ import (division, print_function)
import numpy as np
from collections import defaultdict
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
from tensorboardX import SummaryWriter

from model import *
from dataset import *
from utils.logger import get_logger
from utils.train_helper import snapshot, load_model
from utils.data_parallel import DataParallel
from torch.utils.data import SubsetRandomSampler
from sklearn.metrics import classification_report, accuracy_score

logger = get_logger('exp_logger')
__all__ = ['LiangweiRunner']

NPR = np.random.RandomState(seed=1234)


class LiangweiRunner(object):

    def __init__(self, config):
        self.config = config
        self.seed = config.seed
        self.dataset_conf = config.dataset
        self.model_conf = config.model
        self.train_conf = config.train
        self.test_conf = config.test
        self.use_gpu = config.use_gpu
        self.gpus = config.gpus
        self.device = config.device
        self.writer = SummaryWriter(config.save_dir)
        self.num_gpus = len(self.gpus)
        self.is_shuffle = False

        if self.train_conf.is_resume:
            self.config.save_dir = self.train_conf.resume_dir

    def train(self):
        # create data loader
        train_dataset = eval(self.dataset_conf.loader_name)(self.config)
        num_train = len(train_dataset)
        indices = list(range(num_train))
        np.random.shuffle(indices)
        split = int(np.floor(num_train * self.dataset_conf.train_ratio))
        train_index, val_index = indices[:split], indices[split:]

        train_sampler, val_sampler = SubsetRandomSampler(train_index), SubsetRandomSampler(val_index)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.train_conf.batch_size,
            num_workers=self.train_conf.num_workers,
            drop_last=False,
            sampler=train_sampler)

        val_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.train_conf.batch_size,
            num_workers=self.train_conf.num_workers,
            drop_last=False,
            sampler=val_sampler)

        # create models
        model = eval(self.model_conf.name)(self.config)

        if self.use_gpu:
            # model = DataParallel(model, device_ids=self.gpus).to(self.device)
            model = model.to(self.device)

        # create optimizer
        params = filter(lambda p: p.requires_grad, model.parameters())
        if self.train_conf.optimizer == 'SGD':
            optimizer = optim.SGD(
                params,
                lr=self.train_conf.lr,
                momentum=self.train_conf.momentum,
                weight_decay=self.train_conf.wd)
        elif self.train_conf.optimizer == 'Adam':
            optimizer = optim.Adam(params, lr=self.train_conf.lr, weight_decay=self.train_conf.wd)
        else:
            raise ValueError("Non-supported optimizer!")

        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=self.train_conf.lr_decay_epoch,
            gamma=self.train_conf.lr_decay)

        # reset gradient
        optimizer.zero_grad()

        # resume training
        resume_epoch = 0
        if self.train_conf.is_resume:
            model_file = os.path.join(self.train_conf.resume_dir,
                                      self.train_conf.resume_model)
            load_model(
                model.module if self.use_gpu else model,
                model_file,
                self.device,
                optimizer=optimizer,
                scheduler=lr_scheduler)
            resume_epoch = self.train_conf.resume_epoch

        # Training Loop
        minimum_val_loss = np.inf
        total_iteration = 0
        for epoch in range(resume_epoch, self.train_conf.max_epoch):
            train_loss = 0
            model.train()
            lr_scheduler.step()
            optimizer.zero_grad()

            for iteration, (data, label) in enumerate(train_loader):
                total_iteration += 1
                optimizer.zero_grad()
                output = model(data.to(self.device))
                criterion = eval(self.train_conf.criterion)
                loss = criterion(output, label.to(self.device))
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * data.size()[0]

                self.writer.add_scalar(self.dataset_conf.name + '_training_loss', loss, total_iteration)

                if iteration % self.train_conf.display_iter == 0:
                    logger.info(
                        "Training Loss @ epoch {:04d} iteration {:08d} = {} ###".format(epoch + 1, iteration, loss.item()))

            is_training = model.training
            val_loss = 0
            model.eval()

            for data, label in val_loader:
                output = model(data.to(self.device))
                loss = criterion(output, label.to(self.device))
                val_loss += loss.item() * data.size()[0]

            val_loss /= len(val_index)
            logger.info(
                "Validation Loss @ epoch {:04d} = {}".format(epoch + 1, val_loss))
            self.writer.add_scalar(self.dataset_conf.name + '_validation_loss', val_loss, epoch + 1)

            model.train(is_training)

            avg_train_loss = train_loss / len(train_index)
            logger.info("Total avg Loss @ the end of epoch {:04d} = {} ***".format(epoch + 1, avg_train_loss))
            self.writer.add_scalar(self.dataset_conf.name + '_avg_training_loss', avg_train_loss, epoch + 1)

            # snapshot model
            if (epoch + 1) % self.train_conf.snapshot_epoch == 0:
                if val_loss < minimum_val_loss:
                    minimum_val_loss = val_loss
                    logger.info("Saving Snapshot @ epoch {:04d}".format(epoch + 1))
                    # snapshot(model.module if self.use_gpu else model, optimizer, self.config, epoch + 1,
                    #          scheduler=lr_scheduler, tag=self.dataset_conf.name)
                    snapshot(model, optimizer, self.config, epoch + 1,
                             scheduler=lr_scheduler, tag=self.dataset_conf.name)
                else:
                    logger.info("No need to save @ epoch {:04d} !!!".format(epoch + 1))

        self.writer.close()

        return 1

    def test(self):
        self.config.save_dir = self.test_conf.test_model_dir
        self.writer = SummaryWriter(self.config.save_dir)

        # load model
        model = eval(self.model_conf.name)(self.config)
        model_file = os.path.join(self.config.save_dir, self.test_conf.test_model_name)
        load_model(model, model_file, self.device)

        if self.use_gpu:
            # model = nn.DataParallel(model, device_ids=self.gpus).to(self.device)
            model = model.to(self.device)

        model.eval()

        test_dataset = eval(self.dataset_conf.loader_name)(self.config)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.train_conf.batch_size,
            num_workers=self.train_conf.num_workers,
            drop_last=False)

        model_prediction = []
        target = []
        if self.config.dataset.name == 'gender':
            for data, label in test_loader:
                output = model(data.to(self.device))
                output = torch.heaviside(output.view(output.size()[0]), torch.tensor(0).float())
                model_prediction += output.tolist()
                target += label.view(label.size()[0]).tolist()
            print(model_prediction, target)
            print(classification_report(target, model_prediction))

            print('acc is', accuracy_score(target, model_prediction))
            self.writer.add_text(self.dataset_conf.name, 'accuracy on test is:' +
                                 str(accuracy_score(target, model_prediction)))
        else:
            for data, label in test_loader:
                output = model(data.to(self.device))
                # output = torch.heaviside(output.view(output.size()[0]), torch.tensor(0).float())
                model_prediction += output.tolist()
                target += label.view(label.size()[0]).tolist()

            model_prediction = np.array(model_prediction)
            model_prediction = model_prediction.T
            target = np.array(target)
            corr = np.corrcoef(target, model_prediction)
            self.writer.add_text(self.dataset_conf.name + self.model_conf.name, 'correlation on test is:' +
                                 str(corr[0][1]))

            # add this result to all other results
            all_results_file = open(self.test_conf.all_results_dictionary, "rb")
            all_results_dict = pickle.load(all_results_file)
            all_results_dict[self.model_conf.name][self.dataset_conf.name] = corr[0][1]
            pickle.dump(all_results_dict, all_results_file)
            all_results_file.close()

            print(corr)

        self.writer.close()
