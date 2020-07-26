from pathlib import Path
import os
from datetime import datetime
import torch
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils_model import KLD_func,kl_anneal_function
import numpy as np

class model_run():
    def __init__(self, model, optim, criterion, epochs, train_loader, valid_loader, test_loader, device,save_path,time, log_step=50,viz_step=10,
                 fig_num=10, fig_rand=False, need_visualizing=False):
        '''
        log_step(n) : the time of validation at  (n-th) train loop in a train epoch
        fig_num : how many sample output figures you need
        fig_rand : bool (random choice for visualizing output)
        '''
        self.model = model.to(device)
        self.optim = optim
        self.criterion = criterion
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.device = device
        self.model = model.to(device)
        self.epochs = epochs
        self.log_step = log_step
        self.viz_step = viz_step
        self.need_visualizing = need_visualizing
        self.fig_num = fig_num
        self.fig_rand = fig_rand
        self.valid_history = []
        self.save_path = save_path
        self.best_model_path = None
        self.time = time
        self.step=0

        Path(self.save_path).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(self.save_path, 'model')).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(self.save_path, 'figs')).mkdir(parents=True, exist_ok=True)
        print('this is the criterion class name')
        print(self.criterion.__class__.__name__)

    def train(self):
        self.step=0
        self.valid_history = []
        for epoch in range(self.epochs):
            train_loss = self.train_epoch(epoch)
            print('({})th TOTAL TRAIN LOSS : {:.3f}'.format(epoch,train_loss))


    def train_epoch(self, epoch):
        epoch_loss = 0
        for i, (origin_data,input_data) in enumerate(self.train_loader):
            self.model.train()
            self.optim.zero_grad()
            batch_size = origin_data.shape[0]
            input_data = input_data.to(self.device)
            output_data,_,(z_mu,z_var) = self.model(input_data)

            recon_err = self.criterion(output_data, origin_data)
            KLD_err = KLD_func(z_mu,z_var)
            KLD_weight = kl_anneal_function('logistic',self.step)
            loss = (recon_err + KLD_weight*KLD_err)/batch_size
            loss.backward()
            self.optim.step()
            self.step+=1

            epoch_loss += loss.detach().cpu().item()

            print('{}({}) {} epoch {}/{} >> Total:{:.3f} Recon:{:.3f} KLD:{:.3f}'.format(\
                self.time,'train', epoch, i,len(self.train_loader), loss.detach().item(),recon_err.detach().item()/batch_size,KLD_err.detach().item()/batch_size))

            if self.need_visualizing and (i + 1) % self.viz_step == 0:
                if self.criterion.__class__.__name__ == 'DataParallelCriterion':
                    output_data = output_data[0]
                self.visualization('train',input_data, output_data, epoch, i)


            if (i + 1) % self.log_step == 0 and epoch >= 0:
                valid_loss = self.evaluation(mode='valid', epoch=epoch, iteration=i)
                self.valid_history.append(valid_loss)
                if valid_loss <= min(self.valid_history):
                    model_save_path = os.path.join(self.save_path, 'model',
                                                   '({})epoch_({})iter_{:.3f}.pth'.format(epoch, i, valid_loss))
                    self.best_model_path = model_save_path
                    torch.save(self.model.state_dict(), model_save_path)

        return epoch_loss / len(self.train_loader)

    def evaluation(self, mode=None, epoch='test', iteration='test', visualization=False):
        assert mode in ['valid', 'test'], print('please select mode in [valid,test]')
        dataloader = self.valid_loader if mode == 'valid' else self.test_loader

        if mode =='test':
            print(self.best_model_path)
            self.model.load_state_dict(torch.load(self.best_model_path))

        self.model.eval()
        epoch_loss = 0
        with torch.no_grad():
            for i, (origin_data,input_data) in enumerate(dataloader):
                batch_size = origin_data.shape[0]
                input_data = input_data.to(self.device)
                output_data,_,(z_mu,z_var) = self.model(input_data)
                loss = self.criterion(output_data, origin_data)
                KLD_err = KLD_func(z_mu, z_var)

                epoch_loss += ((loss+KLD_err)/batch_size).item()

                if self.need_visualizing and i == 0:
                    if self.criterion.__class__.__name__ =='DataParallelCriterion':
                        output_data = output_data[0]
                    self.visualization(mode,input_data, output_data, epoch, iteration)

        total_loss = epoch_loss / len(dataloader)
        print('({}) {} epoch {} iter >> loss : {:.4f}'.format(mode, epoch, iteration, total_loss))

        return total_loss

    def visualization(self,mode ,input_data, output_data, epoch, iteration):

        rows = output_data.shape[0]
        if random:
            idx = random.sample(range(rows), self.fig_num) if self.fig_num < rows else list(range(rows))
            output_samples = output_data[idx].detach().cpu().numpy()
            input_samples = input_data[idx].detach().cpu().numpy()
        else:
            output_samples = output_data[range(self.fig_num)].detach().cpu().numpy()
            input_samples = input_data[range(self.fig_num)].detach().cpu().numpy()

        for n, (insample, outsample) in enumerate(zip(input_samples, output_samples)):
            plt.figure(figsize=(24, 12))
            plt.subplot(211)
            plt.title('input')
            plt.plot(insample[0])
            plt.subplot(212)
            plt.plot(outsample[0])
            plt.title('recon')
            plt.savefig(os.path.join(self.save_path, 'figs', '{}({})epoch_({})iter_{}th.png'.format(mode,epoch, iteration, n)))
            plt.cla()
            plt.close()
            # plt.show()

