import torch
from biosignal_dataset import biosignal_Dataset
import pandas as pd
import datetime
import pathlib
from utils_model import info_writer
import argparse
from torch.utils.data import DataLoader
import torch.nn as nn
import os
from pathlib import Path
from model_run import model_run
import git
from VAE import ConvVAE_skip
from VAE2 import fully_res_VAE

#torch, git, pyodbc 설치해야할 듯



def main():
    parser = argparse.ArgumentParser(description='Biosignal Autoencoder multigpu')
    parser.add_argument('--batch_size', type=int, default=256*2, metavar='N',help='input batch size for training (default: 64)')
    parser.add_argument('--test_batch_size', type=int, default=256*4, metavar='N',help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',help='learning rate (default: 0.0001)')
    parser.add_argument('--seed', type=int, default=10, metavar='S',help='random seed (default: 1)')
    parser.add_argument('--num_processes', type=int, default=1, metavar='N',help='how many training processes to use (default: 2)')
    parser.add_argument('--cuda',type=bool, default=True,help='enables CUDA training')
    parser.add_argument('--filter_size', type=int, default=27,help='size of filter(default:25)')
    parser.add_argument('--filter_num', type=int, default=16,help='number of filters (default:32)')
    parser.add_argument('--dropout',type=float,default=0.3,help='the rate of dropout')
    parser.add_argument('--duration', type=int, default=60,help='seconds of sigals length (default:60)')
    parser.add_argument('--add_noise', type=bool, default=False,help='add noise (default:False)')
    parser.add_argument('--add_norm', type=bool, default=True,help='add normalization (default:True)')
    parser.add_argument('--log_step', type=int, default=30,help='valid at the N step in trainning (default:30)')
    parser.add_argument('--fig_num', type=int, default=10,help='The number of figures at each step (default:5)')
    parser.add_argument('--fig_rand', type=bool, default=False,help='random fig in first valid iter  (default:False)')
    parser.add_argument('--need_visualizing', type=bool, default=True,help='need visualizaing (default:False)')
    parser.add_argument('--viz_step', type=int, default=100,help='visualizing at N train intertaion(default:100)')
    parser.add_argument('--gpu_start_num',type=int,default=6,help='the number of main gpu')
    parser.add_argument('--gpu_total_num',type=int,default=2,help='how many gpu will be used')
    parser.add_argument('--m',type=str,default='PH 500Hz dataset',help='text message')
    parser.add_argument('--input_len',type=int,default=2048,help='text message')
    parser.add_argument('--sample_run',type=bool,default=False,help='run with sample data')
    parser.add_argument('--latent_size',type=int,default=60,help='it must be smaller than ')
    parser.add_argument('--sampling_step',type=int,default=2,help='if it is 2, rawdata[::2]')



    args = parser.parse_args()

    time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    save_path = os.path.join(Path(os.getcwd()).parent, 'save',time)
    pathlib.Path(save_path).mkdir(parents=True,exist_ok=True)

    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    info_writer(args,sha,save_path)
    print('all folder and info done')

    torch.manual_seed(1)

    if args.sample_run :

        args.batch_size = 8
        args.test_batch_size = 8
        args.epochs =2
        args.log_step=5

        print('run Sample dataset')
        allfile_list = pd.read_csv('./data_example/ECG_list.csv', index_col=0)
        allfile_list = allfile_list.sort_values('pid')
        print('done read file list')

        gb = allfile_list.groupby('pid')
        gb_filelist = [gb.get_group(x) for x in gb.groups]
        gb_filelist = gb_filelist[:]

        train_filelist = pd.concat(gb_filelist[:]).filepath.tolist()
        valid_filelist = pd.concat(gb_filelist[:]).filepath.tolist()
        test_filelist = pd.concat(gb_filelist[:]).filepath.tolist()
    else:
        allfile_list = pd.read_csv('./data_example/ECG_list.csv', index_col=0)
        allfile_list = allfile_list.sort_values('pid')
        print('done read file list')

        gb = allfile_list.groupby('pid')
        gb_filelist = [gb.get_group(x) for x in gb.groups]
        gb_filelist = gb_filelist[:int(len(gb_filelist))]  # int(len(gb_filelist)/3)

        train_filelist = pd.concat(gb_filelist[:int(len(gb_filelist) * (80 / 100))]).filepath.tolist()
        valid_filelist = pd.concat(gb_filelist[int(len(gb_filelist) * (80 / 100)):int(len(gb_filelist) * (90 / 100))]).filepath.tolist()
        test_filelist = pd.concat(gb_filelist[int(len(gb_filelist) * (90 / 100)):]).filepath.tolist()


    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device('cuda:{}'.format(args.gpu_start_num)if use_cuda else 'cpu')
    gpu_num_list = list(range(args.gpu_start_num,args.gpu_start_num+args.gpu_total_num))

    train_dataset = biosignal_Dataset(train_filelist,device,args.input_len,sampling_step=args.sampling_step,add_norm=args.add_norm,add_noise=args.add_noise)
    valid_dataset = biosignal_Dataset(valid_filelist,device,args.input_len,sampling_step=args.sampling_step,add_norm=args.add_norm,add_noise=False)
    test_dataset = biosignal_Dataset(test_filelist,device,args.input_len,sampling_step=args.sampling_step,add_norm=args.add_norm,add_noise=False)

    #dataloader_kwargs = {'pin_memory': True} if use_cuda else {}

    train_loader = DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True\
                              ,num_workers=0,drop_last=True) #,**dataloader_kwargs
    valid_loader = DataLoader(valid_dataset, batch_size=args.test_batch_size, shuffle=True \
                             , num_workers=0, drop_last=True)  # **dataloader_kwargs
    test_loader = DataLoader(test_dataset,batch_size=args.test_batch_size,shuffle=False\
                              ,num_workers=0,drop_last=True) # **dataloader_kwargs

    print('data load done.')

    #중국 데이터때문에 input_len/2
    model = fully_res_VAE(int(args.input_len),args.filter_size,args.filter_num,args.latent_size,dr=args.dropout).double()
    #model = ConvVAE_skip(args.input_len,args.depth,args.filter_size,args.filter_num,args.latent_size,args.dropout).double()#.to(device)
    model = torch.nn.DataParallel(model,device_ids=gpu_num_list)
    #model.to(device)


    criterion = nn.MSELoss(reduction='sum')
    #criterion = DataParallelCriterion(criterion,device_ids=gpu_num_list)

    optim = torch.optim.Adam(model.parameters(),lr = args.lr)

    excutor = model_run(model,optim,criterion,args.epochs,train_loader,valid_loader,test_loader,device,save_path,time[-6:]
                        ,viz_step=args.viz_step,log_step=args.log_step,fig_num=args.fig_num,need_visualizing=args.need_visualizing)

    excutor.train()
    excutor.evaluation(mode='test')

    return

if __name__ == '__main__':
    main()






