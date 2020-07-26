import pandas as pd
import numpy as np
import torch
import torch.nn as nn

class skip_connection(nn.Module):
    def __init__(self, inchannel, outchannel, keep_dim=True):
        super(skip_connection, self).__init__()

        if inchannel != outchannel:
            self.conv1d = nn.Conv1d(inchannel, outchannel, 1)

    def forward(self, before, after):
        '''
        :param before: the tensor before passing convolution blocks
        :param after: the tensor of output from convolution blocks
        :return: the sum of inputs
        '''

        if before.shape[2] != after.shape[2]:  # if the length is different (1/2)
            before = nn.functional.max_pool1d(before, 2, 2)

        if before.shape[1] != after.shape[1]:
            before = self.conv1d(before)

        return before + after


class residual_block(nn.Module):
    def __init__(self, inchannel, outchannel, pool=False, filtersize=15, activation=nn.LeakyReLU(), dr=0.3):
        super(residual_block, self).__init__()
        self.pool = pool
        self.seq = nn.Sequential(
            nn.BatchNorm1d(inchannel),
            activation,
            # nn.Dropout(dr),
            nn.Conv1d(inchannel, outchannel, filtersize, 1, filtersize // 2),
            nn.BatchNorm1d(outchannel),
            activation,
            # nn.Dropout(dr),
            nn.Conv1d(outchannel, outchannel, filtersize, 1, filtersize // 2)
        )
        self.skip = skip_connection(inchannel, outchannel)
        self.dropout = nn.Dropout(dr)

    def forward(self, input):
        out = self.seq(input)

        if self.pool:
            out = nn.functional.max_pool1d(out, 2, 2)
        out = self.skip(input, out)
        out = self.dropout(out)

        return out


class res_encoder(nn.Module):
    def __init__(self, f_num, f_size, input_size, dr=0.3, activation=nn.LeakyReLU(), channel_list=None):

        assert (f_size // 2) % 2, print('f_size//2 must be odds')

        super(res_encoder, self).__init__()
        self.f_num = f_num
        self.conv1st = nn.Conv1d(1, f_num, f_size, 1, f_size // 2)
        self.bn = nn.BatchNorm1d(f_num)

        if channel_list is None: # (input channels,output channels, 0:output size same 1: output size ->1/2, filter size)
            self.channel_list = [
                (f_num, f_num, 0, f_size), (f_num, f_num, 1, f_size),
                (f_num, f_num * 2,1,f_size),(f_num * 2, f_num * 3, 1, f_size),
                (f_num* 3, f_num * 4, 1, f_size), (f_num * 4, f_num * 4, 1, f_size),
                (f_num * 4, f_num * 5, 1, f_size // 2), (f_num * 5, f_num * 5, 1, f_size // 2),
                (f_num * 5, f_num * 5, 0, f_size // 2)
            ]
        else:
            self.channel_list = channel_list

        self.pool_count = 0
        self.output_size_list = []

        for ch_info in self.channel_list:
            self.pool_count += ch_info[2]

        self.output_channel = self.channel_list[-1][1]
        self.output_size_list.append(input_size)
        self.output_size = input_size
        for i in range(self.pool_count):
            self.output_size = self.output_size // 2
            self.output_size_list.append(self.output_size)

        self.encoder = nn.Sequential()
        for i, (inch, outch, pool, f_size) in enumerate(self.channel_list):
            self.encoder.add_module('residual_block_{}'.format(i),
                                    residual_block(inch, outch, pool, f_size, activation, dr))

    def forward(self, input):
        out = self.conv1st(input)
        out = self.bn(out)
        out = self.encoder(out)
        # out = nn.functional.avg_pool2d(out,(1,self.f_num*3))
        return out


class extract_latent(nn.Module):
    def __init__(self, encode_size, latent_size):
        super(extract_latent, self).__init__()

        assert encode_size > latent_size, print(
            "encode_size {} is samaller than latent_size {}".format(encode_size, latent_size))

        self.encode_size = encode_size
        self.latent_size = latent_size

        self.mu_linear = nn.Sequential(
            nn.Linear(self.encode_size, self.encode_size//2),
            nn.LeakyReLU(),
            nn.Linear(self.encode_size//2, self.latent_size)
        )
        self.std_linear = nn.Sequential(
            nn.Linear(self.encode_size, self.encode_size//2),
            nn.LeakyReLU(),
            nn.Linear(self.encode_size//2, self.latent_size)
        )
    def reparam_trick(self, mu, std):
        std = torch.exp(std / 2)
        try:  # GPU
            eps = torch.randn_like(std).to(std.get_device())
        except:  # CPU
            eps = torch.randn_like(std)

        latent = eps.mul(std).add_(mu)
        return latent

    def forward(self, encoded):
        encoded = encoded.squeeze()
        mu = self.mu_linear(encoded)
        std = self.std_linear(encoded)
        latent = self.reparam_trick(mu, std)
        return mu, std, latent


class skip_connection_de(nn.Module):
    def __init__(self, inchannel, outchannel, uppool, filter_size):
        super(skip_connection_de, self).__init__()
        self.inchannel = inchannel
        self.outchannel = outchannel
        self.uppool = uppool
        self.filter_size = filter_size

        if uppool:  ## input size -> input size*2
            self.pipeline = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='linear'),
                nn.ReflectionPad1d(filter_size // 2),
                nn.Conv1d(inchannel, outchannel, self.filter_size, 1, padding=0, bias=False),
                nn.BatchNorm1d(outchannel),
                nn.LeakyReLU()
            )
        else:  ## input size same
            self.pipeline = nn.Sequential(
                nn.Conv1d(inchannel, outchannel, self.filter_size, 1, filter_size // 2, bias=False),
                nn.BatchNorm1d(outchannel),
                nn.LeakyReLU(),
            )

    def forward(self, before, after):
        '''
        before : Tensor before transconv
        after : Tensor after transconv
        '''
        before = self.pipeline(before)
        out = before + after
        return out


class residual_block_de(nn.Module):
    def __init__(self, inchannel, outchannel, uppool=False, filter_size=15, activation=nn.LeakyReLU(), dr=0.2):
        super(residual_block_de, self).__init__()
        self.pipeline1 = nn.Sequential(
            nn.Conv1d(inchannel, inchannel, filter_size, 1, filter_size // 2, bias=False),
            nn.BatchNorm1d(inchannel),
            activation,
            # nn.Dropout(dropout)
        )
        if uppool:  ## input size -> input size*2
            self.pipeline2 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='linear'),
                nn.ReflectionPad1d(filter_size // 2),
                nn.Conv1d(inchannel, outchannel, filter_size, 1, padding=0, bias=False),
                nn.BatchNorm1d(outchannel),
                activation, )
            # nn.Dropout(dropout))
        else:  ## input size same
            self.pipeline2 = nn.Sequential(
                nn.Conv1d(inchannel, outchannel, filter_size, 1, filter_size // 2, bias=False),
                nn.BatchNorm1d(outchannel),
                activation,
                # nn.Dropout(dropout)
            )

        self.skip = skip_connection_de(inchannel, outchannel, uppool, filter_size)
        self.dropout = nn.Dropout(dr)

    def forward(self, input):
        out = self.pipeline1(input)
        out = self.pipeline2(out)
        #out = self.skip(input, out)
        out = self.dropout(out)
        return out


class res_decoder(nn.Module):
    def __init__(self, latent_size, encode_size, encode_channel, channel_list,activation,dr):
        super(res_decoder, self).__init__()
        self.encode_size = encode_size
        self.encode_channel = encode_channel
        self.reconstruct_size = encode_channel * encode_size
        self.channel_list = channel_list

        self.recon_linear = nn.Linear(latent_size, self.reconstruct_size)

        self.decoder = nn.Sequential()
        for i, (outch, inch, uppool, filter_size) in enumerate(self.channel_list):
            self.decoder.add_module('residual_decode_block({})'.format(i),
                                    residual_block_de(inch, outch, uppool, filter_size,activation,0.2))

        self.final_conv = nn.Conv1d(self.channel_list[-1][0], 1, 1)

    def forward(self, input):
        out = self.recon_linear(input)
        out = out.view(-1, self.encode_channel, self.encode_size)
        out_channels = self.decoder(out)
        out = self.final_conv(out_channels)
        return out,out_channels


class fully_res_VAE(nn.Module):
    def __init__(self, input_size, filter_size, filter_n, latent_size, channel_list=None,
                 activation=nn.LeakyReLU(negative_slope=0.01),dr=0.3):
        super(fully_res_VAE, self).__init__()
        self.input_size = input_size
        self.filter_size = filter_size
        self.filter_n = filter_n
        self.latent_size = latent_size
        #self, f_num, f_size, input_size, dr = 0.3, activation = nn.LeakyReLU(), channel_list = None
        self.encoder = res_encoder(self.filter_n, self.filter_size, self.input_size,dr,activation,channel_list)
        self.latent_extractor = extract_latent(self.encoder.output_channel*self.encoder.output_size, self.latent_size)
        self.decoder = res_decoder(latent_size, self.encoder.output_size, self.encoder.output_channel,
                                   self.encoder.channel_list[::-1],activation,dr)

        encode_size = self.encoder.output_size
        encode_channel = self.encoder.output_channel
        print('encode size: {} encode channel: {} latent_size: {}'.format(encode_size,encode_channel,latent_size))
        print('hear',self.encoder.output_channel*self.encoder.output_size)

    def forward(self, input):
        batchsize = input.shape[0]
        encoded = self.encoder(input)
        #encoded_avg = nn.functional.avg_pool1d(encoded, self.encoder.output_size).squeeze()
        encoded_flatten = encoded.view(batchsize,-1)
        mu, std, latent = self.latent_extractor(encoded_flatten)
        recon_out,de_channels = self.decoder(latent)
        return recon_out,de_channels ,(latent, torch.zeros_like(mu)), (mu, std)