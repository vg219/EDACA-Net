# # import shutil
# # from pathlib import Path
# # import os
# # import re

# # def fused_name_to_int_str_name(fused_name):
# #     str_num = re.compile(r'(\d+)').findall(fused_name)[0]
# #     # unzfill str
# #     fused_name = str(int(str_num))
    
# #     return fused_name + '.png'
 

# # path_sdnet = '/Data3/cao/ZiHanCao/ImageFusionBenchmarks/MFF/MFF-Lytro/U2Fusion'
# # path_used = '/Data3/cao/ZiHanCao/ImageFusionBenchmarks/MFF/RealMFF/TC_MOA'

# # for file in Path(path_sdnet).glob('*'):
# #     # if file.name in os.listdir(path_sdnet):
# #     new_file_name = fused_name_to_int_str_name(file.name)
# #     print(file.name, f'-> {new_file_name}', 'in the dir')
# #     os.makedirs(Path(path_sdnet).parent / 'U2Fusion_2', exist_ok=True)
# #     shutil.copy(file, Path(path_sdnet).parent/'U2Fusion_2'/new_file_name)
# #     # else:
# #     #     print(file.name, 'not in the dir')


# # a = [[1,1],[2,2]]


# # def yield_lst(a):
# #     def _inner_iter(ai):
# #         yield from ai
        
# #     _a = iter(a)
# #     while True:
# #         try:
# #             ai = next(_a)
# #             yield from _inner_iter(ai)
# #         except StopIteration:
# #             _a = iter(a)
# #             ai = next(_a)
# #             yield from _inner_iter(ai)
        
# # for data in yield_lst(a):
# #     print(data)



# # d = {'EN': 5.7146, 'SD': 45.6572, 'SF': 9.2027, 'AG': 3.1401, 
# #      'MI': 1.6429, 'PSNR': 15.252, 'VIF': 0.5966,
# #      'Qabf': 0.4147, 'SSIM': 1.0312,'Qc': 0.684,   
# #      'Qp': 0.9285, 'Qcb': 0.4216, 'Qcv': 1/303.2, 'Qw': 0.7864, 
# #      'Qncie': 0.806, 'Qy': 0.6417}

# # import numpy as np

# # values = np.array(list(d.values()))
# # sum_ = np.sum(values)
# # print(sum_)

# # weights = 1 / (values / sum_)
# # print(weights)



# # import torch
# # import torch.nn as nn
# # from transformers import PretrainedConfig, PreTrainedModel

# # class MyConfig(PretrainedConfig):
# #     model_type = "my_model"
    
# # class MyModel(PreTrainedModel):
# #      config_class = MyConfig

# #      def __init__(self, config):
# #           super().__init__(config)
# #           self.lin = nn.Linear(10, 10)
          
          
# #      def forward(self, x):
# #           return self.lin(x)


    

# # from transformers import AutoModel, AutoConfig, AutoModelForCausalLM

# # AutoConfig.register("my_model", MyConfig)
# # AutoModel.register(MyConfig, MyModel)

# # # config = MyConfig()
# # # model = MyModel(config)
# # # # to bf16
# # # model.to(torch.bfloat16)

# # # # print(model)
# # # model.save_pretrained('./test_model_ckpt/my_model')
# # # print('save model')


# # # load

# # # model = AutoModel.from_pretrained('./test_model_ckpt/my_model')
# # # print(model)


# # # accelerate save state
# # import accelerate

# # # register
# # accelerator = accelerate.Accelerator()
# # model = accelerator.prepare(MyModel(MyConfig()))

# # # save state
# # accelerator.save_state('./test_model_ckpt/my_model_state')
# # print('save state')

# # # # load state
# # # accelerator.load_state('./test_model_ckpt/my_model')
# # # print('load state')




# # import pandas as pd

# # df = pd.read_excel('./fusion_table.xlsx', engine='openpyxl', header=4)

# # print(df)


# # ## to mark down
# # import rich
# # from rich.console import Console
# # from rich.table import Table

# # console = Console()

# # table = Table(title="Fusion Results")

# # table.add_column("Methods", style="cyan", no_wrap=True)
# # metrics = df.columns[1:]
# # for metric in metrics:
# #     table.add_column(metric)
    
# # # methods
# # for method in df.iloc[:, 0]:
# #     row = [method]
# #     for metric in metrics:
# #         row.append(str(df.loc[df.iloc[:, 0] == method, metric].values[0]))
# #     table.add_row(*row)


# # console.print(table)
        

# import torch
# from einops import rearrange

# def window_partition_1d(x: torch.Tensor, window_size: int):
#     """
#     window size: N is the size of the window in 2D, as for in sequence
#                 length, the size should be N^2
#     """
    
#     B, L, C = x.shape
#     window_size = int(window_size ** 2)
#     assert L % window_size == 0, "sequence length must be divisible by window size"
    
#     # partition
#     # B L C -> (B N) L/N C
#     windows = rearrange(x, 'b (l n) c -> (b n) l c', n=window_size).contiguous()
#     # windows = x.view(B * window_size, L // window_size, C)
    
#     return windows

# @torch.compile
# def window_partition_1d_view(x: torch.Tensor, window_size: int):
#     """
#     window size: N is the size of the window in 2D, as for in sequence
#                 length, the size should be N^2
#     """
    
#     B, L, C = x.shape
#     window_size = int(window_size ** 2)
#     assert L % window_size == 0, "sequence length must be divisible by window size"
#     l_wind = L // window_size
    
#     # partition
#     # B L C -> (B N) L/N C
#     # windows = rearrange(x, 'b (l n) c -> (b n) l c', n=window_size).contiguous()
#     windows = x.view(B, l_wind, window_size, C).permute(0, 2, 1, 3).reshape(B * window_size, l_wind, C)
    
#     return windows

# # x = torch.randn(2, 1024, 3).cuda()
# # x_view = window_partition_1d_view(x, 16)
# # print(x_view.shape)

# # x_re = window_partition_1d(x, 16)
# # print(x_re.shape)

# # # check if they are the same
# # print((x_view - x_re).abs().max())


# import torch
# from torch import Tensor
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np


# class SeqNet(nn.Module):
#     def __init__(self, in_c=1, out_c=1, channel=32, n_module=9):
#         super(SeqNet, self).__init__()

#         assert channel % 2 == 0

#         self.first_conv = nn.Conv1d(in_c, channel, 1)
#         self.relu = nn.ReLU()

#         self.b1 = DCBranch(2, channel, n_module)
#         self.b2 = DCBranch(3, channel, n_module)

#         self.tail_conv = nn.ModuleList([])
#         conv_t_ = 4
#         prev_c_ = channel * 2
#         channel_mults = [1 / 2, 1 / 2, 1 / 2]
#         for i in range(conv_t_):
#             if not (i == conv_t_ - 1):
#                 c = int(channel_mults[i] * prev_c_)
#                 self.tail_conv.append(
#                     nn.Sequential(nn.Conv1d(prev_c_, c, 3, 1, 1), nn.ReLU())
#                 )
#             else:
#                 self.tail_conv.append(nn.Conv1d(prev_c_, out_c, 1))

#             prev_c_ = c

#     def forward(self, x):
#         x = self.relu(self.first_conv(x))

#         x1 = self.b1(x)
#         x2 = self.b2(x)

#         x = torch.cat([x1, x2], dim=1)

#         for m in self.tail_conv:
#             x = m(x)

#         return x


# class DCBranch(nn.Module):
#     def __init__(self, stride, in_c, n_modules):
#         super().__init__()
#         self.net = nn.Sequential(
#             Masked1DConv(
#                 in_c, in_c, kernel_size=2 * stride - 1, stride=1, padding=stride - 1
#             ),
#             nn.BatchNorm1d(in_c),
#             nn.ReLU(),
#             nn.Conv1d(in_c, in_c, kernel_size=1),
#             nn.BatchNorm1d(in_c),
#             nn.ReLU(),
#             nn.Conv1d(in_c, in_c, kernel_size=1),
#             nn.ReLU(),
#             *[ResBlock1D(in_c, stride) for _ in range(n_modules)],
#             nn.Conv1d(in_c, in_c, kernel_size=1),
#         )

#     def forward(self, x):
#         x = self.net(x)
#         return x


# class ResBlock1D(nn.Module):
#     def __init__(self, in_c, stride):
#         super().__init__()

#         self.conv = nn.Conv1d(
#             in_c, in_c * 2, kernel_size=3, stride=1, padding=stride, dilation=stride
#         )
#         self.relu = nn.LeakyReLU()
#         self.bn = nn.BatchNorm1d(in_c * 2)
#         self.conv2 = nn.Conv1d(in_c * 2, in_c, kernel_size=1)

#     def forward(self, x):
#         return x + self.conv2(self.relu(self.bn(self.conv(x))))


# class Masked1DConv(nn.Conv1d):
#     def __init__(self, *args, **kwargs):
#         super(Masked1DConv, self).__init__(*args, **kwargs)
#         self.register_buffer("mask", self.weight.data.clone())

#         _, _, ks = self.weight.size()

#         self.mask.fill_(1)
#         self.mask[:, :, ks // 2] = 0

#     def forward(self, input: Tensor):
#         self.weight.data = self.weight.data * self.mask

#         return super().forward(input)


# if __name__ == "__main__":
#     ######################################
#     #* CPU 
#     # bs = 1
#     # time: 0.004255330562591553 seconds per sample
#     # bs = 128
#     # time: 0.0001473569869995117 seconds per sample
    
#     #* GPU
#     # bs = 1
#     # time: 0.003558034896850586 seconds per sample
#     # bs = 128
#     # time: 2.9826909303665162e-05 seconds per sample
    
#     ##* model
    
#     # | module                  | #parameters or shape   | #flops     |
#     # |:------------------------|:-----------------------|:-----------|
#     # | model                   | 15.204K                | 2.021M     |
        
    
    
    
#     net = SeqNet(2, 2, 8, 12).cuda()

#     x = torch.randn(128, 2, 128).cuda()
#     y = torch.randn(128, 2, 128).cuda()

#     # print(net(x).shape)

#     # x_denoised = net(x)
#     # loss = ((x_denoised - y) ** 2).mean()
#     # loss.backward()
#     with torch.no_grad():
#         for _ in range(100):
#             net(x)
    
#     import time
#     t1 = time.time()
#     with torch.no_grad():
#         for _ in range(100):
#             net(x)
#     t2 = time.time()
#     print(f'time: {(t2 - t1) / 100 / x.shape[0]} seconds per sample')
    
#     # import fvcore.nn as fvnn
    
#     # print(
#     #     fvnn.flop_count_table(fvnn.FlopCountAnalysis(net, x))
#     # )
    
#     # for n, p in net.named_parameters():
#     #     if p.grad is None and p.requires_grad:
#     #         print(n)


## test align_corners

# import torch
# import torch.nn.functional as F
# from torchvision.io import read_image

# img = read_image('/Data3/cao/ZiHanCao/datasets/MEF-SICE/over/097.jpg')[None]
# print(img.shape)

# img_interp_align_corners = F.interpolate(img, (128, 128), mode='bilinear', align_corners=True)
# print(img_interp_align_corners.shape)


# img_interp_no_align_corners = F.interpolate(img, (128, 128), mode='bilinear', align_corners=False)
# print(img_interp_no_align_corners.shape)


# img_interp_antialias = F.interpolate(img, (128, 128), mode='bilinear', align_corners=False, antialias=True)
# print(img_interp_antialias.shape)

# from test_log2 import LoguruLogger

# logger = LoguruLogger.logger()
# logger.add('test.log')
# logger.bind(name='log1')
# logger.info('info')

# logger.warning('this is a warning')


# def may_beartype_raise(
#     a: int | float
# ):
#     return a

# # print(may_beartype_raise([1]))

# from beartype.claw import beartype_package, beartype_this_package

# # beartype_package('test_log')
# beartype_this_package()

# # print(may_beartype_raise([1]))
