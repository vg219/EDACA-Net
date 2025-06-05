# import time
# import sys
# import logging
# from functools import partial
# from typing import Callable, List, Union, Protocol
# from contextlib import nullcontext

# from rich.console import Console
# from rich.logging import RichHandler
# from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn, SpinnerColumn


# class EasyProgress:
#     tbar: Progress = None
#     task_desp_ids: dict[str, int] = {}
    
#     @classmethod
#     def console(cls):
#         assert cls.tbar is not None, '`tbar` has not initialized'
#         return cls.tbar.console
    
#     @classmethod
#     def close_all_tasks(cls):
#         if cls.tbar is not None:
#             for task_id in cls.tbar.task_ids:
#                 cls.tbar.stop_task(task_id)
#                 # set the task_id all unvisible
#                 cls.tbar.update(task_id, visible=False)
                
    
#     @classmethod
#     def easy_progress(cls,
#                       task_desciptions: list[str], 
#                       task_total: list[int],
#                       tbar_kwargs: dict={},
#                       task_kwargs: list[dict[str, Union[str, int]]]=None,
#                       is_main_process: bool=True,
#                       *,
#                       start_tbar: bool=True,
#                       debug: bool=False) -> tuple[Progress, Union[list[int], int]]:
#         """get a rich progress bar 

#         Args:
#             task_desciptions (list[str]): list of task descriptions of `len(task_desciptions)` tasks
#             task_total (list[int]): list of length each task
#             tbar_kwargs (dict, optional): kwargs for progress bar. Defaults to {}.
#             task_kwargs (list[dict[str, Union[str, int]]], optional): task kwargs for each task. Defaults to None.
#             is_main_process (bool, optional): if is main process. Defaults to True.
#             start_tbar (bool, optional): start running progress bar when ini. Defaults to True.
#             debug (bool, optional): debug mode, set progress bar to be unvisible. Defaults to False.

#         Returns:
#             tuple[Progress, Union[list[int], int]]: Progress bar and task ids
#         """
        
#         def _add_task_ids(tbar: Progress, task_desciptions, task_total, task_kwargs):
#             task_ids = []
#             if task_kwargs is None:
#                 task_kwargs = [{'visible': False}] * len(task_desciptions)
#             for task_desciption, task_total, id_task_kwargs in zip(task_desciptions, task_total, task_kwargs):
#                 if task_desciption in list(EasyProgress.task_desp_ids.keys()):
#                     task_id = EasyProgress.task_desp_ids[task_desciption]
#                     task_ids.append(task_id)
#                 else:
#                     task_id = tbar.add_task(task_desciption, total=task_total, **id_task_kwargs)
#                     task_ids.append(task_id)
#                     EasyProgress.task_desp_ids[task_desciption] = task_id
                
#             return task_ids if len(task_ids) > 1 else task_ids[0]
        
#         def _new_tbar_with_task_ids(task_desciptions, task_total, task_kwargs):
#             if is_main_process:
#                 if task_kwargs is not None:
#                     assert len(task_desciptions) == len(task_total) == len(task_kwargs)
#                 else:
#                     assert len(task_desciptions) == len(task_total)
                
#                 # if (console := tbar_kwargs.pop('console', None)) is not None:
#                 #     console._color_system = console._detect_color_system()
                
#                 # if 'console' in tbar_kwargs:
#                 #     tbar_kwargs['console']._color_system = tbar_kwargs['console']._detect_color_system()
                    
#                 tbar = Progress(TextColumn("[progress.description]{task.description}"),
#                                 BarColumn(),
#                                 TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
#                                 SpinnerColumn(),
#                                 TimeRemainingColumn(),
#                                 TimeElapsedColumn(),
#                                 **tbar_kwargs)
#                 EasyProgress.tbar = tbar
                
#                 task_ids = _add_task_ids(tbar, task_desciptions, task_total, task_kwargs)
                
#                 return tbar, task_ids
#             else:
#                 return nullcontext(), [None] * len(task_desciptions) if len(task_desciptions) > 1 else None
        
#         def _cached_tbar_with_new_task_ids(task_desciptions, task_total, task_kwargs):
#             if is_main_process:
#                 tbar = EasyProgress.tbar
                
#                 task_ids = []
#                 if task_kwargs is None:
#                     task_kwargs = [{'visible': False}] * len(task_desciptions)
                
#                 task_ids = _add_task_ids(tbar, task_desciptions, task_total, task_kwargs)
                
#                 return tbar, task_ids
#             else:
#                 return nullcontext(), [None] * len(task_desciptions) if len(task_desciptions) > 1 else None
        
#         if not debug:
#             if EasyProgress.tbar is not None:
#                 rets = _cached_tbar_with_new_task_ids(task_desciptions, task_total, task_kwargs)
#             else:
#                 rets = _new_tbar_with_task_ids(task_desciptions, task_total, task_kwargs)
#             if start_tbar and is_main_process and not EasyProgress.tbar.live._started:
#                 EasyProgress.tbar.start()
#             return rets
#         else:
#             return nullcontext(), [None] * len(task_desciptions) if len(task_desciptions) > 1 else None
        


# def easy_logger(level='INFO'):
#     format_str = "[%(asctime)s] %(message)s"
#     rich_handler = RichHandler(show_path=False, level=level)
#     logging.basicConfig(format=format_str,
#                         level=level,
#                         datefmt='%X',
#                         handlers=[rich_handler],
#                         )
    
#     class ProtocalLogger(Protocol):
#         @classmethod
#         def print(ctx, msg, level: Union[str, int]="INFO"):
#             if isinstance(level, str):
#                 level = eval(f'logging.{level.upper()}')
#             logger.log(level, msg, extra={"markup": True})
            
#         @classmethod
#         def debug(ctx, msg):
#             pass
        
#         @classmethod
#         def info(ctx, msg):
#             pass
        
#         @classmethod
#         def warning(ctx, msg):
#             pass
        
#         @classmethod
#         def error(ctx, msg, raise_error: bool=False, error_type=None):
#             ctx.print(msg, level='ERROR')
#             if raise_error:
#                 if error_type is not None:
#                     raise error_type(msg)
                
#                 raise RuntimeError(msg)
            
    
#     logger: ProtocalLogger = logging.getLogger(__name__)
#     # logger.addHandler(RichHandler(show_path=False))
    
#     logger.print = ProtocalLogger.print
#     logger.debug = partial(ProtocalLogger.print, level='DEBUG')
#     logger.info = partial(ProtocalLogger.print, level='INFO')
#     logger.warning = partial(ProtocalLogger.print, level='WARNING')
#     logger.error = ProtocalLogger.error
    
#     logger._console = rich_handler.console
    
#     return logger



# logger = easy_logger()
# pbar, (t1, t2) = EasyProgress.easy_progress(['task1', 'task2'], [100, 200], start_tbar=True,
#                                   tbar_kwargs={'console': logger._console})

# for i in range(100):
#     pbar.update(t1, total=100, completed=i, description='task1', visible=True if i < 100 else False)
#     pbar.update(t2, total=200, completed=i, description='task1', visible=True if i < 200 else False)
    
#     logger.info(i)
    
#     time.sleep(0.6)


# # import time
# # import sys
# # import logging

# # from rich.console import Console
# # from rich.logging import RichHandler
# # from rich.progress import Progress

# # level='INFO'

# # format_str = "[%(asctime)s] %(message)s"
# # rich_handler = RichHandler(console=Console(file=sys.stdout), show_path=False, level=level)
# # logging.basicConfig(format=format_str,
# #                     level=level,
# #                     datefmt='%X',
# #                     handlers=[rich_handler],
# #                     )
# # logger = logging.getLogger('rich')

# # with Progress(expand=False, console=rich_handler.console) as progress:
# #     task = progress.add_task("Loading...",)
# #     while not progress.finished:
# #         logger.info('some log output')
# #         progress.update(task, advance=1)
# #         time.sleep(1)

import torch
import torch.nn as nn
import torch.nn.functional as F

class SobelOp(nn.Module):
    def __init__(self):
        super(SobelOp, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2, 0, 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0, 0, 0],
                  [-1, -2, -1]]
        kernelx = torch.tensor(kernelx, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        kernely = torch.tensor(kernely, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        kernelx = kernelx.repeat(3, 1, 1, 1)
        kernely = kernely.repeat(3, 1, 1, 1)
        self.register_buffer('weightx', kernelx)
        self.register_buffer('weighty', kernely)
        
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1, groups=x.size(1))
        sobely=F.conv2d(x, self.weighty, padding=1, groups=x.size(1))
        
        # sobel_xy = torch.abs(sobelx)+torch.abs(sobely)
        
        sobel_xy = torch.max(
            torch.abs(sobelx), torch.abs(sobely)
        )
        
        return sobel_xy
    
x = torch.randn(1,3,224,224)
sobel = SobelOp()

print(sobel(x).shape)