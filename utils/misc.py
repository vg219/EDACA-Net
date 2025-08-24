import argparse
import json
import os
import os.path as osp
import random
from typing import Any, Dict, Iterable, Mapping, Sequence, Union, Tuple, Optional
import importlib
import h5py

import yaml
import numpy as np
import torch
import torch.distributed as dist
import kornia
import shortuuid
from torch.backends import cudnn


def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def is_none(val):
    return val in ('none', 'None', 'NONE', None)

def set_all_seed(seed=2022):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    cudnn.deterministic = True  # not all operations are deterministic in Pytorch
    cudnn.benchmark = False

def to_numpy(*args):
    l = []
    for i in args:
        if isinstance(i, torch.Tensor):
            l.append(i.detach().cpu().numpy())
    return l


def to_tensor(*args, device, dtype):
    out = []
    for a in args:
        out.append(torch.tensor(a, dtype=dtype).to(device))
    return out

def args_no_str_none(value: str) -> "str | None":
    if value.lower() == "none":
        return None
    return value

def to_device(*args, device):
    out = []
    for a in args:
        out.append(a.to(device))
    return out

def h5py_to_dict(file: h5py.File, keys=None) -> dict[str, np.ndarray]:
    """get all content in a h5py file into a dict contains key and values

    Args:
        file (h5py.File): h5py file
        keys (list, optional): h5py file keys used to extract values.
        Defaults to ["ms", "lms", "pan", "gt"].

    Returns:
        dict[str, np.ndarray]:
    """
    d = {}
    if keys is None:
        keys = list(file.keys())
    for k in keys:
        print(f'reading key {k} array from h5 file...')
        d[k] = file[k][:]
    return d


def dict_to_str(d, decimals=4):
    n = len(d)
    # func = lambda k, v: f"{k}: {torch.round(v, decimals=decimals).item() if isinstance(v, torch.Tensor) else round(v, decimals)}"
    def func(k, v):
        if isinstance(v, torch.Tensor):
            return f"{k}: {round(v.item(), decimals)}"
        elif isinstance(v, np.ndarray):
            return f"{k}: {np.round(v, decimals=decimals)}"
        elif isinstance(v, (float, int, np.floating)):
            return f"{k}: {round(v, decimals)}"
        else:
            raise ValueError(f"Unsupported type: {type(v)}")
        
    s = ""
    for i, (k, v) in enumerate(d.items()):
        s += func(k, v) + (", " if i < n - 1 else "")
    return s


def prefixed_dict_key(d, prefix, sep="_"):
    # e.g.
    # SSIM -> train_SSIM
    d2 = {}
    for k, v in d.items():
        d2[prefix + sep + k] = v
    return d2


# deprecated
class CheckPointManager(object):
    def __init__(
        self,
        model: torch.nn.Module,
        save_path: str,
        save_every_eval: bool = False,
        verbose: bool = True,
    ):
        """
        manage model checkpoints
        Args:
            model: nn.Module, can be single node model or multi-nodes model
            save_path: str like '/home/model_ckpt/resnet.pth' or '/home/model_ckpt/exp1' when @save_every_eval
                       is False or True
            save_every_eval: when False, save params only when ep_loss is less than optim_loss.
                            when True, save params every eval epoch
            verbose: print out all information

        e.g.
        @save_every_eval=False, @save_path='/home/ckpt/resnet.pth'
        weights will be saved like
        -------------
        /home/ckpt
        |-resnet.pth
        -------------

        @save_every_eval=True, @save_path='/home/ckpt/resnet'
        weights will be saved like
        -------------
        /home/ckpt
        |-resnet
            |-ep_20.pth
            |-ep_40.pth
        -------------

        """
        self.model = model
        self.save_path = save_path
        self.save_every_eval = save_every_eval
        self._optim_loss = torch.inf
        self.verbose = verbose

        self.check_path_legal()

    def check_path_legal(self):
        if self.save_every_eval:
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
        else:
            assert self.save_path.endswith(".pth")
            par_dir = os.path.dirname(self.save_path)
            if not os.path.exists(par_dir):
                os.makedirs(par_dir)

    def save(
        self,
        ep_loss: Union[float, torch.Tensor] = None,
        ep: int = None,
        extra_saved_dict: dict = None,
    ):
        """

        Args:
            ep_loss: should be set when @save_every_eval=False
            ep: should be set when @save_every_eval=True
            extra_saved_dict: a dict which contains other information you want to save with model
                            e.g. {'optimizer_ckpt': op_ckpt, 'time': '2023/1/21'}

        Returns:

        """
        if isinstance(ep_loss, torch.Tensor):
            ep_loss = ep_loss.item()

        saved_dict = {}
        if not self.save_every_eval:
            assert ep_loss is not None
            if ep_loss < self._optim_loss:
                self._optim_loss = ep_loss
                path = self.save_path
                saved_dict["optim_loss"] = ep_loss
            else:
                print(
                    "optim loss: {}, now loss: {}, not saved".format(
                        self._optim_loss, ep_loss
                    )
                )
                return
        else:
            assert ep is not None
            path = os.path.join(self.save_path, "ep_{}.pth".format(ep))

        if extra_saved_dict is not None:
            assert "model" not in list(saved_dict.keys())
            saved_dict = extra_saved_dict

        try:
            saved_dict["model"] = self.model.module.state_dict()
        except:
            saved_dict["model"] = self.model.state_dict()

        torch.save(saved_dict, path)

        if self.verbose:
            print(
                f"saved params contains\n",
                *[
                    "\t -{}: {}\n".format(k, v if k != "model" else "model params")
                    for k, v in saved_dict.items()
                ],
                "saved path: {}".format(path),
            )


def is_main_process(func=None):
    """
    check if current process is main process in ddp
    warning: if not in ddp mode, always return True
    :return:
    """
    def _is_main_proc():
        if dist.is_initialized():
            return dist.get_rank() == 0
        else:
            return True
        
    if func is None:
        return _is_main_proc()
    else:
        def warp_func(*args, **kwargs):
            if _is_main_proc():
                return func(*args, **kwargs)
            else:
                return None
            
        return warp_func

def print_args(args):
    d = args.__dict__
    for k, v in d.items():
        print(f"{k}: {v}")


def yaml_load(name, base_path="./configs", end_with="_config.yaml"):
    path = osp.join(base_path, name + end_with)
    if osp.exists(path):
        f = open(path)
        cont = f.read()
        return yaml.load(cont, Loader=yaml.FullLoader)
    else:
        print(f"configuration file {path} not exists")
        raise FileNotFoundError(f'file not exists: {path}')


def json_load(name, base_path="./configs"):
    path = osp.join(base_path, name + "_config.json")
    with open(path) as f:
        return json.load(f)


def config_py_load(name, base_path="configs"):
    args = importlib.import_module(f".{name}_config", package=base_path)
    return args.config


from collections.abc import Mapping

class NameSpace(Mapping):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    
    @property
    def attrs(self):
        return self.__dict__
    
    def to_dict(self):
        d = self.attrs
        if isinstance(d, dict):
            out = {}
            for k, v in d.items():
                if isinstance(v, NameSpace):
                    out[k] = v.to_dict()
                elif isinstance(v, list):
                    lst = []
                    for i in v:
                        if isinstance(i, NameSpace):
                            lst.append(i.to_dict())
                        else:
                            lst.append(i)
                    out[k] = lst
                else:
                    out[k] = v
        elif isinstance(d, list):
            out = []
            for i in d:
                if isinstance(i, NameSpace):
                    out.append(i.to_dict())
                else:
                    out.append(i)
        else:
            out = d
        return out

    def __repr__(self, d=None, nprefix=0):
        repr_str = ""
        if d is None:
            d = self.attrs
        for k, v in d.items():
            if isinstance(v, NameSpace):
                repr_str += (
                    "  " * nprefix
                    + f"{k}: \n"
                    + f"{self.__repr__(v.attrs, nprefix + 1)}"
                )
            else:
                repr_str += "  " * nprefix + f"{k}: {v}\n"

        return repr_str
    
    def __getitem__(self, item):
        if item not in self.attrs:
            raise KeyError(f"{item} not in {self.attrs}")
        return self.attrs[item]

    def __setitem__(self, key, value):
        self.attrs[key] = value
    
    def __contains__(self, key: object) -> bool:
        keys_split = key.split('.')
        current = self
        for k in keys_split:
            if isinstance(current, NameSpace) and k in current.attrs:
                current = current.attrs[k]
            elif isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return False
        return True
        
    def __iter__(self):
        """
        support dict-like iteration and unpacking operator
        for example:
        
            1. dict-like iteration
        >>> args = NameSpace(a=1, b=2)
        >>> for k, v in args:
        >>>     print(k, v)
        >>> a 1
        >>> b 2
        
            2. unpacking operator
        >>> args = NameSpace(a=1, b=2)
        >>> {**args}
        {'a': 1, 'b': 2}
        """
        
        return iter(self.to_dict())
    
    def __next__(self):
        return next(iter(self))
    
    def __len__(self):
        return len(self.__dict__)
    
    @classmethod
    def dict_to_namespace(cls, d: "dict | Any | list"):
        return dict_to_namespace(d)
    
    @classmethod
    def merge_args(cls, *args: "NameSpace"):
        return merge_args(*args)
    
    @classmethod
    def merge_parser_args(cls, parser_args: argparse.Namespace, namespace_args: "NameSpace"):
        return merge_args_namespace(parser_args, namespace_args)
    
    def default_none_getattr(self, key: str) -> "Any | None":
        return default_none_getattr(self, key)
    
    def default_getattr(self, key: str, default: Any) -> Any:
        return default_getattr(self, key, default)
    
    @classmethod
    def init_from_yaml(cls, path: str, end_with: str = "_config.yaml"):
        d = yaml_load(path, end_with=end_with)
        return cls.dict_to_namespace(d)
    
    @classmethod
    def init_from_json(cls, path: str):
        d = json_load(path)
        return cls.dict_to_namespace(d)
    
    
def default_none_getattr(args: NameSpace, key: str) -> "Any | None":
    """
    Retrieve an attribute from a NameSpace object, returning None if the attribute does not exist.

    Args:
        args (NameSpace): The NameSpace object to retrieve the attribute from.
        key (str): The key of the attribute to retrieve.

    Returns:
        Any | None: The value of the attribute if it exists, otherwise None.
    """
    _args = args
    keys = key.split(".")
    for k in keys:
        _args = getattr(_args, k, None)
        if _args is None:
            return None
        
    return _args

def default_getattr(args: NameSpace, key: str, default: Any) -> Any:
    """
    Retrieve an attribute from a NameSpace object, returning a default value if the attribute does not exist.

    Args:
        args (NameSpace): The NameSpace object to retrieve the attribute from.
        key (str): The key of the attribute to retrieve.
        default (Any): The default value to return if the attribute does not exist.

    Returns:
        Any: The value of the attribute if it exists, otherwise the default value.
    """
    _args = args
    keys = key.split(".")
    for k in keys:
        _args = getattr(_args, k, default)
        if _args is default and (not isinstance(_args, NameSpace)):
            return default
        
    return _args
    

def dict_to_namespace(d: "dict | Any | list"):
    """
    convert a yaml-like configuration (dict) to namespace-like class

    e.g.
    {'lr': 1e-3, 'path': './datasets/train_wv3.h5'} ->
    NameSpace().lr = 1e-3, NameSpace().path = './datasets/train_wv3.h5'

    Warning: the value in yaml-like configuration should not be another dict
    :param d:
    :return:
    """
    namespace = NameSpace()
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(namespace, k, dict_to_namespace(v))
            elif isinstance(v, list):
                lst = []
                for i in v:
                    lst.append(dict_to_namespace(i))
                setattr(namespace, k, lst)
            else:
                setattr(namespace, k, v)
        return namespace
    elif isinstance(d, list):
        lst = []
        for i in d:
            lst.append(dict_to_namespace(i))
        return lst
    else:
        return d
    
def merge_args(*args: "NameSpace"):
    """
    merge multiple NameSpace objects into one
    key/value in latter args will cover former args' key/value
    """
    namespace = NameSpace()
    dicts = [d.to_dict() for d in args]
    d = {}
    for di in dicts:
        d.update(di)
        
    namespace = dict_to_namespace(d)
    return namespace

def merge_args_namespace(parser_args: argparse.Namespace, namespace_args: NameSpace):
    """
    merge parser_args and self-made class _NameSpace configurations together for better
    usage.
    return args that support dot its member, like args.optimizer.lr
    :param parser_args:
    :param namespace_args:
    :return:
    """
    # namespace_args.__dict__.update(parser_args.__dict__)
    namespace_d = namespace_args.__dict__
    for k, v in parser_args.__dict__.items():
        if not (k in namespace_d.keys() and v is None):
            setattr(namespace_args, k, v)

    return namespace_args


def generate_id(length: int = 8) -> str:
    # ~3t run ids (36**8)
    run_gen = shortuuid.ShortUUID(alphabet=list("0123456789abcdefghijklmnopqrstuvwxyz"))
    return str(run_gen.random(length))


def find_weight(weight_dir="./weight/", id=None, func=None):
    """
    return weight absolute path referring to id
    Args:
        weight_dir: weight dir that saved weights
        id: weight id
        func: split string function

    Returns: str, absolute path

    """
    assert id is not None, "@id can not be None"
    weight_list = os.listdir(weight_dir)
    if func is None:
        func = lambda x: x.split(".")[0].split("_")[-1]
    for id_s in weight_list:
        only_id = func(id_s)
        if only_id == id:
            return os.path.abspath(os.path.join(weight_dir, id_s))
    print(f"can not find {id}")
    return None

def clip_dataset_into_small_patches(
    file: h5py.File,
    patch_size: int,
    up_ratio: int,
    ms_channel: int,
    pan_channel: int,
    dataset_keys: Union[list[str], tuple[str]] = ("gt", "ms", "lms", "pan"),
    save_path: str = "./data/clip_data.h5",
):
    """
    clip patches at spatial dim
    Args:
        file: h5py.File of original dataset
        patch_size: ms clipped size
        up_ratio: shape of lms divide shape of ms
        ms_channel:
        pan_channel:
        dataset_keys: similar to [gt, ms, lms, pan]
        save_path: must end with h5

    Returns:

    """
    unfold_fn = lambda x, c, ratio: (
        torch.nn.functional.unfold(
            x, kernel_size=patch_size * ratio, stride=patch_size * ratio
        )
        .transpose(1, 2)
        .reshape(-1, c, patch_size * ratio, patch_size * ratio)
    )

    assert len(dataset_keys) == 4, "length of @dataset_keys should be 4"
    assert save_path.endswith("h5"), "saved file should end with h5 but get {}".format(
        save_path.split(".")[-1]
    )
    gt = unfold_fn(torch.tensor(file[dataset_keys[0]][:]), ms_channel, up_ratio)
    ms = unfold_fn(torch.tensor(file[dataset_keys[1]][:]), ms_channel, 1)
    lms = unfold_fn(torch.tensor(file[dataset_keys[2]][:]), ms_channel, up_ratio)
    pan = unfold_fn(torch.tensor(file[dataset_keys[3]][:]), pan_channel, up_ratio)

    print("clipped datasets shape:")
    print("{:^20}{:^20}{:^20}{:^20}".format(*[k for k in dataset_keys]))
    print(
        "{:^20}{:^20}{:^20}{:^20}".format(
            str(gt.shape), str(ms.shape), str(lms.shape), str(pan.shape)
        )
    )

    base_path = os.path.dirname(save_path)
    if not os.path.exists(base_path):
        os.makedirs(base_path)
        print(f"make dir {base_path}")

    save_file = h5py.File(save_path, "w")
    for k, data in zip(dataset_keys, [gt, ms, lms, pan]):
        save_file.create_dataset(name=k, data=data)
        print(f"create data {k}")

    file.close()
    save_file.close()
    print("file closed")
    
def dist_gather_object(obj, n_ranks=1, dest=0, all_gather=False):
    def _iter_tensor_to_rank(rank_obj, dest=0):
        if isinstance(rank_obj, dict):
            for k, v in rank_obj.items():
                if isinstance(v, torch.Tensor):
                    rank_obj[k] = v.to(dest)
                elif isinstance(v, Iterable):
                    rank_obj[k] = _iter_tensor_to_rank(v, dest)
        elif isinstance(rank_obj, (list, tuple)):
            if isinstance(rank_obj, tuple):
                rank_obj = list(rank_obj)
            for i, v in enumerate(rank_obj):
                if isinstance(v, torch.Tensor):
                    rank_obj[i] = v.to(dest)
                elif isinstance(v, Iterable):
                    rank_obj[i] = _iter_tensor_to_rank(v, dest)
        elif isinstance(rank_obj, torch.Tensor):
                rank_obj = rank_obj.to(dest)
    
        return rank_obj
    
    if n_ranks == 1:
        return obj
    elif n_ranks > 1:
        rank_objs = [None] * n_ranks
        if all_gather:
            # all proc to proc dest
            dist.all_gather_object(rank_objs, obj)
            # if is_main_process():
            #     _scattered_objs_lst = [rank_objs] * n_ranks
            # else:
            #     _scattered_objs_lst = [None] * n_ranks
            # received_objs = [None]
            # dist.scatter_object_list(received_objs, _scattered_objs_lst)
            rank_objs = _iter_tensor_to_rank(rank_objs, dest=dest)
        else:
            dist.gather_object(obj, rank_objs if is_main_process() else None, dest)
            if is_main_process():
                rank_objs = _iter_tensor_to_rank(rank_objs, dest)
        return rank_objs
    else:
        raise ValueError("n_ranks should be greater than 0")
    

    
    

if __name__ == "__main__":
    # path = "/home/ZiHanCao/datasets/HISI/new_harvard/x8/test_harvard(with_up)x8_rgb.h5"
    # file = h5py.File(path)
    # clip_dataset_into_small_patches(
    #     file,
    #     patch_size=16,
    #     up_ratio=8,
    #     ms_channel=31,
    #     pan_channel=3,
    #     dataset_keys=["GT", "LRHSI", "HSI_up", "RGB"],
    #     save_path="/home/ZiHanCao/datasets/HISI/new_harvard/x8/test_clip_128.h5",
    # )


    # vis = torch.randn(1, 3, 256, 256).clip(0, 1)
    # ir =  torch.randn(1, 1, 256, 256).clip(0, 1)

    
    # model = lambda vis, ir: vis

    # with y_pred_model_colored(vis, enable=True) as (y, back_to_rgb):
    #     pred_y = model(y, ir)
    #     pred_rgb = back_to_rgb(pred_y)
        
    # # assert equal
    # print(torch.isclose(pred_rgb, vis))
    
    # mean_diff = torch.mean(torch.abs(vis - pred_rgb))
    # print("mean difference:", mean_diff.item())
    
    # d = dict(
    #     a=1, b=2,
    #     c=dict(
    #         ca=1,
    #         cb=2,
    #     ),
    #     d=[1,2,3,4],
    #     e=[
    #         dict(
    #             ea=1,
    #             eb=2,
    #         ),
    #         dict(
    #             ea=3,
    #             eb=4,
    #         ),
    #     ],
    #     h = [[1,2,3], {'ha': 1, 'hb': 2}]
    # )
    
    # args = NameSpace.dict_to_namespace(d)
    # print('c.ca' in args)
    
    # from copy import deepcopy
    # args2 = deepcopy(args)
    # args2.d = [1,2,3]
    # args2['i'] = 'asdasfac'
    
    # args3 = merge_args(args, args2)
    # print(args3)
    
    
    # args_dict = args.to_dict()
    # print(args_dict)
    
    # print(args.a)
    # # print(args['c']['ca'])

    # print(default_getattr(args, 'a', 0), default_getattr(args, 'c.cb', 1))
    # print(default_getattr(args, 'c.cc', 0))
    
    
    
    # padder = WindowBasedPadder(64)
    
    # imgs = torch.randn(1, 6, 257, 257)
    
    # imgs = padder(imgs)
    
    # print(imgs.shape)
    
    # import PIL.Image
    # from torchvision.io import read_image
    
    # far = read_image("/Data3/cao/ZiHanCao/datasets/MEF-SICE/over/052.jpg").float() / 255.0
    # near = read_image("/Data3/cao/ZiHanCao/datasets/MEF-SICE/under/052.jpg").float() / 255.0
    
    # far = far.unsqueeze(0)
    # near = near.unsqueeze(0)
    
    # print(far.shape)
    # print(near.shape)
    
    # model = lambda x1, x2: (x1 + x2) / 2
    
    # with y_pred_model_colored((far, near), enable=True) as ((y1, y2), back_to_rgb):
    #     pred_y = model(y1, y2)
    #     pred_rgb = back_to_rgb(pred_y)
        
    #     PIL.Image.fromarray((pred_rgb.squeeze().permute(1, 2, 0).numpy() * 255).astype(np.uint8)).save("test.png")
        
    #     pass
    
    
    # pass
    
    # fast copy h5 array to a dict
    
    h5_file = '/Data3/cao/ZiHanCao/datasets/HISI/Chikusei_x4/train_Chikusei.h5'
    h5_file = h5py.File(h5_file)
    
    print('reading h5 file...')
    d = h5py_to_dict(h5_file)
    print(d.keys())