from pathlib import Path
import torch
from torch.utils.data import DataLoader
import sys
import importlib
import accelerate
from accelerate.utils import set_seed, gather_object, gather
from accelerate import Accelerator
from PIL.Image import fromarray as PIL_from_array
from rich.console import Console
from argparse import ArgumentParser
from omegaconf import OmegaConf

from model import build_network
from utils import (
    AnalysisFusionAcc,
    EasyProgress,
    easy_logger,
    get_eval_dataset,
    module_load,
    check_fusion_mask_inp,
    pad_any,
    unpad_any,
    LoguruLogger,
    y_pred_model_colored,
    MetricsByTask,
)
from task_datasets import DATASET_KEYS

logger = LoguruLogger.logger()

def ascii_tensor_to_string(ascii_tensor):
    # batched tensor of ascii code
    
    ascii_array = ascii_tensor.detach().cpu().numpy()
    
    string_s = []
    for arr in ascii_array:
        characters = [chr(code) for code in arr]
        string_s.append(''.join(characters))
        
    return string_s

def get_args():
    parser = ArgumentParser()
    parser.add_argument('-c', '--config_file', type=str, default=None, help='config file path')
    parser.add_argument('-m', '--model_class', type=str, default=None, help='model class name')
    parser.add_argument('--arch', type=str, default=None)
    parser.add_argument('--sub_arch', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='tno')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--load_spec_key', type=str, default=None)
    parser.add_argument('-p', '--pad_size_base', type=int, default=56)
    parser.add_argument('--val_bs', type=int, default=1)
    parser.add_argument('--reduce_label', action='store_true', default=False)
    parser.add_argument('--only_y', action='store_true', default=False)
    parser.add_argument('--dataset_mode', type=str, default='test', choices=['test', 'detection', 'train', 'all'])
    parser.add_argument('--save_path', type=str, default='visualized_img/')
    parser.add_argument('--extra_save_name', type=str, default='')
    parser.add_argument('--debug', action='store_true', default=False, help='debug mode')
    parser.add_argument('--analysis_fused', action='store_true', default=False, help='analysis fused image')
    parser.add_argument('--pad_window_base', type=int, default=32)  # 56
    parser.add_argument('--normalize', action='store_true', default=False, help='normalize the fused image')
    
    args = parser.parse_args()
    
    # to omegaconf
    conf = OmegaConf.create(args.__dict__)
    
    # load network config
    # old model loading
    if args.model_class is None:
        conf.full_arch = args.arch + "_" + args.sub_arch if args.sub_arch is not None else args.arch
        conf.network_configs_path = Path('configs') / f'{conf.arch}_config.yaml'
        
        # merge args and yaml_cfg
        yaml_cfg = OmegaConf.load(conf.network_configs_path)
        conf = OmegaConf.merge(conf, yaml_cfg)
        conf.network_configs = getattr(yaml_cfg['network_configs'], conf.full_arch, yaml_cfg['network_configs'])
    # new model loading
    else:
        assert args.config_file is not None, 'config file should be provided'
        yaml_cfg = OmegaConf.load(args.config_file)
        conf.full_arch = conf.model_class.replace('.', '_')
        model_init_kwargs = getattr(yaml_cfg.network_configs, conf.full_arch, yaml_cfg.network_configs)
        # merge args and yaml_cfg
        conf.merge_with(yaml_cfg)
        conf.network_configs = model_init_kwargs
        logger.info(f'initilize model {conf.model_class} using configure: {conf.network_configs}')
        
    # dataset and save path config
    conf.dataset = conf.dataset.lower()
    conf.path = yaml_cfg.path
    conf.path.base_dir = getattr(conf.path, f'{conf.dataset}_base_dir')
    conf.save_path = Path(args.save_path) / conf.full_arch / (conf.dataset + (f'_{args.extra_save_name}' if args.extra_save_name else ''))
    
    return conf


def inference_main(args):
    accelerator = Accelerator(mixed_precision='no')
    set_seed(2024)
    
    # logger
    logger = easy_logger()
    
    # device
    args.device = str(accelerator.device)
        
    # multiprocessing config
    device = accelerator.device
    n_process = accelerator.num_processes
    is_main_process = accelerator.is_main_process
    use_ddp = n_process > 1
    _n_saved_imgs = 0
    
    # print function
    only_rank_zero_print = (lambda *msg, level='info': getattr(logger, level)(*msg) if is_main_process else
                            lambda *args, **kwargs: None)
    
    # dataset config
    val_ds, val_dl = get_eval_dataset(args, logger)
    if val_ds is not None:
        val_dl = DataLoader(val_ds, batch_size=args.val_bs, shuffle=False)
    
    # model config
    if args.model_class is None:
        network = build_network(args.full_arch, **args.network_configs)
    else:
        logger.info(f'loading model from class: {args.model_class} with config file: {args.config_file}')
        _module_name, _class_name = args.model_class.split('.')
        _module = importlib.import_module(_module_name, package='model')
        network = getattr(_module, _class_name)(**OmegaConf.to_container(args.network_configs))
        
    logger.info(f'pad window base: {args.pad_window_base}')
    logger.info(f'use normalize: {args.normalize} to normalize the fused image')
        
    network = module_load(args.model_path, network, device=device, spec_key=args.load_spec_key)
    
    # metric analysor
    # only_on_y_component: fast on y component, slow on RGB
    analysor = AnalysisFusionAcc(only_on_y_component=True, test_metrics=MetricsByTask.ON_TRAIN)
    
    # prepare network, dataloader, and image saved path
    network, val_dl = accelerator.prepare(network, val_dl)
    network.eval()
    
    if is_main_process:
        save_path = args.save_path
        save_path.mkdir(exist_ok=True, parents=True)
        only_rank_zero_print(f'Ready to save images at: {save_path}')
    
    # for-loop inference
    tbar, task_id = EasyProgress.easy_progress(['inference'], [len(val_dl)],
                                               is_main_process=is_main_process,
                                               start_tbar=True, debug=args.debug,
                                               tbar_kwargs={'console': logger._console})
    only_rank_zero_print('start inference...')
    for i, data in enumerate(val_dl):
        split_context = accelerator.split_between_processes(data)
        
        with torch.no_grad() and torch.inference_mode() and split_context as split_d:
            keys_to_pad = DATASET_KEYS[args.fusion_task]
            keys_to_pad = keys_to_pad[:keys_to_pad.index('txt')]
            
            # prepare data and network inference
            split_d = check_fusion_mask_inp(split_d, dtype=torch.float32)
            split_d, padder = pad_any(args, split_d, args.pad_window_base, keys_to_pad=keys_to_pad, pad_mode='resize_tv')
            with y_pred_model_colored(split_d, data_modality_keys=keys_to_pad[:2], enable=args.only_y) as (_data, back_to_rgb):
                fused = network(**_data, cfg=args, to_rgb_fn=back_to_rgb, mode='fusion_eval')
            fused = padder.inverse(fused)
            if args.normalize:
                fused = (fused - torch.min(fused)) / (torch.max(fused) - torch.min(fused))
            else:
                fused = fused.clip(0, 1)
            split_d = unpad_any(args, _data, padder, keys_to_unpad=keys_to_pad)
            
            # analysis fused image
            if args.analysis_fused:
                analysor(split_d['gt'], fused)
    
            # save figs
            file_name = split_d['name']
            if use_ddp:
                gathered_fused = gather(fused).reshape(-1, *fused.size()[1:])  # [np * b, c, h, w]
                if isinstance(file_name, (list, tuple)):
                    _gathered_file_names = gather_object(file_name)
                    gathered_file_names = []
                    for batched_file_names in _gathered_file_names:
                        for name in batched_file_names:
                            gathered_file_names.append(name)
                elif isinstance(file_name, torch.Tensor):
                    gathered_file_names = gather(file_name)
                    gathered_file_names = ascii_tensor_to_string(gathered_file_names)
            else:
                gathered_fused = fused
                if isinstance(file_name, (list, tuple)):
                    gathered_file_names = file_name
                else:
                    gathered_file_names = ascii_tensor_to_string(file_name)
            
            assert gathered_fused.size(0) == len(gathered_file_names), 'gathered_fused and gathered_file_names should have the same length.'
                
            if is_main_process:
                for idx, fused in enumerate(gathered_fused):
                    saved_name = gathered_file_names[idx]
                    # saved_name = saved_name.split('.')[0] + '.png'
                    save_name = save_path / saved_name
                    fused = fused.permute(1, 2, 0).cpu().numpy()  # [h, w, c]
                    fused = (fused * 255).astype('uint8')
                    pil_img = PIL_from_array(fused)
                    if args.dataset.lower() in ['tno', 'roadscene']:
                        pil_img = pil_img.convert("L")
                    logger.info(f'saving figs {saved_name} ...')
                    pil_img.save(save_name, quality=100)
                _n_saved_imgs += gathered_fused.size(0)
                
            # advance progress bar
            if not args.debug:
                tbar.update(task_id, completed=i+1, total=len(val_dl), 
                            visible=True if i+1 < len(val_dl) else False,
                            description='Inference...')
                        
    
    # print results
    if use_ddp:
        analysors = gather_object(analysor)
        analysor.ave_result_with_other_analysors(analysors, ave_to_self=True)
        only_rank_zero_print(analysor.result_str())
    else:
        logger.info(analysor.result_str())
        
    only_rank_zero_print('Inference Done.')
    

if __name__ == '__main__':
    args = get_args()
    
    try:
        inference_main(args)
    except Exception as e:
        EasyProgress.close_all_tasks()
        logger.exception(e)
        # raise e
    
    
    
    
        
            
            
    
    
    
    