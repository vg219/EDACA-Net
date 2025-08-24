import importlib
import os
import os.path as osp
import sys
import argparse
import warnings
from rich.console import Console
from types import SimpleNamespace

import accelerate
from accelerate.utils import DummyOptim, DummyScheduler, set_seed

from accelerate_run_hyper_ms_engine import train as train_sharpening
from accelerate_run_fusion_engine import train as train_fusion
from accelerate.utils import ProjectConfiguration, DataLoaderConfiguration, GradientAccumulationPlugin
from model import build_network, lazy_load_network
from utils import (
    NameSpace,
    TensorboardLogger,
    BestMetricSaveChecker,
    LoguruLogger,
    EasyProgress,
    config_load,
    convert_config_dict,
    get_optimizer,
    get_scheduler,
    h5py_to_dict,
    is_main_process,
    merge_args_namespace,
    module_load,
    get_loss,
    generate_id,
    # get_fusion_dataset,
    get_train_dataset,
    args_no_str_none,
)
logger = LoguruLogger.logger(sink=sys.stdout)
LoguruLogger.add('log_file/running_traceback.log', format="{time:MM-DD hh:mm:ss} {level} {message}", 
                level="WARNING", backtrace=True, diagnose=True, mode='w')
LoguruLogger.add(sys.stderr, format="{time:MM-DD hh:mm:ss} {level} {message}", 
                level="ERROR", backtrace=True, diagnose=True)


def get_main_args():
    parser = argparse.ArgumentParser("PANFormer")

    # network
    # NOTE: may decrepted
    parser.add_argument("-a", "--arch", type=str, default="pannet")
    parser.add_argument("--sub_arch", default=None, help="panformer sub-architecture name")
    
    parser.add_argument("-c", "--config_file", type=str, default=None, help='config file path')
    parser.add_argument("-m", "--model_class", type=str, help="model import path")
    
    # train config
    parser.add_argument("--pretrain_model_path", type=str, default=None, help='pretrained model path')
    parser.add_argument("--non_load_strict", action="store_false", default=True)
    parser.add_argument("-e", "--num_train_epochs", type=int, default=500)
    parser.add_argument("--val_n_epoch", type=int, default=30)
    parser.add_argument("--warm_up_epochs", type=int, default=10)
    parser.add_argument("-l", "--loss", type=str, default="mse")
    parser.add_argument("--pad_window_base", type=int, default=32)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--checkpoint_every_n", default=None, type=int, help='checkpointing the running state whether the saving condition is met or not '
                                                                             '(see the `check_save_fn` in the training function)')
    parser.add_argument("--ckpt_max_limit", type=int, default=None, help='maximum number of checkpoints to keep')
    parser.add_argument("--mix_precison", default='fp32', choices=['fp32', 'fp16'], help="mixed precision training")
    parser.add_argument("--sanity_check", action="store_true", default=False)
    parser.add_argument("--regardless_metrics_save", action="store_true", default=False, help='regardless of the metric and save the model when validation process is done')
    # decrepted
    parser.add_argument("--pretrain", action="store_true", default=False)
    parser.add_argument("--pretrain_id", type=str, default=None)

    # resume training config
    parser.add_argument("--resume_path", default=None, required=False, help="path for resuming state")
    # decrepcted
    parser.add_argument("--resume_lr", type=float, required=False, default=None)
    parser.add_argument("--resume_total_epochs", type=int, required=False, default=None)
    parser.add_argument("--nan_no_raise", action="store_true", default=False, help='not raise error when nan loss')
    
    # path and load
    parser.add_argument("-p", "--path", type=str, default=None, help="only for unsplitted dataset")
    parser.add_argument("--split_ratio", type=float, default=None)
    parser.add_argument("--load", action="store_true", default=False, help="resume training")
    parser.add_argument("--save_base_path", type=str, default="./weight")
    parser.add_argument("--proj_root", type=str, default="log_file", help="proj_root, do not set it", required=False)

    # datasets config
    parser.add_argument("--dataset", type=str, default="wv3")
    parser.add_argument("-b", "--batch_size", type=int, default=1028, help='set train and val bs')
    parser.add_argument("--train_bs", type=int, default=None)
    parser.add_argument("--val_bs", type=int, default=None)
    parser.add_argument("--fusion_crop_size", type=int, default=72, help='image cropped size for fusion task')
    parser.add_argument("--only_y_train", action="store_true", default=False, help='train model only on Y channel, distinguish it from `only_y` argument from dataset configuration')
    parser.add_argument("--shuffle", type=bool, default=True)
    parser.add_argument("--fast_eval_n_samples", type=int, default=128)
    parser.add_argument("--aug_probs", nargs="+", type=float, default=[0.0, 0.0], help='augmentation probabilities for train and validation dataset')
    parser.add_argument("-s", "--seed", type=int, default=3407)
    parser.add_argument("-n", "--num_workers", type=int, default=8)
    parser.add_argument("--ergas_ratio", type=int, choices=[2, 4, 8, 16, 20], default=4)

    # logger config
    parser.add_argument("--logger_on", action="store_true", default=False)
    parser.add_argument("--proj_name", type=str, default="panformer_wv3")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--resume_id", type=str, default="None", help='resume training id')
    parser.add_argument("--run_id", type=str, default=generate_id())
    parser.add_argument("--watch_log_freq", type=int, default=10)
    parser.add_argument("--watch_type", type=str, default="None")
    parser.add_argument("--log_metrics", action="store_true", default=True)
    parser.add_argument("--debug", action="store_true", default=False)

    # ddp setting
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--dist-url", default="env://", help="url used to set up distributed training")
    parser.add_argument("--ddp", action="store_true", default=False)

    # some comments
    parser.add_argument("--comment", type=str, required=False, default="")

    return parser.parse_args()

def main(args):
    accelerator = accelerate.Accelerator(mixed_precision='no',
                                         dataloader_config=DataLoaderConfiguration(split_batches=False,
                                                                                   even_batches=False,
                                                                                   non_blocking=True,
                                                                                   dispatch_batches=None,
                                                                                   use_seedable_sampler=False,
                                                                                   use_stateful_dataloader=False),
                                         project_config=ProjectConfiguration(project_dir=args.proj_root,
                                                                             total_limit=args.ckpt_max_limit,
                                                                             automatic_checkpoint_naming=True,),
                                         gradient_accumulation_plugin=GradientAccumulationPlugin(num_steps=args.grad_accum_steps,
                                                                                                 adjust_scheduler=False,
                                                                                                 sync_with_dataloader=False,
                                                                                                 sync_each_batch=False,),
                                         )
    set_seed(args.seed, device_specific=True)
    print(f'>>> PID - {os.getpid()}: accelerate launching...')
    # logger.print(repr(args))
    
    device = accelerator.device
    args.device = str(device)
    args.ddp = accelerator.use_distributed
    args.world_size = accelerator.num_processes
    
    # get network config
    if args.config_file is not None:
        if not args.config_file.endswith('.yaml'):
            args.config_file += '.yaml'
        configs = config_load(args.config_file, end_with="")
    else:
        # old config loading
        configs = config_load(args.arch, "./configs")
    args = merge_args_namespace(args, convert_config_dict(configs))
    
    # define network
    if args.model_class is not None:
        _module_name, _class_name = args.model_class.split('.')
        args.model_class = '_'.join([_module_name, _class_name])
        network_configs = args.network_configs.to_dict()
        model_init_kwargs = getattr(args.network_configs, args.model_class)
        try:
            network = lazy_load_network(module_name=_module_name, model_class_name=_class_name,
                                        **vars(model_init_kwargs)).to(device)
        except ValueError as e:
            print(f"Failed to load {args.model_class} from model/__init__.py file, try to load directly")
            _module = importlib.import_module(_module_name, package='model')
            network = getattr(_module, _class_name)(**vars(model_init_kwargs)).to(device)
        args.full_arch = args.model_class
    else:
        # may decrepted
        warnings.warn('initializing network from arch and sub_arch is deprecated, use model_class instead', DeprecationWarning)
        full_arch = args.arch + "_" + args.sub_arch if args.sub_arch is not None else args.arch
        args.full_arch = full_arch
        network_configs = getattr(args.network_configs, full_arch, args.network_configs).to_dict()
        network = build_network(full_arch, **network_configs).to(device)
        
    # get logger
    if args.logger_on and accelerator.is_main_process:
        logger = TensorboardLogger(
            tsb_comment=args.run_id,
            args=args,
            file_stream_log=True,
            method_dataset_as_prepos=True,
        )
        logger.watch(
            network=network,
            watch_type=args.watch_type,
            freq=args.watch_log_freq,
        )
    else:
        from utils import NoneLogger
        logger = NoneLogger(cfg=args, name=args.proj_name)
        
    # make save path legal
    
    # FIXME: temporary solution (handle the run_id) using ddp to exchange the run_id
    if accelerator.is_main_process:
        weight_path = osp.join(logger.log_file_dir, 'weights')
        if accelerator.is_main_process:
            os.makedirs(weight_path, exist_ok=True)
        args.output_dir = weight_path
        args.save_model_path = osp.join(args.output_dir, 'ema_model.pth')
    else:
        # for ddp broadcast
        args.output_dir = args.save_model_path = args.save_base_path = [None] 
        
    if accelerator.use_distributed:
        (args.output_dir, 
         args.save_model_path,
         args.save_base_path) = accelerate.utils.broadcast_object_list([args.output_dir, 
                                                                        args.save_model_path, 
                                                                        args.save_base_path])
    accelerator.project_configuration.project_dir = args.output_dir
    logger.info(f'model weights will be save at {args.output_dir}')
            
    # handle the optimizer and lr_scheduler
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in network.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.optimizer.weight_decay,},{
            "params": [p for n, p in network.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    # get loss and dataset
    loss_cfg = getattr(args, 'loss_cfg', NameSpace())  # TODO: make `loss_cfg` to `loss_configs`
    loss_cfg.fusion_model = network
    loss_cfg.device = device
    loss_cfg.model_is_y_pred = args.only_y_train
    criterion = get_loss(args.loss, **vars(loss_cfg[args.loss])).to(device)
    
    if args.train_bs is None:
        args.train_bs = args.batch_size
    if args.val_bs is None:
        args.val_bs = args.batch_size
    
    # get train and val dataset and dataloader
    (_, train_dl, _, val_dl), args = get_train_dataset(args)
    
    # Creates Dummy Optimizer if `optimizer` was specified in the config file else creates Adam Optimizer
    if (accelerator.state.deepspeed_plugin is None or
        "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config):
        optimizer = get_optimizer(network, network.parameters(), **args.optimizer.to_dict())
    else:
        optimizer = DummyOptim(optimizer_grouped_parameters)
        
    # Creates Dummy Scheduler if `scheduler` was specified in the config file else creates `args.lr_scheduler_type` Scheduler
    if (accelerator.state.deepspeed_plugin is None or
        "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config):
        # transformers lr scheduler
        # from transformers import get_scheduler
        # lr_scheduler = get_scheduler(
        #     name=args.lr_scheduler_type,
        #     optimizer=optimizer,
        #     num_warmup_steps=args.num_warmup_steps,
        #     num_training_steps=args.max_train_steps,
        # )
        lr_scheduler = get_scheduler(optimizer, **args.lr_scheduler.to_dict())
    else:
        # deepspeed lr_scheduler placeholder
        lr_scheduler = DummyScheduler(optimizer)
                                    #   total_num_steps=args.num_train_epochs, 
                                    #   warmup_num_steps=args.warm_up_epochs)
    
    logger.info("network params and training states are saved at [dim green]{}[/dim green]".format(args.save_model_path))

    # save checker and train process tracker
    if not args.regardless_metrics_save:
        metric_name = args.metric_name_for_save[args.task]
        check_order = args.metric_name_for_save.check_order[args.task]
        save_checker = BestMetricSaveChecker(metric_name=vars(metric_name) if isinstance(metric_name, NameSpace) else metric_name, 
                                             check_order=check_order)
    else:
        save_checker = None
    
    # start training
    if args.task == 'fusion':
        train_fn = train_fusion
    elif args.task == 'sharpening':
        train_fn = train_sharpening
    else:
        raise NotImplementedError(f'Task {args.task} is not implemented')
    
    train_fn(
        accelerator,
        network,
        optimizer,
        criterion,
        lr_scheduler,
        train_dl,
        val_dl,
        args.num_train_epochs,
        args.val_n_epoch,
        args.save_model_path,
        logger=logger,
        resume_epochs=1,
        check_save_fn=save_checker,
        args=args,
    )
        
    # logger finish
    if is_main_process() and logger is not None:
        logger.writer.close()
        
if __name__ == "__main__":
    args = get_main_args()
    
    try:
        main(args)
    except Exception as e:
        if is_main_process():
            EasyProgress.close_all_tasks()
            logger.error('An Error Occurred! Please check the stacks in log_file/running_traceback.log')
            logger.exception(e)
        else:
            logger.exception(e)
            raise RuntimeError(f'Process {os.getpid()} encountered an error: {e}')