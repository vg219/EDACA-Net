from functools import partial
import json
import os
from pathlib import Path

# os.environ["HF_DATASETS_OFFLINE"]="1"
# os.environ["HF_HUB_OFFLINE"]="1"

import requests
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoImageProcessor,
    AutoModelForUniversalSegmentation,
    MaskFormerImageProcessor,
)

import argparse
from argparse import Namespace
from typing import Any, Literal, Mapping
from tqdm import tqdm
from datasets import Dataset, DatasetDict, Image
import evaluate
from accelerate import Accelerator
import albumentations as A
from torchmetrics.segmentation import MeanIoU

import sys

sys.path.append("./")
from utils import easy_logger, catch_any_error


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrain-model-name", type=str, default="facebook/maskformer-swin-base-coco"
    )
    parser.add_argument(
        "--finetune-dir", "-f", type=str, help="Path to the fine-tuned model directory"
    )
    parser.add_argument("--dataset-name", "-d", type=str, choices=["llvip", "msrs"])
    parser.add_argument("--method-name", "-m", type=str)
    parser.add_argument(
        "--id2label-path", "-i", type=str, help="Path to the id2label json file"
    )
    parser.add_argument("--vi-dir", type=str)
    parser.add_argument("--ir-dir", type=str)
    parser.add_argument("--fused-dir", type=str)
    parser.add_argument("--mask-dir", type=str)
    parser.add_argument(
        "--img-ext", "-e", type=str, help="image extension to load in dataset"
    )
    parser.add_argument("--img-height", type=int, default=384)
    parser.add_argument("--img-width", type=int, default=384)
    parser.add_argument("--per_device_eval_batch_size", "-bs", type=int, default=28)
    parser.add_argument("--test-only-on-fused", action="store_true", default=False)
    parser.add_argument("--num_workers", "-nw", type=int, default=4)

    args = parser.parse_args()

    return args


def default_cfgs(args):
    default_dict = dict(
        finetune_dir="/Data3/cao/ZiHanCao/exps/panformer/downstreams/segmentation/runs",
        dataset_name="msrs",
        method_name="vi",
        img_ext="jpg",
        fused_dir="/Data3/cao/ZiHanCao/exps/panformer/visualized_img/panRWKV_v3/msrs",
        vi_dir="/Data3/cao/ZiHanCao/datasets/MSRS/test/vi",
        ir_dir="/Data3/cao/ZiHanCao/datasets/MSRS/test/ir",
        mask_dir="/Data3/cao/ZiHanCao/datasets/MSRS/test/Segmentation_labels",
        id2label_path="/Data3/cao/ZiHanCao/datasets/MSRS/MSRS-id2label.json",
    )

    # if key not provided in args
    args_dict = vars(args)
    for k, v in args_dict.items():
        if v is None and k in default_dict:
            args_dict[k] = default_dict[k]

    return Namespace(**args_dict)


def nested_cpu(tensors):
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_cpu(t) for t in tensors)
    elif isinstance(tensors, Mapping):
        return type(tensors)({k: nested_cpu(t) for k, t in tensors.items()})
    elif isinstance(tensors, torch.Tensor):
        return tensors.cpu().detach()
    else:
        return tensors


@torch.no_grad()
def evaluation_loop_semantic(
    model,
    image_processor: AutoImageProcessor,
    accelerator: Accelerator,
    dataloader,
    id2label,
    miou_method: Literal["torchmetrics", "hf_evaluate"] = "hf_evaluate",
):

    logger = easy_logger()

    if miou_method == "torchmetrics":
        logger.info("computing miou using torchmetrics")
        metric = MeanIoU(
            num_classes=len(id2label), per_class=True, include_background=False
        )
    else:
        logger.warning("the huggingface implememted miou is extremly slow")
        metric = evaluate.load(
            "mean_iou", num_labels=len(id2label), trust_remote_code=True
        )

    metrics: list[torch.Tensor] = []
    n_bs: list[int] = []

    for bi, inputs in tqdm(
        enumerate(dataloader, 1),
        total=len(dataloader),
        disable=not accelerator.is_local_main_process,
        leave=True,
    ):
        with torch.no_grad():
            outputs = model(**inputs)

        inputs = accelerator.gather_for_metrics(inputs)
        inputs = nested_cpu(inputs)

        outputs = accelerator.gather_for_metrics(outputs)
        outputs = nested_cpu(outputs)

        # Collect targets
        targets = []
        predictions = []
        target_sizes = []

        # Collect targets
        for masks in inputs["mask_labels"]:
            target_sizes.append(masks.shape[-2:])

        def to_seg_map(shape: str, outputs, target_sizes, n_classes):
            assert shape in ("hw", "chw")

            # Collect predictions
            seg_map = image_processor.post_process_semantic_segmentation(
                outputs,
                target_sizes=target_sizes,
            )
            if shape == "hw":
                return seg_map
            else:
                # to one-hot seg maps
                # shape [b, c, h, w]

                for i in range(len(seg_map)):
                    seg_map[i] = torch.nn.functional.one_hot(
                        seg_map[i].long(), num_classes=n_classes
                    ).permute(2, 0, 1)

            return seg_map

        _seg_map_shape = "chw" if miou_method == "torchmetrics" else "hw"
        seg_map = to_seg_map(
            _seg_map_shape, outputs, target_sizes, n_classes=len(id2label)
        )

        for target, class_label, output in zip(
            inputs["mask_labels"], inputs["class_labels"], seg_map
        ):  # target[1]
            # Update metric for batch targets and predictions
            if miou_method == "hf_evaluate":
                target = class_label[:, None, None].expand_as(target) * target
                target = target.sum(dim=0)
                output = output.to(dtype=target.dtype)

                targets.append(target.cpu().numpy())
                predictions.append(output.cpu().numpy())
            else:
                labels = torch.zeros(
                    len(id2label),
                    *target.shape[-2:],
                    dtype=torch.long,
                    device=target.device,
                )
                for i, cls in enumerate(class_label):
                    _t = target[i]
                    _t[_t == 0] = 0
                    _t[_t > 0] = 1
                    labels[cls] = _t

                targets.append(labels)

        if miou_method == "hf_evaluate":
            metric.add_batch(predictions=predictions, references=targets)
        else:
            seg_map = torch.stack(seg_map, dim=0)
            targets = torch.stack(targets, dim=0)
            metrics.append(metric(seg_map.long(), targets.long()))
            n_batches = targets.shape[0]
            n_bs.append(n_batches)

        # if bi > 3:
        #     break

    # Compute metrics
    if miou_method == "torchmetrics":
        avg_metric = []
        for m, bs in zip(metrics, n_bs):  # metric of one batch
            batched_metric = m * bs
            avg_metric.append(batched_metric)
        avg_metric = torch.stack(avg_metric).sum(dim=0) / sum(n_bs)

        metric_values = avg_metric
        metric_names = [cls for cls in id2label.values() if cls != "background"]
        if metric.include_background:
            metric_names = ["background"] + metric_names

        metrics = {}
        for name, value in zip(metric_names, metric_values.tolist()):
            metrics[f"{name}_iou"] = value
    else:
        metrics = metric.compute(
            num_labels=len(id2label), ignore_index=0, reduce_labels=False
        )

        for i, name in id2label.items():
            metrics[f"{name}_iou"] = metrics["per_category_iou"][i]
        metrics.pop("per_category_iou")

    return metrics


def create_msrs_seg_hf_dataset(args):
    logger = easy_logger()

    # create dataset
    test_fused_dir = Path(args.fused_dir)
    test_vi_dir = Path(args.vi_dir)
    test_ir_dir = Path(args.ir_dir)
    test_mask_dir = Path(args.mask_dir)

    test_mask_files = list(test_mask_dir.glob("*.png"))  # mask files are in png format
    logger.info("test files: {}".format(len(test_mask_files)))

    test_fused_files = list(test_fused_dir.glob(f"*.*"))
    test_fused_files = [str(f) for f in test_fused_files]

    test_mask_files = [str(f) for f in test_mask_files]

    def create_dataset(image_paths, label_paths):
        dataset = Dataset.from_dict(
            {"image": sorted(image_paths), "annotation": sorted(label_paths)}
        )
        dataset = dataset.cast_column("image", Image())
        dataset = dataset.cast_column("annotation", Image())

        return dataset

    dataset = DatasetDict()

    fused_test_ds = create_dataset(test_fused_files, test_mask_files)
    dataset.update({"test_fused": fused_test_ds})
    if not args.test_only_on_fused:
        # vi imgs
        test_vi_files = list(test_vi_dir.glob(f"*.{args.img_ext}"))
        assert len(test_vi_files) == len(
            test_mask_files
        ), "number of ir images and masks do not match"
        test_vi_files = [str(f) for f in test_vi_files]
        dataset.update({"test_vi": create_dataset(test_vi_files, test_mask_files)})

        # ir imgs
        test_ir_files = list(test_ir_dir.glob(f"*.{args.img_ext}"))
        assert len(test_ir_files) == len(
            test_mask_files
        ), "number of ir images and masks do not match"
        test_ir_files = [str(f) for f in test_ir_files]
        dataset.update({"test_ir": create_dataset(test_ir_files, test_mask_files)})

    return dataset


def augment_and_transform_batch(
    examples: Mapping[str, Any], transform, image_processor: AutoImageProcessor
):
    logger = easy_logger()

    batch = {
        "pixel_values": [],
        "mask_labels": [],
        "class_labels": [],
    }

    for pil_image, pil_annotation in zip(examples["image"], examples["annotation"]):
        image = np.array(pil_image)
        pil_annotation = np.array(
            pil_annotation
        )  # [..., np.newaxis].repeat(2, axis=-1)
        semantic_and_instance_masks = pil_annotation  # [..., :2]

        # Apply augmentations
        output = transform(image=image, mask=semantic_and_instance_masks)

        aug_image = output["image"]
        aug_semantic_and_instance_masks = output["mask"]
        aug_instance_mask = aug_semantic_and_instance_masks  # [..., 1]

        # Create mapping from instance id to semantic id
        # unique_semantic_id_instance_id_pairs = np.unique(aug_semantic_and_instance_masks.reshape(-1, 2), axis=0)
        # instance_id_to_semantic_id = {
        #     instance_id: semantic_id for semantic_id, instance_id in unique_semantic_id_instance_id_pairs
        # }

        # print(aug_image.shape, aug_instance_mask.shape)
        # Apply the image processor transformations: resizing, rescaling, normalization
        try:
            model_inputs = image_processor(
                images=[aug_image],
                segmentation_maps=[aug_instance_mask],
                # instance_id_to_semantic_id=instance_id_to_semantic_id,
                return_tensors="pt",
            )
        except Exception as e:
            logger.warning("met some loading errors, skip this image")
            continue

        batch["pixel_values"].append(model_inputs.pixel_values[0])
        batch["mask_labels"].append(model_inputs.mask_labels[0])
        batch["class_labels"].append(model_inputs.class_labels[0])

    return batch


def collate_fn(examples):
    batch = {}
    batch["pixel_values"] = torch.stack(
        [example["pixel_values"] for example in examples]
    )
    batch["class_labels"] = [example["class_labels"] for example in examples]
    batch["mask_labels"] = [example["mask_labels"] for example in examples]
    if "pixel_mask" in examples[0]:
        batch["pixel_mask"] = torch.stack(
            [example["pixel_mask"] for example in examples]
        )
    return batch


def main():
    # get args
    args = args_parse()
    args = default_cfgs(args)

    accelerator = Accelerator()
    logger = easy_logger()

    # load model
    logger.info("Loading fine-tuned model")
    logger.info(f"model pretrained on {args.pretrain_model_name}")
    logger.info(f"model finetuned on {args.finetune_dir}")
    
    id2label = json.load(open(args.id2label_path, "r"))
    id2label = {int(k): v for k, v in id2label.items()}
    label2id = {v: k for k, v in id2label.items()}

    processor = AutoImageProcessor.from_pretrained(
        args.pretrain_model_name,
        do_resize=True,
        size={"height": args.img_height, "width": args.img_width},
        do_reduce_labels=False,
        ignore_mismatched_sizes=True
    )
    model = AutoModelForUniversalSegmentation.from_pretrained(
        args.finetune_dir,
        label2id=label2id,
        id2label=id2label,
    )
    model = accelerator.prepare(model)
    model = model.eval()

    # load dataset
    test_ds = create_msrs_seg_hf_dataset(args)
    logger.info("load dataset done")

    logger.info(f"testing method: {args.method_name}")

    stages = ["test_fused", "test_vi"] if not args.test_only_on_fused else ["test_fused"]
    for test_stage in stages:
        # logger.info('testing on {}\n\n'.format(test_stage))
        ds = test_ds[test_stage]

        test_transform_batch = partial(
            augment_and_transform_batch,
            transform=A.Compose([A.NoOp()]),
            image_processor=processor,
        )

        with accelerator.main_process_first():
            ds = ds.with_transform(test_transform_batch)

        test_dl = DataLoader(
            ds,
            shuffle=False,
            batch_size=args.per_device_eval_batch_size,
            num_workers=args.num_workers,
            persistent_workers=True if args.num_workers > 0 else False,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        test_dl = accelerator.prepare(test_dl)

        logger.info("start evaluation")

        metrics = evaluation_loop_semantic(
            model, processor, accelerator, test_dl, id2label
        )
        logger.info("==" * 20 + "\n")
        logger.info("Evaluation results:")
        results = f"{test_stage}\n"
        for k, v in metrics.items():
            results += "{}: {}\n".format(k, v)

        logger.info(results)
        logger.info("==" * 20 + "\n")


if __name__ == "__main__":
    with catch_any_error():
        main()
