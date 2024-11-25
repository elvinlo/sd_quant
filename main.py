import sys
sys.path.append('./')

import argparse, os, sys, glob, importlib, csv
from datetime import datetime
import shutil
import yaml
import copy
import numpy as np
import pandas as pd

from packaging import version
from omegaconf import OmegaConf
from torch.utils.data import random_split, DataLoader, Dataset, Subset
from functools import partial
from PIL import Image
from einops import rearrange

import torch
import torch.backends.cudnn as cudnn
import torchvision
from torch.utils.tensorboard import SummaryWriter
import pytorch_lightning as pl

from ldm.data.base import Txt2ImgIterableBaseDataset
from ldm.data.coco import CocoImagesAndCaptionsTrain2017, CocoImagesAndCaptionsValidation2017
from ldm.util import instantiate_from_config, str_or_int
from ldm.eval.clip_score import CLIPScore
from ldm.eval.fid_score import FrechetInceptionDistance
        

parser = argparse.ArgumentParser()

parser.add_argument("--config", nargs="*", metavar="base_config.yaml", help="configs path", default=list())
parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint path')

parser.add_argument('--evaluate', action='store_true', help='evaluate model on testing set')
parser.add_argument('--batch_size', default=None, type=int, help='batch size per GPU (overrides configs in yaml file)')
parser.add_argument("--train_samples", type=int, help="training set size", default=None)
parser.add_argument("--val_samples", type=int, help="validation set size", default=500)
parser.add_argument("--eval_samples", type=int, help="evaluation set size", default=None)
parser.add_argument("--sample_steps", type=int, default=50, help="number of DDIM sampling steps for generating images during validation and testing")
parser.add_argument("--cfg_scale", type=float, default=9.0, help="unconditional guidance scale for validation and testing")
parser.add_argument('--monitor_fid', action='store_true', help='monitor FID for accuracy signal instead of CLIP score')

parser.add_argument('--disable_xformers', action='store_true', help='disable xformers and use vanilla attention implementation instead')
parser.add_argument('--plot', action='store_true', help='visualize model during evaluation loop')
parser.add_argument('--plot_only', action='store_true', help='exit after generating the first plot')
parser.add_argument("--plot_layer", default=None, type=str_or_int, help="when plotting, plot specified layers; provide layer number (e.g., 4), comma-separated list (e.g., 1,2,3,4), or a range (e.g., 1-4)")
parser.add_argument("--plot_log", action='store_true', help="use log-scale when plotting")
parser.add_argument("--save_images", action='store_true', help="save images during evaluation")
parser.add_argument('--print_freq', default=50, type=int, help='print every k batches')
parser.add_argument('--print_model', action='store_true', help='print model modules')
parser.add_argument("--tag", default="", type=str, help="experiment identifier (string to prepend to save directory names")
parser.add_argument('--activations', action='store_true', help='collect activation statistics')
parser.add_argument('--dynamic_quantize', action='store_true', help='dynamic quantization of model')
parser.add_argument('--static_quantize', action='store_true', help='static quantization of model')
parser.add_argument('--quantize_bitwidth', default=8, type=int, help='number of bits to quantize weights to (not activations)')



def get_experiment_str(args):
    experiment_str = args.tag + '_' + datetime.now().strftime('%Y-%m-%d_%H-%M/')
    return experiment_str


class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()

    dataset = worker_info.dataset
    worker_id = worker_info.id

    if isinstance(dataset, Txt2ImgIterableBaseDataset):
        split_size = dataset.num_records // worker_info.num_workers
        # reset num_records to the true number to retain reliable length information
        dataset.sample_ids = dataset.valid_ids[worker_id * split_size:(worker_id + 1) * split_size]
        current_id = np.random.choice(len(np.random.get_state()[1]), 1)
        return np.random.seed(np.random.get_state()[1][current_id] + worker_id)
    else:
        return np.random.seed(np.random.get_state()[1][0] + worker_id)


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None, predict=None,
                 wrap=False, num_workers=None, shuffle_test_loader=False, use_worker_init_fn=False,
                 shuffle_val_dataloader=False):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.use_worker_init_fn = use_worker_init_fn
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(self._val_dataloader, shuffle=shuffle_val_dataloader)
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)
        if predict is not None:
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = self._predict_dataloader
        self.wrap = wrap

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self):
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False if is_iterable_dataset else True,
                          worker_init_fn=init_fn)

    def _val_dataloader(self, shuffle=False):
        if isinstance(self.datasets['validation'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          worker_init_fn=init_fn,
                          shuffle=shuffle)

    def _test_dataloader(self, shuffle=False):
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None

        # do not shuffle dataloader for iterable dataset
        shuffle = shuffle and (not is_iterable_dataset)

        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn, shuffle=shuffle)

    def _predict_dataloader(self, shuffle=False):
        if isinstance(self.datasets['predict'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["predict"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn)


def evaluate_accuracy(model, dataloader=None, args=None):
    print(args.activations)
    model.eval()
    save_dir = f'./results/{args.experiment_str}/'
    
    if args.monitor_fid:
        acc_metric = FrechetInceptionDistance().set_dtype(torch.float64).to(args.device)
    else:
        acc_metric = CLIPScore(model_name_or_path="laion/CLIP-ViT-g-14-laion2B-s12B-b42K").to(args.device)

    if args.evaluate and args.save_images and (args.accelerator is None or args.accelerator.is_main_process):
        images_dir = save_dir + 'images/'
        os.makedirs(images_dir, exist_ok=True)

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            # check if we want to plot this batch
            plot = args.plot and (i < 20)
            activations = args.activations
            
            batch[model.first_stage_key] = batch[model.first_stage_key].to(args.device)
            samples, images, prompts = model.sample_images(batch, batch_idx=i, epoch=None, save_dir=save_dir, plot=plot, activations = activations)

            if (plot and args.plot_only) or activations:
                exit()
            
            # save the first 10 images, and then save one image every 100 iterations
            if args.evaluate and args.save_images and (args.accelerator is None or args.accelerator.is_main_process) and (i < 10 or i % 100 == 0):
                filename = f"{i:05}.png"
                path = os.path.join(images_dir, filename)
                sample = rearrange(samples[0].detach().cpu().numpy(), 'c h w -> h w c')
                Image.fromarray(sample.astype(np.uint8)).save(path)

            # if args.accelerator is not None:
            #     samples = args.accelerator.gather_for_metrics(samples).to(args.device)
            #     prompts = args.accelerator.gather_for_metrics(prompts)
            if args.monitor_fid:
                acc_metric.update(images, real=True)
                acc_metric.update(samples, real=False)
            else:
                acc_metric.update(samples, prompts)

            if args.print_freq != 0 and i != 0 and i % args.print_freq == 0:
                acc_so_far = acc_metric.compute().detach().cpu().numpy()
                print(f'\tbatch {i}/{len(dataloader)}: test acc {acc_so_far:.2f}')

    test_acc = acc_metric.compute().detach().cpu().numpy()
    
    return test_acc


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("No GPUs found!")
        exit()
    
    args, unknown = parser.parse_known_args()

    os.environ["USE_XFORMERS"] = '0' if args.disable_xformers else '1' # TODO: implement option to disable xformers
    os.environ["ATTN_PRECISION"] = "fp32"

    # TODO: try enforcing deterministic behavior by setting seeds and using torch.use_deterministic_algorithms

    # TODO: we should support multi-GPU runs with HF accelerate or torchrun once we get there
    # if args.accelerate:
    #     from accelerate import Accelerator
    #     args.accelerator = Accelerator()
    #     args.device = args.accelerator.device
    #     args.num_gpus = args.accelerator.state.num_processes
    #     os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # else:
    #     args.num_gpus = 1
    
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    args.accelerator = None

    # init configs
    configs = [OmegaConf.load(cfg) for cfg in args.config]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    
    # update config from parsed arguments
    if args.train_samples is not None:
        config.data.params.train.params.dataset_size = args.train_samples
    if args.val_samples is not None:
        config.data.params.validation.params.dataset_size = args.val_samples
    if args.eval_samples is not None:
        config.data.params.test.params.dataset_size = args.eval_samples

    config.data.params.batch_size = args.batch_size if args.batch_size is not None else config.data.params.batch_size
    # args.lr = args.lr if args.lr is not None else config.model.base_learning_rate
    # config.model.base_learning_rate = args.lr

    args.config = config
    batch_size, base_lr = config.data.params.batch_size, config.model.base_learning_rate

    # model
    print(f'\nCreating Stable Diffusion v2.1 model\n')
    
    model = instantiate_from_config(config.model)
    model.learning_rate = base_lr

    # data
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()
    print("\nLoading datasets:")
    try:
        for k in data.datasets:
            print(f"  - {k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")
    except:
        print("datasets not yet initialized.")
    
    train_dataloader, val_dataloader, test_dataloader = data._train_dataloader(), data._val_dataloader(), data._test_dataloader()

    # TODO: quantization
    ### ----- Dynamic Quantization ------
    def quantize_weight(tensor, num_bits=8):
        qmin = -(2 ** (num_bits - 1))
        qmax = (2 ** (num_bits - 1)) - 1

        max_val = tensor.abs().max()
        if max_val == 0:
            scale = 1.0
        else:
            scale = max_val / qmax

        # Quantize
        q_x = (tensor / scale).round().clamp(qmin, qmax)
        # Dequantize
        dq_x = q_x * scale

        return dq_x

    def quantize_model_weights(model, num_bits=8):
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                with torch.no_grad():
                    module.weight.data = quantize_weight(module.weight.data, num_bits)
                    if module.bias is not None:
                        module.bias.data = quantize_weight(module.bias.data, num_bits)

    def quantize_activation(tensor, num_bits=8):
        qmin = 0
        qmax = (2 ** num_bits) - 1

        min_val = tensor.min()
        max_val = tensor.max()
        if min_val == max_val:
            scale = 1.0
            zero_point = 0
        else:
            scale = (max_val - min_val) / (qmax - qmin)
            zero_point = qmin - min_val / scale

        # Quantize
        q_x = ((tensor / scale) + zero_point).round().clamp(qmin, qmax)
        # Dequantize
        dq_x = (q_x - zero_point) * scale

        return dq_x

    def activation_quantization_hook(module, input, output):
        return quantize_activation(output, num_bits=8)

    def register_activation_quantization_hooks(model):
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                module.register_forward_hook(activation_quantization_hook)

    ### ----- Static Quantization ------
    def quantize_weight_static(tensor, max, num_bits=8):
        qmin = -(2 ** (num_bits - 1))
        qmax = (2 ** (num_bits - 1)) - 1

        max_val = max
        if max_val == 0:
            scale = 1.0
        else:
            scale = max_val / qmax

        # Quantize
        q_x = (tensor / scale).round().clamp(qmin, qmax)
        # Dequantize
        dq_x = q_x * scale

        return dq_x

    def quantize_model_weights_static(model, df, num_bits=8):
        layer_num = 0;
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                layer_max = max(np.abs(df[df["layer_num"] == layer_num]["max"].values[0]),
                                np.abs(df[df["layer_num"] == layer_num]["min"].values[0]))
                with torch.no_grad():
                    module.weight.data = quantize_weight_static(module.weight.data, layer_max, num_bits)
                    if module.bias is not None:
                        module.bias.data = quantize_weight_static(module.bias.data, layer_max, num_bits)
            layer_num += 1

    def quantize_activation_static(tensor, num_bits=8):
        qmin = 0
        qmax = (2 ** num_bits) - 1

        min_val = tensor.min()
        max_val = tensor.max()
        if min_val == max_val:
            scale = 1.0
            zero_point = 0
        else:
            scale = (max_val - min_val) / (qmax - qmin)
            zero_point = qmin - min_val / scale

        # Quantize
        q_x = ((tensor / scale) + zero_point).round().clamp(qmin, qmax)
        # Dequantize
        dq_x = (q_x - zero_point) * scale

        return dq_x

    def activation_quantization_hook_static(module, input, output):
        return quantize_activation_static(output, num_bits=8)

    def register_activation_quantization_hooks_static(model):
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                module.register_forward_hook(activation_quantization_hook_static)
    
    # load from checkpoint
    if args.checkpoint is None or not os.path.isfile(args.checkpoint):
        print("\nNo checkpoint found! Proceeding without checkpoint.")
    else:
        print(f'\nLoading model from {args.checkpoint}\n')
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"], strict=False)

    if args.print_model:
        with open("layers.txt", 'w') as f:
            layer_num = 0
            for name, module in model.model.named_modules():
                f.write(f"{layer_num}: {name}\n")
                layer_num += 1
        # print(model)

    args.experiment_str = get_experiment_str(args)

    args.calibrating_range = False # this tells DDIM sampler when to print clip_vals
    model.args = args
    
    # if args.accelerate:
    #     model, train_dataloader, val_dataloader, test_dataloader = args.accelerator.prepare(model, train_dataloader, val_dataloader, test_dataloader)
    #     args.train_dataloader_len = len(train_dataloader)
    #     args.val_dataloader_len = len(val_dataloader)
    #     args.test_dataloader_len = len(test_dataloader)

    # Apply quantization before moving the model to GPU
    if args.dynamic_quantize:
        print(f"Applying dynamic {args.quantize_bitwidth}-bit quantization (W{args.quantize_bitwidth}A8) to the model's Linear and Conv2d layers...")
        quantize_model_weights(model.model, num_bits=args.quantize_bitwidth)
        register_activation_quantization_hooks(model.model)
    elif args.static_quantize:
        print(f"Applying static {args.quantize_bitwidth}-bit quantization (W{args.quantize_bitwidth}A8) to the model's Linear and Conv2d layers...")
        df = pd.read_csv('static_activation_stats/total_activation_statistics.csv')
        quantize_model_weights_static(model.model, df=df, num_bits=args.quantize_bitwidth)
        register_activation_quantization_hooks_static(model.model)


    model = model.to(args.device)

    if args.evaluate:
        iter_str = f' for {args.eval_samples} samples' if args.eval_samples is not None else ''
        print(f'\n\n\nEvaluating model on COCO validation dataset{iter_str}\n\n')
        test_acc = evaluate_accuracy(model, dataloader=test_dataloader, args=args)
        print(f'\nEvaluation accuracy: {test_acc:.2f}')