

import numpy as np  # sometimes needed to avoid mkl-service error
import sys

from Datamodel.Dictionary import Dictionary
from Datamodel.MolDataset import MolPretrainDataset
from Datamodel.MolecularDataModule import MolecularDataModule
from Datamodel.utils import LoadFromCheckpoint, LoadFromFile, number, save_argparse
from model import output_modules
from model.module import LNNP
from model.utils import rbf_class_mapping, act_class_mapping

sys.path.append(sys.path[0]+'/..')
# from datasets.Dictionary import Dictionary

import torch
import os
import argparse
import logging
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.strategies import DDPStrategy
# from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.utilities import rank_zero_only
from pathlib import Path
import wandb
os.environ['CUDA_VISIBLE_DEVICES']='0'
# cuda_version="12.1"
# os.environ['CUDA_PATH']=f'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1'
# os.environ['PATH']=os.environ['CUDA_PATH'] + "/bin" + os.environ['PATH']
# os.environ['PL_TORCH_DISTRIBUTED_BACKEND']='gloo'
os.environ['WANDB_SILENT']= "true"
import warnings

# 忽略特定类型的警告
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)




def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--mode', default='finetune2''classification', type=str,help='pretrain1''pretrain2''finetune1''finetune2''classification''regression')
    parser.add_argument('--load-model', action=LoadFromCheckpoint, help='Restart training using a model checkpoint')  # keep first
    parser.add_argument('--conf', '-c', type=open, action=LoadFromFile, help='Configuration yaml file')  # keep second
    parser.add_argument('--num-epochs', default=3, type=int, help='number of epochs')
    parser.add_argument('--num-steps', default=400000, type=int, help='Maximum number of gradient steps.')
    parser.add_argument('--batch-size', default=8, type=int, help='batch size')
    parser.add_argument('--inference-batch-size', default=8, type=int, help='Batchsize for validation and tests.')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--lr-schedule', default="reduce_on_plateau", type=str, choices=['cosine', 'cosine_warmup','reduce_on_plateau'], help='Learning rate schedule.')
    parser.add_argument('--lr-patience', type=int, default=5, help='Patience for lr-schedule. Patience per eval-interval of validation')
    parser.add_argument('--lr-min', type=float, default=1.0e-07, help='Minimum learning rate before early stop')
    parser.add_argument('--lr-factor', type=float, default=0.8, help='Minimum learning rate before early stop')
    parser.add_argument('--lr-warmup-steps', type=int, default=10000, help='How many steps to warm-up over. Defaults to 0 for no warm-up')
    parser.add_argument('--lr-cosine-length', type=int, default=400000, help='Cosine length if lr_schedule is cosine.')
    parser.add_argument('--early-stopping-patience', type=int, default=10, help='Stop training after this many epochs without improvement')
    parser.add_argument('--ema-alpha-y', type=float, default=1.0, help='The amount of influence of new losses on the exponential moving average of y')
    parser.add_argument('--ema-alpha-dy', type=float, default=1.0, help='The amount of influence of new losses on the exponential moving average of dy')
    parser.add_argument('--ngpus', type=int, default=1, help='Number of GPUs, -1 use all available. Use CUDA_VISIBLE_DEVICES=1, to decide gpus')
    parser.add_argument('--num-nodes', type=int, default=1, help='Number of nodes')
    parser.add_argument('--precision', type=int, default=32, choices=[16, 32], help='Floating point precision')
    parser.add_argument('--log-dir', '-l', default='./logs', help='log file')
    parser.add_argument('--splits', default=None, help='Npz with splits idx_train, idx_val, idx_test')
    parser.add_argument('--train-size', type=number, default=0.8, help='Percentage/number of samples in training set (None to use all remaining samples)')
    parser.add_argument('--val-size', type=number, default=0.1, help='Percentage/number of samples in validation set (None to use all remaining samples)')
    parser.add_argument('--test-size', type=number, default=0.1, help='Percentage/number of samples in test set (None to use all remaining samples)')
    parser.add_argument('--test-interval', type=int, default=1, help='Test interval, one test per n epochs (default: 10)')
    parser.add_argument('--save-interval', type=int, default=1, help='Save interval, one save per n epochs (default: 10)')
    parser.add_argument('--seed', type=int, default=123, help='random seed (default: 1)')
    parser.add_argument('--distributed-backend', default='ddp', help='Distributed backend: dp, ddp, ddp2')
    parser.add_argument('--num-workers', type=int, default=0, help='Number of workers for data prefetch')############
    parser.add_argument('--redirect', type=bool, default=False, help='Redirect stdout and stderr to log_dir/log')
    parser.add_argument('--wandb-notes', default="", type=str, help='Notes passed to wandb experiment.')
    parser.add_argument('--job-id', default="1", type=str, help='Job ID. If auto, pick the next available numeric job id.')
    parser.add_argument('--pretrained-model', default=None, type=str, help='Pre-trained weights checkpoint.')
    parser.add_argument('--weight-decay', type=float, default=1e-2, help='Weight decay strength')
    # dataset specific
    parser.add_argument('--dataset', default='bbbp', type=str, help='Name of the torch_geometric dataset')#choices=datasets.__all__
    parser.add_argument('--dataset-root', default='./dataset/BBBP', type=str, help='Data storage directory (not used if dataset is "CG")')
    parser.add_argument('--dataset_arg', default=None, type=str, help='Additional dataset argument, e.g. target property for QM9 or molecule for MD17')
    parser.add_argument('--position-noise-scale', default=0.02, type=float, help='Scale of Gaussian noise added to positions.')

    # parser.add_argument('--denoising-only', type=bool, default=True, help='If the task is denoising only (then val/test datasets also contain noise).')
    
    parser.add_argument('--use-dataset-md17', type=bool, default=False, help='use md17 as the eval dataset.')
    parser.add_argument('--derivative', default=False, type=bool,
                        help='If true, take the derivative of the prediction w.r.t coordinates')
    # model architecture
    parser.add_argument('--model', type=str, default='equivariant-transformer',  help='Which model to train')  #choices=models.__all__,
    parser.add_argument('--output-model', type=str, default='Scalar', choices=output_modules.__all__, help='The type of output model')
    parser.add_argument('--mask-model', type=str, default='Molformer')
    parser.add_argument('--output-model-noise', type=str, default='VectorOutput', choices=output_modules.__all__ + ['VectorOutput'], help='The type of output model for denoising')

    # architectural args
    parser.add_argument('--embedding-dimension', type=int, default=256, help='Embedding dimension')
    parser.add_argument('--num-layers', type=int, default=6, help='Number of interaction layers in the model')
    parser.add_argument('--num-rbf', type=int, default=32, help='Number of radial basis functions in model')
    parser.add_argument('--activation', type=str, default='silu', choices=list(act_class_mapping.keys()), help='Activation function')
    parser.add_argument('--rbf-type', type=str, default='expnorm', choices=list(rbf_class_mapping.keys()), help='Type of distance expansion')
    parser.add_argument('--trainable-rbf', type=bool, default=False, help='If distance expansion functions should be trainable')
    parser.add_argument('--neighbor-embedding', type=bool, default=True, help='If a neighbor embedding should be applied before interactions')
    parser.add_argument('--aggr', type=str, default='add', help='Aggregation operation for CFConv filter output. Must be one of \'add\', \'mean\', or \'max\'')

    # Transformer specific
    parser.add_argument('--distance-influence', type=str, default='both', choices=['keys', 'values', 'both', 'none'], help='Where distance information is included inside the attention')
    parser.add_argument('--attn-activation', default='silu', choices=list(act_class_mapping.keys()), help='Attention activation function')
    parser.add_argument('--num-heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--layernorm-on-vec', type=str, default=None, choices=['whitened'], help='Whether to apply an equivariant layer norm to vec features. Off by default.')

    # other args
    parser.add_argument('--cutoff-lower', type=float, default=0.0, help='Lower cutoff in model')
    parser.add_argument('--cutoff-upper', type=float, default=5.0, help='Upper cutoff in model')
    parser.add_argument('--atom-filter', type=int, default=-1, help='Only sum over atoms with Z > atom_filter')
    parser.add_argument('--max-z', type=int, default=128, help='Maximum atomic number that fits in the embedding matrix')
    parser.add_argument('--max-num-neighbors', type=int, default=32, help='Maximum number of neighbors to consider in the network')
    parser.add_argument('--standardize', type=bool, default=False, help='If true, multiply prediction by dataset std and add mean')
    parser.add_argument('--reduce-op', type=str, default='add', choices=['add', 'mean'], help='Reduce operation to apply to atomic predictions')
    # fmt: on
    parser.add_argument('--reduce-lr-when-bad', type=bool, default=False, help='reduce lr when the val_loss is bad')
    parser.add_argument('--input-data-norm-type', type=str, default='minmax', choices=['minmax', 'log', 'log10', 'None'], help='which type of norm method do you want for spectra data')
    # unimol
    parser.add_argument('--max_atoms', default=128,type=int )
    parser.add_argument('--dict_path', default="./dataset/dict2.txt", type=str)
    parser.add_argument('--pad_idx', default=0, type=int)
    parser.add_argument("--noise-type", type=str, default="uniform",
                        choices=["trunc_normal", "uniform", "normal", "none"],
                        help="Type of noise to add to coordinates")
    parser.add_argument("--noise", type=float, default=1.0, help="Standard deviation of the noise")
    parser.add_argument("--mask-prob", type=float, default=0.35, help="Probability of masking an atom")
    parser.add_argument("--encoder-layers", type=int, default=6)
    parser.add_argument("--encoder-embed-dim", type=int, default=256)
    parser.add_argument("--encoder-ffn-embed-dim", type=int, default=256)
    parser.add_argument("--encoder-attention-heads", type=int, default=256)
    parser.add_argument("--activation-fn", type=str,default='gelu')

    parser.add_argument("--emb-dropout", type=float,default=0.15 )
    parser.add_argument("--dropout", type=float,default=0.15, help="dropout probability")
    parser.add_argument("--attention-dropout", type=float, default=0.15)
    parser.add_argument("--activation-dropout", type=float,default=0.15 )
    # parser.add_argument("--pooler-dropout", type=float,default=0.0 )
    parser.add_argument("--max-seq-len", type=int,default=256, help="number of positional embeddings to learn")
    # parser.add_argument("--post-ln", type=bool, default=False,help="use post layernorm or pre layernorm")
    parser.add_argument("--masked-token-loss", type=float, default=1)
    parser.add_argument("--masked-dist-loss", type=float, default=1)
    parser.add_argument("--masked-coord-loss", type=float, default=3)
    parser.add_argument("--x-norm-loss", type=float, default=0.01)
    parser.add_argument("--delta-pair-repr-norm-loss", type=float,  default=0.01)

    #weight
    parser.add_argument("--mask_weight", type=float, default=0.02)
    parser.add_argument('--denoising-weight', default=0.13, type=float,
                        help='Weighting factor for denoising in the loss function.')
    parser.add_argument('--classweight', default=0.85, type=float, help='Weighting factor for classfication')
    args = parser.parse_args()

    if args.job_id == "auto":
        assert len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) == 1, "Might be problematic with DDP."
        if Path(args.log_dir).exists() and len(os.listdir(args.log_dir)) > 0:
            next_job_id = str(max([int(x.name) for x in Path(args.log_dir).iterdir() if x.name.isnumeric()])+1)
        else:
            next_job_id = args.datasets
        args.job_id = next_job_id
        # args.job_id = "0"

    args.log_dir = str(Path(args.log_dir, args.job_id))
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    if args.redirect:
        sys.stdout = open(os.path.join(args.log_dir, "log"), "w")
        sys.stderr = sys.stdout
        logging.getLogger("pytorch_lightning").addHandler(
            logging.StreamHandler(sys.stdout)
        )

    if args.inference_batch_size is None:
        args.inference_batch_size = args.batch_size

    save_argparse(args, os.path.join(args.log_dir, "input.yaml"), exclude=["conf"])

    return args


def main():
    args = get_args()
    pl.seed_everything(args.seed, workers=True)
    # print(args)
    # 创建临时 trainer（仅用于获取 rank）

    if args.dict_path is not None:
       vocab = Dictionary.load(args.dict_path)
    else:
       vocab =None

    data= MolecularDataModule(args,args.mode,vocab)
    data.prepare_data()
    data.setup("fit")

    # if args.mode != 'pretrain1' or args.mode != 'pretrain2':  # 如果不是预训练，设置任务
    if args.mode in ["pretrain1" ,"pretrain2","pretrain3"]:
        args.num_tasks = None
        args.loss_type = None
        # model = LNNP(args, args.mode, vocab, None, mean=data.mean, std=data.std)
        model = LNNP(
            args,
            mode=args.mode,
            vocab=vocab,
            loss_type=None,
            mean=data.mean,
            std=data.std
        )
    else:
        args.num_tasks = data.num_tasks
        args.loss_type = data.loss_type
        model = LNNP(
            args,
            mode=args.mode,
            vocab=vocab,
            loss_type=data.loss_type,
            mean=data.mean,
            std=data.std
        )
    # else:
    #     args.num_tasks = data.num_tasks
    #     args.loss_type = data.loss_type
    #     model = LNNP(args, args.mode, vocab, args.loss_type, mean=data.mean, std=data.std)

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.log_dir,
        monitor="val_loss",
        save_top_k=10,  # 10,  # -1 to save all
        every_n_epochs=1,
        # period=args.save_interval,
        filename="{step}-{epoch}-{val_loss:.4f}-{test_loss:.4f}-{train_per_step:.4f}",
        save_last=True,
    )
    early_stopping = EarlyStopping("val_loss", patience=args.early_stopping_patience,mode='min',
                                   min_delta=0.001,
                                   verbose=True,
                                   check_on_train_epoch_end=False  # 只在验证结束时检查
                                   )

    tb_logger = pl.loggers.TensorBoardLogger(
        args.log_dir, name="tensorbord", version="", default_hp_metric=False
    )
    csv_logger = CSVLogger(args.log_dir, name="", version="")
    # wandb_logger = WandbLogger(
    #     name=args.job_id,
    #     project="MolSpectra-0122",
    #     notes=args.wandb_notes,
    #     settings=wandb.Settings(start_method="spawn", code_dir="."),
    # )

    # @rank_zero_only
    # def log_code():
    #     wandb_logger.experiment # runs wandb.init, so then code can be logged next
    #     wandb.run.log_code(".", include_fn=lambda path: path.endswith(".py") or path.endswith(".yaml"))
    #
    # log_code()
    #
    # ddp_plugin = None
    if "ddp" in args.distributed_backend:
        # ddp_plugin = DDPPlugin(find_unused_parameters=True, num_nodes=args.num_nodes)
        strategy = DDPStrategy(
            find_unused_parameters=True,
            # num_nodes=args.num_nodes
        )
    else:
        strategy = "auto"
    # trainer = pl.Trainer(
    #     max_epochs=args.num_epochs,
    #     max_steps=args.num_steps,
    #     gpus=args.ngpus,
    #     num_nodes=args.num_nodes,
    #     accelerator=args.distributed_backend,
    #     default_root_dir=args.log_dir,
    #     auto_lr_find=False,
    #     resume_from_checkpoint=args.load_model,
    #     # strategy="ddp",  # 使用 strategy 参数指定分布式训练策略
    #     # accelerator="auto",
    #     # callbacks=[early_stopping, checkpoint_callback],
    #     callbacks=[checkpoint_callback],
    #     logger=[tb_logger, csv_logger],
    #     # reload_dataloaders_every_n_epochs=False,
    #     reload_dataloaders_every_epoch=False,
    #     precision=args.precision,
    #     plugins=[ddp_plugin],
    # )
    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        max_steps=args.num_steps,
        accelerator="cuda",
        devices=1,
        # strategy=strategy,
        accumulate_grad_batches=4,
        num_nodes=args.num_nodes,
        default_root_dir=args.log_dir,
        # auto_lr_find=False,
        # ckpt_path=args.load_model,  # 使用 ckpt_path 替代 resume_from_checkpoint
        callbacks=[checkpoint_callback,early_stopping],
        logger=[tb_logger, csv_logger],
        reload_dataloaders_every_n_epochs=False,  # 替代 reload_dataloaders_every_epoch
        precision=args.precision,
        detect_anomaly=True,  # 检测数值异常
        gradient_clip_val=1.0,  # 添加梯度裁剪
    )

    trainer.fit(model, data,ckpt_path=r"E:\MolSPE\logs\17\last.ckpt") #,ckpt_path=args.load_model
    # best_model_path = checkpoint_callback.best_model_path
    # if best_model_path:
    #     print(f"Loading best model from {best_model_path}")
    # run test set after completing the fit
    trainer.test(model, data,ckpt_path=r"E:\MolSPE\logs\17\last.ckpt")
 # 如果测试未自动调用on_test_epoch_end，手动调用
    model.on_test_epoch_end()

if __name__ == "__main__":
    main()
