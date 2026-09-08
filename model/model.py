import re
from typing import Optional, List, Tuple
import torch
from torch.autograd import grad
from torch import nn
from torch_scatter import scatter
from pytorch_lightning.utilities import rank_zero_warn
import torch.nn.functional as F

from MaskModel.MultiTaskTransformer3D import MultiTaskTransformer3D
from MaskModel.loss import MolLoss
from model import output_modules
from model.torchmd_et import TorchMD_ET
# from torchmdnet.Molformer.MultiTaskTransformer3D import MultiTaskTransformer3D


# from torchmdnet.models import output_modules
# from torchmdnet.models.wrappers import AtomFilter
# from torchmdnet import priors
import warnings

from moudels.wrappers import AtomFilter


def create_model(args, mean=None, std=None,vocab=None):
    shared_args = dict(
        hidden_channels=args["embedding_dimension"],
        num_layers=args["num_layers"],
        num_rbf=args["num_rbf"],
        rbf_type=args["rbf_type"],
        trainable_rbf=args["trainable_rbf"],
        activation=args["activation"],
        neighbor_embedding=args["neighbor_embedding"],
        cutoff_lower=args["cutoff_lower"],
        cutoff_upper=args["cutoff_upper"],
        max_z=args["max_z"],
        max_num_neighbors=args["max_num_neighbors"],
        mode=args["mode"],
        # num_tasks=args["num_tasks"],
    )

    # representation network
    # if args["model"] == "graph-network":
    #     from torchmdnet.models.torchmd_gn import TorchMD_GN
    #
    #     is_equivariant = False
    #     representation_model = TorchMD_GN(
    #         num_filters=args["embedding_dimension"], aggr=args["aggr"], **shared_args
    #     )
    # elif args["model"] == "transformer":
    #     from torchmdnet.models.torchmd_t import TorchMD_T
    #
    #     is_equivariant = False
    #     representation_model = TorchMD_T(
    #         attn_activation=args["attn_activation"],
    #         num_heads=args["num_heads"],
    #         distance_influence=args["distance_influence"],
    #         **shared_args,
    #     )
    if args["model"] == "equivariant-transformer":
        # from models.torchmd_et import TorchMD_ET

        is_equivariant = True
        representation_model = TorchMD_ET(
            attn_activation=args["attn_activation"],
            num_heads=args["num_heads"],
            distance_influence=args["distance_influence"],
            layernorm_on_vec=args["layernorm_on_vec"],
            use_dataset_md17=args["use_dataset_md17"],
            **shared_args,
        )
    else:
        raise ValueError(f'Unknown architecture: {args["model"]}')


    mask_model = None
    mask_loss = None
    if args["mask_model"] == "Molformer":

        mask_model=MultiTaskTransformer3D(args,vocab)
        mask_loss=MolLoss(args)

    # atom filter
    if not args["derivative"] and args["atom_filter"] > -1:
        representation_model = AtomFilter(representation_model, args["atom_filter"])
    elif args["atom_filter"] > -1:
        raise ValueError("Derivative and atom filter can't be used together")


    # create output network
    output_prefix = "Equivariant" if is_equivariant else ""
    output_model = getattr(output_modules, output_prefix + args["output_model"])(
        args["embedding_dimension"], args["activation"]
    )

    # create the denoising output network
    output_model_noise = None
    if args['output_model_noise'] is not None:
        output_model_noise = getattr(output_modules, output_prefix + args["output_model_noise"])(
            args["embedding_dimension"], args["activation"],
        )

    # combine representation and output network
    model = TorchMD_Net(
        representation_model,
        output_model,
        reduce_op=args["reduce_op"],
        mean=mean,
        std=std,
        derivative=args["derivative"],
        output_model_noise=output_model_noise,
        position_noise_scale=args["position_noise_scale"],
        mask_model=mask_model,
        mask_loss=mask_loss,
        mode=args["mode"],
        num_tasks=args['num_tasks'],
    )
    return model


def load_model(filepath, args=None, device="cpu", mean=None, std=None, **kwargs):
    ckpt = torch.load(filepath, map_location="cpu")
    if args is None:
        args = ckpt["hyper_parameters"]

    for key, value in kwargs.items():
        if not key in args:
            warnings.warn(f'Unknown hyperparameter: {key}={value}')
        args[key] = value

    model = create_model(args)

    state_dict = {re.sub(r"^model\.", "", k): v for k, v in ckpt["state_dict"].items()}

    # NOTE for debug
    new_state_dict = {}
    for k, v in state_dict.items():
        # if 'pos_normalizer' not in k:
        if "output_model_noise.0" in k:
            k = k.replace("output_model_noise.0", "output_model_noise")
        if "head.2" in k:
            continue
        new_state_dict[k] = v

    current_model_dict = model.state_dict()
    # ommit mismatching shape
    new_state_dict2 = {}
    for k in current_model_dict:
        if k in new_state_dict:
            # print(k, current_model_dict[k].size(), new_state_dict[k].size())
            if current_model_dict[k].size() == new_state_dict[k].size():
                new_state_dict2[k] = new_state_dict[k]
            else:
                print(f"warning {k} shape mismatching, not loaded")
                new_state_dict2[k] = current_model_dict[k]

    # loading_return = model.load_state_dict(state_dict, strict=False)
    loading_return = model.load_state_dict(new_state_dict2, strict=False)

    if len(loading_return.unexpected_keys) > 0:
        # Should only happen if not applying denoising during fine-tuning.
        assert all(
            (
                "output_model_noise" in k
                or "pos_normalizer" in k
                # or "representation_spec_model" in k
                # or "output_model_spec" in k
                # or "output_model_mol" in k
            )
            for k in loading_return.unexpected_keys
        )
    # assert len(loading_return.missing_keys) == 0, f"Missing keys: {loading_return.missing_keys}"

    if mean:
        model.mean = mean
    if std:
        model.std = std

    return model.to(device)


@torch.jit.script
def gaussian(x, mean, std):
    pi = 3.14159
    a = (2*pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)

class GaussianLayer(nn.Module):
    def __init__(self, K=128, edge_types=11):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        nn.init.uniform_(self.means.weight, 2, 3)
        nn.init.uniform_(self.stds.weight, 1, 1.5)


    def forward(self, x, edge_types):
        # mul = self.mul(edge_types)
        # bias = self.bias(edge_types)
        # x = mul * x.unsqueeze(-1) + bias
        x = x.unsqueeze(-1)
        mean = self.means.weight.float().view(-1).unsqueeze(0)
        std = (self.stds.weight.float().view(-1).abs() + 1e-2).unsqueeze(0)
        x = gaussian(x.float(), mean, std).type_as(self.means.weight)
        return x


class TorchMD_Net(nn.Module):

    def __init__(
        self,
        representation_model,
        output_model,
        reduce_op="add",
        mean=None,
        std=None,
        derivative=False,
        output_model_noise=None,
        position_noise_scale=0.0,
        mask_model=None,
        mask_loss=None,
        mode=None,
        num_tasks=None,
    ):
        super(TorchMD_Net, self).__init__()
        self.representation_model = representation_model
        self.output_model = output_model
        self.mask_model=mask_model
        self.maskloss_fn=mask_loss
        self.num_tasks=num_tasks

        self.reduce_op = reduce_op
        self.derivative = derivative
        self.output_model_noise = output_model_noise
        self.position_noise_scale = position_noise_scale
        self.mode =mode

        mean = torch.scalar_tensor(0) if mean is None else mean
        self.register_buffer("mean", mean)
        std = torch.scalar_tensor(1) if std is None else std
        self.register_buffer("std", std)
        if self.num_tasks is not None:
            self.classifier = nn.Sequential(
                nn.Linear(2 * 256, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Dropout(0.3),
                nn.Linear(256, num_tasks),
                # nn.Sigmoid()
            )
            self.down_stream_out_fn = nn.Linear(256, self.num_tasks)

        if self.position_noise_scale > 0:
            self.pos_normalizer = AccumulatedNormalization(accumulator_shape=(3,))
            # 分类器
        # self.classifier = nn.Sequential(
        #         nn.Linear(2 * 256, 256),
        #         nn.ReLU(),
        #         nn.BatchNorm1d(256),
        #         nn.Dropout(0.3),
        #         nn.Linear(256, num_tasks),
        #         # nn.Sigmoid()
        #     )
        self.reset_parameters()

    def reset_parameters(self):
        self.representation_model.reset_parameters()
        self.output_model.reset_parameters()
        if self.output_model_noise is not None:
            self.output_model_noise.reset_parameters()
        # if self.mask_model is not None:
        #      self.mask_model.reset_parameters()


    def forward(self,  batch: Optional[torch.Tensor] = None):
        if batch.batch is None:
            batch_size = batch.batch_size
            num_atoms_per_sample = batch.z.size(0) // batch_size
            new_batch = torch.repeat_interleave(torch.arange(batch_size, device=batch.z.device), num_atoms_per_sample)
            # 如果有剩余的原子，将它们分配到最后一个 batch
            if batch.z.size(0) % batch_size != 0:
                new_batch = torch.cat(
                    [new_batch, torch.full((batch.z.size(0) % batch_size,), batch_size - 1, device=batch.z.device)],
                    dim=0)
            batch.batch = new_batch
        if self.mode=="pretrain1" or "finetune1" in self.mode:
            # assert batch.z.dim() == 1 and batch.z.dtype == torch.long
            # batch.batch = torch.zeros_like(batch.z) if batch.batch is None else batch.batch
            x, v, z, pos, batch1,empp_loss= self.representation_model(batch.z, batch.pos_denoise, batch.n_atom,batch.mask_idx,batch.motif_atoms,batch.batch)
            # pos_noise=batch.pos_nosie
            mask_loss=None
            noise_pred = None
            if self.output_model_noise is not None:
                noise_pred = self.output_model_noise.pre_reduce(x, v, z, pos, batch1)
            # apply the output network
            out = scatter(x, batch1, dim=0, reduce=self.reduce_op)
            # x1 = pool.global_add_pool(x, batch1)
            if "finetune1" in self.mode:
               out = self.down_stream_out_fn(out)


        if self.mode=="pretrain2" or "finetune2" in self.mode:
            # batch.batch = None
            x, v, z, pos, batch1,empp_loss = self.representation_model(batch.z, batch.pos_denoise,batch.n_atom,batch.mask_idx,batch.motif_atoms,batch.batch)
            #x(2381,256)
            sample = {
                    "net_input": {
                        "src_tokens": batch.mask_z,
                        "src_coord": batch.mask_coord,
                        "src_distance": batch.mask_dist,
                        # "src_edge_type": batch.src_edge_type,
                        "motif_atoms":batch.motif_atoms,
                        'src_edge_type':batch.bond_adj
                    },
                    "target": {
                        "tokens_target": batch.targets,
                        "distance_target": batch.dist,
                        "coord_target": batch.coord,

                    }
                }
            mask_loss,new_encoder_rep= self.maskloss_fn(self.mask_model, sample)
            noise_pred = None
            if self.output_model_noise is not None:
                noise_pred = self.output_model_noise.pre_reduce(x, v, z, pos, batch1)
            out = None
            if "finetune2" in self.mode:
            # trans_motif_embed = self.motif_trans(batch.motif_vocab) #卷积合
            # # apply the output network
            #    x = self.output_model.pre_reduce(x, v, z, pos, batch1)  ##EquivariantVectorOutput.pre_reduce  GNN卷积
               x = scatter(x, batch1, dim=0, reduce='mean')
            # out = torch.cat((out, trans_motif_embed), dim=1)
            #
            # out = self.down_stream_out_fn(out)

               atom_rep = torch.mean(new_encoder_rep[:, :128, :],dim=1)  # (batch_size32, embed_dim256)
               motif_rep = torch.mean(new_encoder_rep[:, 128:, :],dim=1)  # (batch_size32, embed_dim256)
               # atom_rep=atom_rep+x
               # 拼接两种表示
               mol_rep = torch.cat([atom_rep, motif_rep], dim=1)  # (batch_size32, 2*embed_dim512)

               out = self.classifier(mol_rep)

        if self.mode == "pretrain3" or "finetune3" in self.mode:
            sample = {
                "net_input": {
                    "src_tokens": batch.mask_z,
                    "src_coord": batch.mask_coord,
                    "src_distance": batch.mask_dist,
                    # "src_edge_type": batch.src_edge_type,
                    "motif_atoms": batch.motif_atoms,
                    'src_edge_type': batch.bond_adj
                },
                "target": {
                    "tokens_target": batch.targets,
                    "distance_target": batch.dist,
                    "coord_target": batch.coord,

                }
            }
            mask_loss, new_encoder_rep = self.maskloss_fn(self.mask_model, sample)
            noise_pred = None
            empp_loss = None

            out = None
            if "finetune3" in self.mode:
                atom_rep = torch.mean(new_encoder_rep[:, :128, :], dim=1)  # (batch_size, embed_dim)
                motif_rep = torch.mean(new_encoder_rep[:, 128:, :], dim=1)  # (batch_size, embed_dim)

                # 拼接两种表示
                mol_rep = torch.cat([atom_rep, motif_rep], dim=1)  # (batch_size, 2*embed_dim)

                out = self.classifier(mol_rep)
        return out, noise_pred, mask_loss,empp_loss


class AccumulatedNormalization(nn.Module):
    """Running normalization of a tensor."""
    def __init__(self, accumulator_shape: Tuple[int, ...], epsilon: float = 1e-8):
        super(AccumulatedNormalization, self).__init__()

        self._epsilon = epsilon
        self.register_buffer("acc_sum", torch.zeros(accumulator_shape))
        self.register_buffer("acc_squared_sum", torch.zeros(accumulator_shape))
        self.register_buffer("acc_count", torch.zeros((1,)))
        self.register_buffer("num_accumulations", torch.zeros((1,)))

    def update_statistics(self, batch: torch.Tensor):
        batch_size = batch.shape[0]
        self.acc_sum += batch.sum(dim=0)
        self.acc_squared_sum += batch.pow(2).sum(dim=0)
        self.acc_count += batch_size
        self.num_accumulations += 1

    @property
    def acc_count_safe(self):
        return self.acc_count.clamp(min=1)

    @property
    def mean(self):
        return self.acc_sum / self.acc_count_safe

    @property
    def std(self):
        return torch.sqrt(
            (self.acc_squared_sum / self.acc_count_safe) - self.mean.pow(2)
        ).clamp(min=self._epsilon)

    def forward(self, batch: torch.Tensor):
        if self.training:
            self.update_statistics(batch)
        return ((batch - self.mean) / self.std)
