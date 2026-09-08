from pathlib import Path

import torch
import torchmetrics
from sklearn.decomposition import PCA
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.nn.functional import mse_loss, l1_loss, smooth_l1_loss
from torch import nn
from pytorch_lightning import LightningModule
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
# from torchmdnet.models.model import create_model, load_model
from sklearn.multioutput import MultiOutputClassifier
from torch_scatter import scatter
from math import inf
from model.model import load_model, create_model
import numpy as np
import torch.nn.functional as F

class PlateauScheduler(ReduceLROnPlateau):
    def __init__(self, factor, patience):

        self.factor = factor
        self.patience = patience
        self.threshold = 1e-4
        self.mode = "min"
        self.threshold_mode = "rel"
        self.best = inf
        self.num_bad_epochs = None
        self.eps = 1e-8
        self.last_epoch = 0

    def step(self, metrics, epoch=None):
        current = float(metrics)
        self.last_epoch += 1

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs > self.patience:
            self.num_bad_epochs = 0
            return self.factor

        return 1.0


class LNNP(LightningModule):
    def __init__(self, hparams, mode, vocab,loss_type,mean=None, std=None):
        super(LNNP, self).__init__()
        self.save_hyperparameters(hparams, ignore=['vocab'])
        self.mode=mode
        self.mean=mean
        self.vocab=vocab
        self.std=std
        self.loss_type=loss_type
        self.test_embeddings = []  # 存储分子表示
        self.test_labels = []  # 存储标签
        self.test_smiles = []  # 可选：存储SMILES用于标记]
        self.mask_weight_param = nn.Parameter(torch.tensor(float(self.hparams["mask_weight"])))
        self.classweight_param = nn.Parameter(torch.tensor(float(self.hparams["classweight"])))
        self.denoising_weight_param = nn.Parameter(torch.tensor(float(self.hparams["denoising_weight"])))

        # 损失EMA跟踪 (指数移动平均)
        self.loss_ema = {"mask": None, "y": None, "empp": None}
        self.ema_alpha = 0.9  # EMA平滑系数
        # 自适应调整参数
        # self.task_weights = nn.Parameter(torch.zeros(3))  # [mask, class, empp]
        if 'classification' in self.mode:
            if self.loss_type == 'bce':
                self.down_stream_criterion = nn.BCEWithLogitsLoss()
            else:
                self.down_stream_criterion = nn.CrossEntropyLoss()
            if self.hparams["num_tasks"]==1:
                self.aucroc = torchmetrics.AUROC(task='binary')
            else:
                self.aucroc = torchmetrics.AUROC(task='multilabel',num_labels=self.hparams["num_tasks"])

        elif 'regression' in self.mode:
            self.down_stream_criterion = mse_loss()

        if self.hparams.load_model:
            self.model = load_model(self.hparams.load_model, args=self.hparams)
        elif self.hparams.pretrained_model:
            self.model = load_model(self.hparams.pretrained_model, args=self.hparams, mean=mean, std=std)
        else:
            if self.mode=="pretrain1" or "finetune1" in self.mode:
              self.model = create_model(self.hparams, mean, std)
            elif self.mode=="pretrain2" or "finetune2" in self.mode:
              self.model = create_model(self.hparams, mean, std, self.vocab)

        # initialize exponential smoothing
        self.ema = None
        self._reset_ema_dict()

        # initialize loss collection
        self.losses = None
        self._reset_losses_dict()

        self.last_epoch = 0
        self.lr_gen_scheduler = PlateauScheduler(
            factor=self.hparams.lr_factor,
            patience=self.hparams.lr_patience,
        )
        self.val_loss = None


    def configure_optimizers(self):
        # optimizer = AdamW(
        #     self.model.parameters(),
        #     lr=self.hparams.lr,
        #     weight_decay=self.hparams.weight_decay,
        # )
        # 确保权重参数也被优化 (包含所有可学习参数)
        optimizer = AdamW(
            list(self.model.parameters()) +
            [self.mask_weight_param, self.classweight_param, self.denoising_weight_param],
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        # 训练时添加梯度裁剪
        # self.trainer = self.trainer(gradient_clip_val=1.0, gradient_clip_algorithm="norm")
        if self.hparams.lr_schedule == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, self.hparams.lr_cosine_length, eta_min=1e-7)
            lr_scheduler = {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        elif self.hparams.lr_schedule == 'cosine_warmup':
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=self.hparams.lr_cosine_length, T_mult=2, eta_min=1e-7)
            lr_scheduler = {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        elif self.hparams.lr_schedule == 'reduce_on_plateau':
            scheduler = ReduceLROnPlateau(
                optimizer,
                "min",
                factor=self.hparams.lr_factor,
                patience=self.hparams.lr_patience,
                min_lr=self.hparams.lr_min,
            )
            lr_scheduler = {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            }
        else:
            raise ValueError(f"Unknown lr_schedule: {self.hparams.lr_schedule}")
        return [optimizer], [lr_scheduler]

    # def forward(self, z, pos, spec,batch=None):
    #     return self.model(z, pos, spec,batch=batch)
    def forward(self, batch):
        return self.model(batch)

    # def training_step(self, batch, batch_idx):
    #     return self.step(batch, mse_loss, "train")

    def validation_step(self, batch, batch_idx, *args):
        if len(args) == 0 or (len(args) > 0 and args[0] == 0):
            return self.step(batch, mse_loss, "val")
        # test step
        return self.step(batch, l1_loss, "test")
    #
    def test_step(self, batch, batch_idx):
        return self.step(batch, l1_loss, "test")

    # def test_step(self, batch, batch_idx):
    def training_step(self, batch, batch_idx):
    #     # 首先调用原始的 step 方法计算损失
    #     loss = self.step(batch, l1_loss, "test")
        loss = self.step(batch, mse_loss, "train")
        # 添加调试信息
        print(f"Processing batch {batch_idx}")
        # 添加代码：提取分子表示并存储
        if "finetune1" in self.mode or "finetune2" in self.mode or "finetune3" in self.mode:
            # 提取分子表示
            mol_repr = self.extract_molecular_embeddings(batch)

            # 存储分子表示和标签
            self.test_embeddings.append(mol_repr.detach().cpu())

            # 如果测试集有标签（不是预测阶段），存储标签
            if hasattr(batch, 'labels'):
                labels = batch.labels
                # 如果标签是多任务分类（维度大于1），取第一个任务的标签
                if labels.dim() > 1 and labels.size(1) > 1:
                    labels = labels[:, 0]  # 取第一个任务的标签用于可视化
                self.test_labels.append(labels.detach().cpu())
            # 如果数据集中包含 SMILES，也存储起来
            if hasattr(batch, 'smiles'):
                self.test_smiles.extend(batch.smiles)
        return loss

    def step(self, batch, loss_fn, stage):
        with torch.set_grad_enabled(stage == "train" or self.hparams.derivative):
            # if self.mode == "pretrain1" or self.mode == "pretrain2":
            pred, noise_pred, loss_reconstruct,empp_loss = self(batch)
            # if self.mode == "finetune1" or self.mode == "finetune2":
            #    pred, noise_pred, loss_reconstruct = self(batch)


        # 是否启用去噪任务
        denoising_is_on = ("pos_denoise" in batch) and (self.hparams.denoising_weight > 0) and (noise_pred is not None)
        reconstruct_is_on = (self.hparams.denoising_weight > 0) and (loss_reconstruct is not None)
        if reconstruct_is_on: #掩码重构
            self.losses[stage + "_reconstruct"].append(loss_reconstruct.detach())
        else:
            loss_reconstruct = 0

        loss_reconstruct = torch.nan_to_num(loss_reconstruct, nan=0.0, posinf=1e6, neginf=-1e6)
        empp_loss = torch.nan_to_num(empp_loss, nan=0.0, posinf=1e6, neginf=-1e6)
        loss_y=0
        if denoising_is_on: #计算去噪任务的损失
            # normalized_pos_target = self.model.pos_normalizer(batch.pos_noise)
            # loss_pos = loss_fn(noise_pred, normalized_pos_target)
            # self.losses[stage + "_pos"].append(loss_pos.detach())
            self.losses[stage + "_empp"].append(empp_loss.detach())
        else:
            empp_loss=0
        if "labels" in batch:
            cur_pre = torch.sigmoid(pred)
            # labels=batch.labels.float()
            mask=torch.where(batch.labels !=-1,True,False)
            # pred=pred.view(-1)
            loss_y = self.down_stream_criterion(cur_pre[mask].view(-1), batch.labels[mask].view(-1).float())  # 属性预测损失
            loss_y = torch.nan_to_num(loss_y, nan=0.0, posinf=1e6, neginf=-1e6)
            # loss_y1 = F.binary_cross_entropy(cur_pre[mask].view(-1), batch.labels[mask].view(-1).float())
            self.losses[stage + "_y"].append(loss_y.detach())
            # labels = batch.labels[mask]
            # # labels=labels.long
            # self.aucroc.update(cur_pre[mask],labels)
            # aucroc_val = self.aucroc.compute()
            # 多标签
            row_mask = (batch.labels != -1).any(dim=1)  # 形状[8]
            # 获取所有有效的行（样本）
            valid_cur_pre = cur_pre[row_mask]  # 形状[k,2]，k<=8
            valid_labels = batch.labels[row_mask]
            valid_labels = valid_labels.long()  # 形状[k,2]
            # labels = batch.labels[mask]
            # labels=labels.long
            self.aucroc.update(valid_cur_pre, valid_labels)
            aucroc_val = self.aucroc.compute()
            self.log(f'{stage}_aucroc', aucroc_val, sync_dist=True)
            print('AUCROC:', aucroc_val)


        # 1. 先获取当前损失值（不中断梯度）
        scalar_loss_reconstruct = loss_reconstruct.item() if torch.is_tensor(loss_reconstruct) else loss_reconstruct
        scalar_loss_y = loss_y.item() if torch.is_tensor(loss_y) else loss_y
        scalar_empp_loss = empp_loss.item() if torch.is_tensor(empp_loss) else empp_loss
        # 1. 更新损失EMA（指数移动平均）
        with torch.no_grad():
            # 更新EMA
            for name, value in zip(["mask", "y", "empp"],
                                   [scalar_loss_reconstruct, scalar_loss_y, scalar_empp_loss]):
                if self.loss_ema[name] is None:
                    self.loss_ema[name] = value
                else:
                    self.loss_ema[name] = self.ema_alpha * self.loss_ema[name] + (1 - self.ema_alpha) * value

            # 计算自适应权重
            adaptive_weights = {
                "mask": 1 / (self.loss_ema["mask"] + 1e-8),
                "y": 1 / (self.loss_ema["y"] + 1e-8),
                "empp": 1 / (self.loss_ema["empp"] + 1e-8)
            }
            total_inv = sum(adaptive_weights.values())
            adaptive_weights = {k: v / total_inv for k, v in adaptive_weights.items()}

            # 5. 可选：基于AUC的动态调整
            if stage == "train" and "y" in self.loss_ema and self.loss_ema["y"] > 0:
                current_auc = self.aucroc.compute()
                # 当AUC低于0.6时增强分类任务权重
                if current_auc < 0.6:
                    with torch.no_grad():
                        # # 增加分类任务权重10%，上限为5.0
                        # new_weight = self.classweight_param + 0.1
                        # self.classweight_param.copy_(torch.clamp(new_weight, 0, 10.0))
                        adjustment_factor = max(0.05, (0.6 - current_auc) / 0.6 * 0.2)
                        new_weight = self.classweight_param + adjustment_factor

                        # 对数调整确保大权重时调整更平稳
                        if new_weight > 1.0:
                            new_weight = new_weight * (1.0 + adjustment_factor / 10)

                        self.classweight_param.copy_(torch.clamp(new_weight, 0, 10.0))
                # 当AUC高于0.8时恢复权重
                # elif current_auc > 0.8:
                #     with torch.no_grad():
                #         # 恢复初始设置的分类权重
                #         self.classweight_param.copy_(torch.tensor(float(self.hparams["classweight"])))

        # =============================================================
        # total loss
        # 使用sigmoid确保权重为正数但不过大
        mask_coeff = self.mask_weight_param.sigmoid() * adaptive_weights["mask"]
        y_coeff = self.classweight_param.sigmoid() * adaptive_weights["y"]
        empp_coeff = self.denoising_weight_param.sigmoid() * adaptive_weights["empp"]

        # 4. 计算总损失
        loss = (
                mask_coeff * loss_reconstruct +
                y_coeff * loss_y +
                empp_coeff * empp_loss
        )
        # loss = (
        #          # loss_pos * self.hparams.denoising_weight \
        #         loss_reconstruct * self.hparams.mask_weight\
        #         +loss_y*self.hparams.classweight\
        #         +empp_loss* self.hparams.denoising_weight
        # )
        # 损失值平滑（自适应加权）
        # weights = F.softmax(torch.tensor([
        #     self.hparams.mask_weight,
        #     self.hparams.classweight,
        #     self.hparams.denoising_weight
        # ], device=self.device), dim=0)
        #
        # loss = (
        #         weights[0] * loss_reconstruct +
        #         weights[1] * loss_y +
        #         weights[2] * empp_loss
        # )

        self.losses[stage].append(loss.detach())
        print('total loss:', loss)
        print('mask_loss:', mask_coeff * loss_reconstruct)
        print('loss_y:', y_coeff * loss_y)
        print("empp_loss:", empp_coeff * empp_loss)
        # print('mask_loss:', weights[0] * loss_reconstruct)
        # print('loss_y:',  weights[1] * loss_y )
        #
        # # print('auc:', auc)
        # print("empp_loss:", weights[2] * empp_loss)
        # print('mask_loss:', loss_reconstruct * self.hparams.mask_weight)
        # print('loss_y:', loss_y*self.hparams.classweight)
        #
        # # print('auc:', auc)
        # print("empp_loss:", empp_loss* self.hparams.denoising_weight)
        # Frequent per-batch logging for training 记录每个批次（batch）的训练指标
        if stage == 'train':
            # train_metrics = {k + "_per_step": v[-1] for k, v in self.losses.items() if (k.startswith("train") and len(v) > 0)}
            # train_metrics['lr_per_step'] = self.trainer.optimizers[0].param_groups[0]["lr"]
            # train_metrics['step'] = self.trainer.global_step
            # train_metrics['batch_pos_mean'] = batch.pos_denoise.mean().item()
            train_metrics = {
                "mask_coeff": mask_coeff.item(),
                "y_coeff": y_coeff.item(),
                "empp_coeff": empp_coeff.item(),
                "lr_per_step": self.trainer.optimizers[0].param_groups[0]["lr"],
                "step": self.trainer.global_step,
                "batch_pos_mean": batch.pos_denoise.mean().item(),
            }
            self.log_dict(train_metrics, sync_dist=True)

        return loss


    def optimizer_step(self, *args, **kwargs):
        epoch = kwargs["epoch"] if "epoch" in kwargs else args[0]
        batch_idx = kwargs["batch_idx"] if "batch_idx" in kwargs else args[1]
        optimizer = kwargs["optimizer"] if "optimizer" in kwargs else args[2]
        if self.trainer.global_step < self.hparams.lr_warmup_steps:
            lr_scale = min(
                1.0,
                float(self.trainer.global_step + 1)
                / float(self.hparams.lr_warmup_steps),
            )

            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams.lr
            # ================= 核心修改：梯度归一化处理 =================
            # 在调用super()之前处理梯度
        # for name, param in self.named_parameters():
        #   print(name)
        #     if param.grad is not None:
        #             # 根据不同任务头部应用不同的梯度尺度
        #          if "mask_head" in name and param.grad is not None:
        #                 # 减少mask分支梯度影响
        #              param.grad *= 0.1
        #          elif "pred_head" in name and param.grad is not None:
        #             # 增强分类分支梯度
        #              param.grad *= 2.0
            # =========================================================

        # elif self.hparams.reduce_lr_when_bad:
        #     if self.val_loss is not None and optimizer.param_groups[0]["lr"] > self.hparams.lr * 0.1:
        #         lr_scale = self.lr_gen_scheduler.step(self.val_loss.item())
        #         self.val_loss = None
        #     else:
        #         lr_scale = 1.0
        #
        #     for pg in optimizer.param_groups:
        #         pg["lr"] *= lr_scale

        super().optimizer_step(*args, **kwargs)
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

    def on_training_epoch_end(self): #def training_epoch_end(self, training_step_outputs):
        dm = self.trainer.datamodule
        if hasattr(dm, "test_dataset") and len(dm.test_dataset) > 0:
            should_reset = (
                self.current_epoch % self.hparams.test_interval == 0
                or (self.current_epoch - 1) % self.hparams.test_interval == 0
            )
            if should_reset:
                self.trainer.reset_val_dataloader(self)


    def on_validation_epoch_end(self):  #def validation_epoch_end(self, validation_step_outputs):
        if not self.trainer.sanity_checking:
        # if not self.trainer.running_sanity_check:
            result_dict = {
                "epoch": self.current_epoch,
                "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
                "train_loss": torch.stack(self.losses["train"]).mean(),
                "val_loss": torch.stack(self.losses["val"]).mean(),
            }
            self.val_loss = result_dict["val_loss"]

            # add test loss if available
            if len(self.losses["test"]) > 0:
                result_dict["test_loss"] = torch.stack(self.losses["test"]).mean()

            # if prediction and derivative are present, also log them separately
            if len(self.losses["train_y"]) > 0 and len(self.losses["train_dy"]) > 0:
                result_dict["train_loss_y"] = torch.stack(self.losses["train_y"]).mean()
                result_dict["train_loss_dy"] = torch.stack(
                    self.losses["train_dy"]
                ).mean()
                result_dict["val_loss_y"] = torch.stack(self.losses["val_y"]).mean()
                result_dict["val_loss_dy"] = torch.stack(self.losses["val_dy"]).mean()

                if len(self.losses["test"]) > 0:
                    result_dict["test_loss_y"] = torch.stack(
                        self.losses["test_y"]
                    ).mean()
                    result_dict["test_loss_dy"] = torch.stack(
                        self.losses["test_dy"]
                    ).mean()

            if len(self.losses["train_y"]) > 0:
                result_dict["train_loss_y"] = torch.stack(self.losses["train_y"]).mean()
            if len(self.losses['val_y']) > 0:
                result_dict["val_loss_y"] = torch.stack(self.losses["val_y"]).mean()
            if len(self.losses["test_y"]) > 0:
                result_dict["test_loss_y"] = torch.stack(
                    self.losses["test_y"]
                ).mean()

            # if denoising is present, also log it
            # if len(self.losses["train_pos"]) > 0:
            #     result_dict["train_loss_pos"] = torch.stack(
            #         self.losses["train_pos"]
            #     ).mean()
            #
            # if len(self.losses["val_pos"]) > 0:
            #     result_dict["val_loss_pos"] = torch.stack(
            #         self.losses["val_pos"]
            #     ).mean()
            #
            # if len(self.losses["test_pos"]) > 0:
            #     result_dict["test_loss_pos"] = torch.stack(
            #         self.losses["test_pos"]
            #     ).mean()
            if len(self.losses["train_empp"]) > 0:
               result_dict["train_loss_empp"] = torch.stack(
                self.losses["train_empp"]).mean()

            if len(self.losses["val_empp"]) > 0:
                result_dict["val_loss_empp"] = torch.stack(
                self.losses["val_empp"]).mean()

            if len(self.losses["test_empp"]) > 0:
                result_dict["test_loss_empp"] = torch.stack(
                self.losses["test_empp"]).mean()

            # if contrast is present, also log it
            if len(self.losses["train_contrast"]) > 0:
                result_dict["train_loss_contrast"] = torch.stack(
                    self.losses["train_contrast"]
                ).mean()

            if len(self.losses["val_contrast"]) > 0:
                result_dict["val_loss_contrast"] = torch.stack(
                    self.losses["val_contrast"]
                ).mean()

            if len(self.losses["test_contrast"]) > 0:
                result_dict["test_loss_contrast"] = torch.stack(
                    self.losses["test_contrast"]
                ).mean()

            # if reconstruct is present, also log it
            if len(self.losses["train_reconstruct"]) > 0:
                result_dict["train_loss_reconstruct"] = torch.stack(
                    self.losses["train_reconstruct"]
                ).mean()

            if len(self.losses["val_reconstruct"]) > 0:
                result_dict["val_loss_reconstruct"] = torch.stack(
                    self.losses["val_reconstruct"]
                ).mean()

            if len(self.losses["test_reconstruct"]) > 0:
                result_dict["test_loss_reconstruct"] = torch.stack(
                    self.losses["test_reconstruct"]
                ).mean()

            self.log_dict(result_dict, sync_dist=True)
        self._reset_losses_dict()

    def on_test_epoch_end(self):
        """测试结束后执行t-SNE可视化"""
        if not self.test_embeddings:
            return

        embeddings = torch.cat(self.test_embeddings).numpy()
        labels = torch.cat(self.test_labels).numpy()
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        scaler = MinMaxScaler()
        embeddings = scaler.fit_transform(embeddings) #归一化

        # pca = PCA(n_components=50, random_state=42)
        # embeddings = pca.fit_transform(embeddings) #PCA降维到50
        # 执行t-SNE降维
        from sklearn.manifold import TSNE
        # tsne = TSNE(n_components=2, perplexity=40, learning_rate="auto",max_iter=1200,n_iter_without_progress=300,random_state=42)
        # embeddings_2d = tsne.fit_transform(embeddings)
        from umap import UMAP
        # reducer = UMAP(n_neighbors=50,min_dist=0.5, n_components=2,n_epochs=500, random_state=123, metric="euclidean",densmap=True, # 启用密度模式
        #                dens_lambda=2.0,         # 密度敏感度
        #                output_dens=True)         # 输出密度值
        reducer = UMAP(n_neighbors=15, min_dist=0.01, n_components=2,spread=0.8, n_epochs=800, repulsion_strength=1.5,random_state=42, metric="euclidean")
        embeddings_2d = reducer.fit_transform(embeddings)
        # 绘制结果
        # self.plot_tsne(embeddings_2d, labels)
        # HDBSCAN聚类
        from hdbscan import HDBSCAN
        clusterer = HDBSCAN(
            min_cluster_size=50,
            metric="euclidean",
            cluster_selection_method="eom",
            gen_min_span_tree=True
        )
        cluster_labels = clusterer.fit_predict(embeddings_2d)

        # 计算ARI
        from sklearn.metrics import adjusted_rand_score
        ari_score = adjusted_rand_score(labels, cluster_labels)

        # 创建可视化
        import matplotlib.pyplot as plt
        import seaborn as sns
        import matplotlib.patches as mpatches

        plt.figure(figsize=(10, 8))
        # 创建颜色映射
        unique_labels = np.unique(labels)
        palette = sns.color_palette("husl", len(unique_labels))
        color_map = {label: palette[i] for i, label in enumerate(unique_labels)}

        # 绘制每个数据点
        for label in unique_labels:
            idx = labels == label
            plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1],
                        color=color_map[label], alpha=0.7, s=15, label=f'Label {int(label)}')

        # 创建图例
        legend_handles = [mpatches.Patch(color=color_map[label], label=f'Label {label}')
                          for label in unique_labels]
        plt.legend(handles=legend_handles, title="True Labels", loc='best')

        # 添加标题和ARI值
        # plt.title(f'Molecular Representation Clustering\nAdjusted Rand Index = {ari_score:.4f}',
        #           fontsize=14, pad=20)
        # plt.xlabel('UMAP Dimension 1')
        # plt.ylabel('UMAP Dimension 2')
        plt.grid(alpha=0.2)

        # 保存图片
        umap_path = Path(self.hparams.log_dir) / 'molecular_clustering_result.png'
        plt.savefig(umap_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved molecular clustering visualization to: {umap_path}")

    def plot_tsne(self, embeddings, labels):
        """绘制t-SNE图并保存"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.figure(figsize=(8, 6))

        # 分类任务使用不同颜色
        if self.loss_type == 'bce':
            unique_labels = np.unique(labels)
            palette = sns.color_palette("husl", len(unique_labels))
            for i,lbl in enumerate(unique_labels):
                idx = (labels == lbl).flatten()
                embedding=embeddings[idx]

                plt.scatter(embedding[:, 0], embedding[:, 1],
                color=palette[i], label=f'Class {int(lbl)}',
                alpha=0.7, s=10)
        # 回归任务使用颜色渐变
        else:
            scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1],
                                  c=labels, cmap='viridis', alpha=0.6)
            plt.colorbar(scatter, label='Target Value')
        # # 设置坐标轴范围[0,1]
        # plt.xlim(0, 1)
        # plt.ylim(0, 1)
        plt.title('t-SNE Visualization of Molecular Embeddings')
        # plt.xlabel('t-SNE Dimension 1')
        # plt.ylabel('t-SNE Dimension 2')

        # 保存图片
        tsne_path = Path(self.hparams.log_dir) / 'tsne_visualization.png'
        plt.savefig(tsne_path, dpi=300, bbox_inches='tight')
        # plt.close()
        print(f"Saved t-SNE visualization to: {tsne_path}")

    def _reset_losses_dict(self):
        self.losses = {
            "train": [],
            "val": [],
            "test": [],
            "train_y": [],
            "val_y": [],
            "test_y": [],
            "train_dy": [],
            "val_dy": [],
            "test_dy": [],
            # "train_pos": [],
            # "val_pos": [],
            # "test_pos": [],
            "train_empp": [],
            "val_empp": [],
            "test_empp": [],
            "train_contrast": [],
            "val_contrast": [],
            "test_contrast": [],
            "train_reconstruct": [],
            "val_reconstruct": [],
            "test_reconstruct": [],
        }

    def _reset_ema_dict(self):
        self.ema = {"train_y": None, "val_y": None, "train_dy": None, "val_dy": None}

    def ctr_loss_fn(self, molecule_feature, sp_feature, temperature=0.07):
        from torch.nn import functional as F

        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(molecule_feature[:, None, :], sp_feature[None, :, :], dim=-1)

        postive_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        # InfoNCE loss
        cos_sim = cos_sim / temperature
        nll = -cos_sim[postive_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()

        return nll

    def extract_molecular_embeddings(self, batch):
        """根据模型类型提取分子表示"""
        if "finetune1" in self.mode :
            # finetune1 模式：直接从 ET 模型获取分子级表示
            x, _, z, pos, batch_idx, _ = self.model.representation_model(
                batch.z, batch.pos_denoise, batch.n_atom, batch.mask_idx,
                batch.motif_atoms, batch.batch
            )
            mol_repr = scatter(x, batch.batch, dim=0, reduce='mean')
            return mol_repr

        elif "finetune2" in self.mode or "finetune3" in self.mode:
            # finetune2/3 模式：获取 Transformer 的分子表示
            # 直接调用模型的前向传播，但只获取表示部分
            _, _, _, _, _,encoder_rep,new_encoder_rep = self.model.mask_model(
                batch.mask_z, batch.mask_dist, batch.mask_coord,
                batch.bond_adj, batch.motif_atoms
            )

            if "finetune2" in self.mode:
                # finetune2 模式：使用整个序列的平均
                # mol_repr = new_encoder_rep.mean(dim=1)
                # mol_repr = new_encoder_rep
                # atom_rep = torch.mean(new_encoder_rep[:, :128, :], dim=1)
                # motif_rep = torch.mean(new_encoder_rep[:, 128:, :], dim=1)
                # mol_repr = torch.cat([atom_rep, motif_rep], dim=1)
                mol_repr = encoder_rep.mean(dim=1)

            else:
                # finetune3 模式：拼接原子和基序表示
                atom_rep = torch.mean(new_encoder_rep[:, :128, :], dim=1)
                motif_rep = torch.mean(new_encoder_rep[:, 128:, :], dim=1)
                mol_repr = torch.cat([atom_rep, motif_rep], dim=1)

            return mol_repr
        else:
            # 其他模式（如预训练）不返回表示
            return None

