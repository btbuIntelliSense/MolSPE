import argparse
import torch
import os
import pytorch_lightning as pl
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from torch_scatter import scatter
from tqdm import tqdm

from Datamodel.MolecularDataModule import MolecularDataModule
from model.module import LNNP


# 根据您的项目结构导入必要的模块
# from main import get_args, MolecularDataModule
# from module import LNNP


def extract_molecular_embeddings(model, batch):
    """根据模型类型提取分子表示"""
    mode = model.hparams.mode

    if mode in ["finetune1", "finetune2", "finetune3"]:
        with torch.no_grad():
            if mode == "finetune1":
                # ET模型表示
                x, _, _, _, batch_idx, _ = model.model.representation_model(
                    batch.z, batch.pos_denoise, batch.n_atom,
                    batch.mask_idx, batch.motif_atoms, batch.batch
                )
                mol_repr = scatter(x, batch.batch, dim=0, reduce='mean')
                return mol_repr.cpu()

            elif mode == "finetune2":
                # Transformer的表示
                _, _, _, _, new_encoder_rep = model.model.mask_model(
                    batch.src_tokens, batch.src_distance, batch.src_coord,
                    batch.src_edge_type, batch.motif_atoms
                )
                return new_encoder_rep.mean(dim=1).cpu()

            elif mode == "finetune3":
                # 分类前的拼接表示
                _, _, _, _, new_encoder_rep = model.model.mask_model(
                    batch.src_tokens, batch.src_distance, batch.src_coord,
                    batch.src_edge_type, batch.motif_atoms
                )
                atom_rep = torch.mean(new_encoder_rep[:, :128, :], dim=1)
                motif_rep = torch.mean(new_encoder_rep[:, 128:, :], dim=1)
                return torch.cat([atom_rep, motif_rep], dim=1).cpu()

    return None


def plot_tsne(embeddings, labels, smiles=None, output_path="tsne_visualization.png"):
    """绘制t-SNE图并保存"""
    plt.figure(figsize=(12, 10))

    # 分类任务使用不同颜色
    if len(np.unique(labels)) < 10:  # 离散标签
        unique_labels = np.unique(labels)
        palette = sns.color_palette("husl", len(unique_labels))

        for i, lbl in enumerate(unique_labels):
            idx = labels == lbl
            plt.scatter(embeddings[idx, 0], embeddings[idx, 1],
                        color=palette[i], label=f'Class {int(lbl)}',
                        alpha=0.7, s=50)
        plt.legend(fontsize=12)
        plt.title('t-SNE Visualization of Molecular Embeddings (Classification)', fontsize=16)

    # 回归任务使用颜色渐变
    else:  # 连续标签
        norm = plt.Normalize(vmin=min(labels), vmax=max(labels))
        scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1],
                              c=labels, cmap='viridis', norm=norm,
                              alpha=0.7, s=50)
        cbar = plt.colorbar(scatter)
        cbar.set_label('Target Value', fontsize=12)
        plt.title('t-SNE Visualization of Molecular Embeddings (Regression)', fontsize=16)

    plt.xlabel('t-SNE Dimension 1', fontsize=14)
    plt.ylabel('t-SNE Dimension 2', fontsize=14)
    plt.grid(alpha=0.2)

    # 保存图片
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved t-SNE visualization to: {output_path}")

    # 保存数据用于进一步分析
    data_path = output_path.replace(".png", ".csv")
    df = pd.DataFrame({
        'Dim1': embeddings[:, 0],
        'Dim2': embeddings[:, 1],
        'Label': labels
    })
    if smiles is not None:
        df['SMILES'] = smiles
    df.to_csv(data_path, index=False)
    print(f"Saved t-SNE data to: {data_path}")


def main():
    parser = argparse.ArgumentParser(description='Run t-SNE visualization using saved checkpoint')
    parser.add_argument('--ckpt_path', type=str, default=r'E:\MolSPE\logs\17\last.ckpt', help='Path to .ckpt file')
    parser.add_argument('--dataset', type=str, default='clintox', help='Dataset name')
    parser.add_argument('--dataset_root', type=str, default='./dataset', help='Dataset root directory')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for inference')
    parser.add_argument('--perplexity', type=int, default=30, help='t-SNE perplexity parameter')
    # parser.add_argument('--max_samples', type=int, default=1000, help='Maximum samples for t-SNE')
    parser.add_argument('--output_dir', type=str, default='./tsne_results', help='Output directory')
    args = parser.parse_args()

    # 创建输出目录
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # 加载训练参数
    ckpt_args = torch.load(args.ckpt_path)['hyper_parameters']

    # 手动设置关键参数（确保与训练时一致）
    # ckpt_args.dataset = args.dataset
    # ckpt_args.dataset_root = args.dataset_root
    # ckpt_args.batch_size = args.batch_size
    # ckpt_args.mode = 'finetune2'  # 假设您想使用finetune2模式
    # ckpt_args.log_dir = args.output_dir  # 重定向日志

    # 初始化数据模块
    vocab = None
    if ckpt_args["dict_path"]:
        from Datamodel.Dictionary import Dictionary
        vocab = Dictionary.load(ckpt_args["dict_path"])

    data = MolecularDataModule(ckpt_args, ckpt_args["mode"], vocab)
    data.prepare_data()
    data.setup("test")

    # 加载模型
    model = LNNP.load_from_checkpoint(
        args.ckpt_path,
        hparams=ckpt_args,
        mode=ckpt_args["mode"],
        vocab=vocab,
        loss_type=ckpt_args["loss_type"],
        mean=data.mean,
        std=data.std
    )
    model.eval().cuda()

    # 收集所有测试数据的分子表示
    all_embeddings = []
    all_labels = []
    all_smiles = []

    for batch in tqdm(data.test_dataloader(), desc="Processing test batches"):
        # 移动数据到GPU
        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # 提取分子表示
        mol_repr = extract_molecular_embeddings(model, batch)
        all_embeddings.append(mol_repr.cpu())

        # 存储标签
        if 'labels' in batch:
            labels = batch['labels'].cpu()
            if len(labels.shape) > 1 and labels.shape[1] > 1:
                labels = labels[:, 0]  # 取第一个任务的标签
            all_labels.append(labels)

        # 如果数据集中包含SMILES，也存储起来
        if 'smiles' in batch:
            all_smiles.extend(batch['smiles'])

    # 连接所有批次的数据
    embeddings = torch.cat(all_embeddings).numpy()
    labels = torch.cat(all_labels).numpy() if all_labels else np.zeros(len(embeddings))

    # 如果数据太多，随机采样
    if len(embeddings) > args.max_samples:
        indices = np.random.choice(len(embeddings), args.max_samples, replace=False)
        embeddings = embeddings[indices]
        labels = labels[indices]
        if all_smiles and len(all_smiles) == len(embeddings):
            all_smiles = [all_smiles[i] for i in indices]

    # 运行t-SNE
    tsne = TSNE(n_components=2, perplexity=args.perplexity,
                random_state=42, n_iter=1000, verbose=1)
    embeddings_2d = tsne.fit_transform(embeddings)

    # 绘制结果
    output_path = os.path.join(args.output_dir, f"tsne_{args.dataset}.png")
    plot_tsne(embeddings_2d, labels, all_smiles, output_path)


if __name__ == '__main__':
    main()