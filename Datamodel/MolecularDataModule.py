import os
from functools import partial
from os.path import join
from tqdm import tqdm
import torch
from torch.utils.data import Subset
from torch_geometric.data import DataLoader, Batch, Data,InMemoryDataset
from torch_geometric.data import Dataset
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities import rank_zero_warn

from Datamodel.MolDataset import MolPretrainDataset, MolfineDataSet
from Datamodel.data_utils import get_downstream_task_info

# from utils import make_splits, MissingEnergyException
from torch_scatter import scatter

from Datamodel.utils import make_splits, MissingEnergyException


class MolecularDataModule(LightningDataModule):
    def __init__(
            self,
            hparams,
            mode,
            vocab,
    ):
        super().__init__()
        # 初始化缓存
        self.raw_data = None
        self._mean, self._std = None, None
        self.mode=mode
        self.vocab=vocab
        # self.hparams = hparams.__dict__ if hasattr(hparams, "__dict__") else hparams
        self.save_hyperparameters(hparams, ignore=['vocab'])
        # self.true_max_atoms=29
        self.mask_idx = self.vocab.add_symbol("[MASK]", is_special=True)
        self._saved_dataloaders = dict()
    def prepare_data(self):
        self.dataset_root = self.hparams["dataset_root"]
        dataset_name = self.hparams["dataset"]
        if self.mode in ['pretrain1', 'pretrain2','pretrain3']:
            self.raw_data=MolPretrainDataset(
            root=self.dataset_root,
            mode=self.mode,
            vocab=self.vocab,
            pad_idx=self.hparams['pad_idx'],
            noise=self.hparams['noise'],
            noise_type=self.hparams['noise_type'],
            noise_scale=self.hparams['position_noise_scale'],
            mask_prob=self.hparams['mask_prob'],
            max_atoms=self.hparams['max_atoms'],
            fast_read =  False,
            save_cache = True,
        )
        else:
           config_dict = {'task_name':dataset_name, 'root': self.dataset_root}
           print(dataset_name)
           print(self.dataset_root)
           config_dict = get_downstream_task_info(config_dict)
           self.loss_type = config_dict['loss_type']
           self.num_tasks = len(config_dict['target'])
           self.raw_data = MolfineDataSet(
               root=config_dict['root'],
               task_name=config_dict['task_name'],
               target=config_dict['target'],
               mode=self.mode,
               vocab=self.vocab,
               pad_idx=self.hparams['pad_idx'],
               noise=self.hparams['noise'],
               noise_type=self.hparams['noise_type'],
               noise_scale=self.hparams['position_noise_scale'],
               mask_prob=self.hparams['mask_prob'],
               max_atoms=self.hparams['max_atoms'],
               fast_read=False,
               save_cache=True,
           )

        # processed_file = os.path.join(dataset_root, "processed", dataset_name + ".pt")
        # self.raw_data = torch.load(r"D:\work\data_v4.pt")
        # self.raw_data = torch.load(r"D:\work\MolSpectra\scripts\dataset\QM9\processed\qm9_v3.pt")
    def setup(self, stage: str = None):
        """并行操作：数据预处理和分割"""
        if self.mode=='pretrain1':
           # self.dataset=collate_data_denoise(self.raw_data.data)
           self.dataset=Collate_Dataset_Denoise(self.dataset_root,self.raw_data,self.mode)
           # self.dataset=molgraph_to_graph_data_denoise(self.raw_data.data)
        if self.mode == 'pretrain2':
           # self.dataset =Collate_Dataset_Maskll(self.raw_data)
           # self.dataset = Collate_Dataset(self.raw_data)
           # self.dataset = Collate_Dataset(self.raw_data)
           self.dataset = Collate_Dataset_Denoise(self.dataset_root, self.raw_data, self.mode)
        if self.mode == 'pretrain3':
            # self.dataset =Collate_Dataset_Maskll(self.raw_data)
            # self.dataset = Collate_Dataset(self.raw_data)
            # self.dataset = Collate_Dataset(self.raw_data)
            self.dataset = Collate_Dataset_Denoise(self.dataset_root, self.raw_data, self.mode)
        # if 'finetune1' in self.mode:
        #    # self.dataset = Collate_Dataset_fineDenoise(self.dataset_root,self.raw_data)
        #    # self.dataset = Collate_Dataset_finetune(self.raw_data)
        #    self.dataset = Collate_Dataset_fine(self.raw_data)
        if 'finetune' in self.mode:
            self.dataset = Collate_Dataset_finemol(self.raw_data)


        self.idx_train, self.idx_val, self.idx_test = make_splits(
                len(self.dataset),
                self.hparams["train_size"],
                self.hparams["val_size"],
                self.hparams["test_size"],
                self.hparams["seed"],
                join(self.hparams["log_dir"], "splits.npz"),
                self.hparams["splits"],
            )
        print(
                f"train {len(self.idx_train)}, val {len(self.idx_val)}, test {len(self.idx_test)}"
            )

        self.train_dataset = Subset(self.dataset, self.idx_train)

        # If denoising is the only task, test/val datasets are also used for measuring denoising performance.

        self.val_dataset = Subset(self.dataset, self.idx_val)
        self.test_dataset = Subset(self.dataset, self.idx_test)

        if self.hparams["standardize"]:
            self._standardize()

    def train_dataloader(self):
        return self._get_dataloader(self.train_dataset, "train")

    def val_dataloader(self):
        loaders = [self._get_dataloader(self.val_dataset, "val")]
        if (
            len(self.test_dataset) > 0
            and self.trainer.current_epoch % self.hparams["test_interval"] == 0
        ):
            loaders.append(self._get_dataloader(self.test_dataset, "test"))
        return loaders

    def test_dataloader(self):
        return self._get_dataloader(self.test_dataset, "test")

    @property
    def atomref(self):
        if hasattr(self.dataset, "get_atomref"):
            return self.dataset.get_atomref()
        return None

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return self._std

    def _get_dataloader(self, dataset, stage, store_dataloader=True):
        store_dataloader = (
               # store_dataloader and not self.trainer.reload_dataloaders_every_epoch
               store_dataloader and not self.trainer.reload_dataloaders_every_n_epochs
        )
        if stage in self._saved_dataloaders and store_dataloader:
               # storing the dataloaders like this breaks calls to trainer.reload_train_val_dataloaders
               # but makes it possible that the dataloaders are not recreated on every testing epoch
                return self._saved_dataloaders[stage]

        if stage == "train":
              batch_size = self.hparams["batch_size"]
              shuffle = True
        elif stage in ["val", "test"]:
               batch_size = self.hparams["inference_batch_size"]
               shuffle = False

        # if self.mode=="pretrain1":
        dl = DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=self.hparams["num_workers"],
                pin_memory=True,
                drop_last=True
            )

        if store_dataloader:
            self._saved_dataloaders[stage] = dl
        return dl

    def _standardize(self):
        def get_energy(batch, atomref):
            if batch.y is None:
                raise MissingEnergyException()

            if atomref is None:
                return batch.y.clone()

            # remove atomref energies from the target energy
            atomref_energy = scatter(atomref[batch.z], batch.batch, dim=0)
            return (batch.y.squeeze() - atomref_energy.squeeze()).clone()

        data = tqdm(
            self._get_dataloader(self.train_dataset, "val", store_dataloader=False),
            desc="computing mean and std",
        )
        try:
            # only remove atomref energies if the atomref prior is used
            atomref = self.atomref if self.hparams["prior_model"] == "Atomref" else None
            # extract energies from the data
            ys = torch.cat([get_energy(batch, atomref) for batch in data])
        except MissingEnergyException:
            rank_zero_warn(
                "Standardize is true but failed to compute dataset mean and "
                "standard deviation. Maybe the dataset only contains forces."
            )
            return

        # compute mean and standard deviation
        self._mean = ys.mean(dim=0)
        self._std = ys.std(dim=0)
        print(f"y mean: {self.mean}; y std: {self.std}")



class Collate_Dataset_Denoise(InMemoryDataset):
    def __init__(self,root, batch, mode,transform=None, pre_transform=None):
        # self.batch = batch
        super().__init__(root,transform, pre_transform)
        # 直接处理数据并存储
        self.data,self.slices= self.process_data(batch,mode)

    def process_data(self, batch,mode):
        self.data_list = []
        for mol in batch:
            if mode=='pretrain1':
                data = Data(
                    z=mol['z'],
                    pos=mol['pos'],
                    pos_denoise=mol['pos_denoise'],
                    pos_noise=mol['noise'],
                    motif_atoms=mol['motif_atom'].unsqueeze(0),
                    mask_idx=mol['mask_idx'],
                    n_atom=mol['n_atom'],
                )
                self.data_list.append(data)
            elif mode == 'pretrain2':
                data = Data(
                    z=mol['z'],
                    src_z=mol['src_z'].unsqueeze(0),
                    mask_z=mol['mask_z'].unsqueeze(0),
                    coord=mol['coord'].unsqueeze(0),
                    mask_coord=mol['mask_coord'].unsqueeze(0),
                    dist=mol['dist'].unsqueeze(0),
                    mask_dist=mol['mask_dist'].unsqueeze(0),
                    targets=mol['targets'].unsqueeze(0),
                    bond_adj=mol['bond_adj'].unsqueeze(0),
                    src_edge_type=mol['src_edge_type'].unsqueeze(0),
                    pos_denoise=mol['pos_denoise'],
                    pos_noise=mol['noise'],
                    # motif_vocab=mol['motif_vocab'].unsqueeze(0),
                    motif_atoms=mol['motif_atom'].unsqueeze(0),
                    mask_idx=mol['mask_idx'],
                    # mask_z_nopad=mol['mask_z_nopad'],
                    # num_part=mol['num_part'],
                    n_atom=mol['n_atom'],
                )
                self.data_list.append(data)
            elif mode == 'pretrain3':
                data = Data(
                    # z=mol['z'],
                    src_z=mol['src_z'].unsqueeze(0),
                    mask_z=mol['mask_z'].unsqueeze(0),
                    coord=mol['coord'].unsqueeze(0),
                    mask_coord=mol['mask_coord'].unsqueeze(0),
                    dist=mol['dist'].unsqueeze(0),
                    mask_dist=mol['mask_dist'].unsqueeze(0),
                    targets=mol['targets'].unsqueeze(0),
                    bond_adj=mol['bond_adj'].unsqueeze(0),
                    src_edge_type=mol['src_edge_type'].unsqueeze(0),
                    # pos_denoise=mol['pos_denoise'],
                    # pos_noise=mol['noise'],
                    # motif_vocab=mol['motif_vocab'].unsqueeze(0),
                    motif_atoms=mol['motif_atom'].unsqueeze(0),
                    mask_idx=mol['mask_idx'],
                    # mask_z_nopad=mol['mask_z_nopad'],
                    # num_part=mol['num_part'],
                    n_atom=mol['n_atom'],
                )
                self.data_list.append(data)

            # data = Data(
            #     z=mol['z'],
            #     pos=mol['pos'],
            #     pos_denoise=mol['pos_denoise'],
            #     noise=mol['noise'],
            #     idx=mol['idx']
            # )
            # self.data_list.append(data)

        return self.collate(self.data_list)

    def len(self):
        return len(self.data_list)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return []

    def download(self):
        pass

    def process(self):
        pass



class Collate_Dataset(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def len(self):
        return len(self.dataset)

    def get(self, idx):
        mol = self.dataset[idx]
        data = Data(
            z=mol['z'],
            src_z=mol['src_z'].unsqueeze(0),
            mask_z=mol['mask_z'].unsqueeze(0),
            coord=mol['coord'].unsqueeze(0),
            mask_coord=mol['mask_coord'].unsqueeze(0),
            dist=mol['dist'].unsqueeze(0),
            mask_dist=mol['mask_dist'].unsqueeze(0),
            targets=mol['targets'].unsqueeze(0),
            # bond_adj=mol['bond_adj'].unsqueeze(0),
            src_edge_type=mol['src_edge_type'].unsqueeze(0),
            pos_denoise=mol['pos_denoise'],
            pos_noise=mol['noise'],
            motif_z= mol['motif_z'],
            edge_index=mol['edge_index'],
            edge_attr =mol['edge_attr'] ,
            num_part=mol['num_part'],
            # idx=mol['idx']
        )

        return data

class Collate_Dataset_fine(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def len(self):
        return len(self.dataset)

    def get(self, idx):
        mol = self.dataset[idx]
        data = Data(
            z=mol['z'],
            pos_denoise=mol['pos_denoise'],
            pos_noise=mol['noise'],
            labels=mol['labels'].unsqueeze(0)
        )

        return data


class Collate_Dataset_finemol(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def len(self):
        return len(self.dataset)

    def get(self, idx):
        mol = self.dataset[idx]
        data = Data(
            z=mol['z'],
            src_z=mol['src_z'].unsqueeze(0),
            mask_z=mol['mask_z'].unsqueeze(0),
            coord=mol['coord'].unsqueeze(0),
            mask_coord=mol['mask_coord'].unsqueeze(0),
            dist=mol['dist'].unsqueeze(0),
            mask_dist=mol['mask_dist'].unsqueeze(0),
            targets=mol['targets'].unsqueeze(0),
            bond_adj=mol['bond_adj'].unsqueeze(0),
            src_edge_type=mol['src_edge_type'].unsqueeze(0),
            pos_denoise=mol['pos_denoise'],
            pos_noise=mol['noise'],
            # motif_vocab= mol['motif_vocab'].unsqueeze(0),
            motif_atoms=mol['motif_atom'].unsqueeze(0),
            mask_idx=mol['mask_idx'],
            # mask_z_nopad =mol['mask_z_nopad'] ,
            # num_part=mol['num_part'],
            n_atom=mol['n_atom'],
            labels=mol['labels'].unsqueeze(0)
        )

        return data
