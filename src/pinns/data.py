from pathlib import Path
import pickle
from typing import Callable
from abc import ABC, abstractmethod
from tqdm import tqdm
import math

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from skimage.measure import block_reduce


from src.dataset_creation.statistics import Metadata, DatasetStats


def load_dataset_stats(dataset_location: str | Path):
    dataset_location = Path(dataset_location)

    pickle_file_path = dataset_location / "gprmax_input_files/metadata/all_data.pkl"

    with open(pickle_file_path, 'rb') as f:
        stats : DatasetStats = pickle.load(f)

    return stats

class DatasetCreator():

    def __init__(self, dataset_location: str | Path, split_percentages: list[float] = [70, 10, 20], seed: int = 42):
        self.dataset_location = dataset_location
        self.dataset_stats = load_dataset_stats(dataset_location)
        rng = np.random.default_rng(seed)
        all_samples = list(self.dataset_stats.stats.keys())
        rng.shuffle(all_samples)

        if sum(split_percentages) != 100:
            raise ValueError("Split percentages must sum to 100!")

        train_index = split_percentages[0] * len(all_samples) // 100
        val_index = ((split_percentages[0] + split_percentages[1]) * len(all_samples)) // 100

        self.train_set = all_samples[:train_index]
        self.val_set = all_samples[train_index:val_index]
        self.test_set = all_samples[val_index:]

        print("train samples:", len(self.train_set))
        print("val samples:", len(self.val_set))
        print("test samples:", len(self.test_set))

    def filter(self, splits: list[str], predicate: Callable[[Metadata], bool]) -> None:
        """
        Filters the specified splits based on the return value of the predicate

        Parameters
        ----------
        splits : list[str]
            the splits to filter, "train, "val", "test" or "all".
        predicate : Callable[Metadata, ]
            the function to apply to the Metadata object to filter the split.
            it returns True if the object should be kept, False if it should be discarded.
        """

        if splits == "all":
            splits = ["train", "val", "test"]

        if "train" in splits:
            for s in self.train_set.copy():
                if not predicate(self.dataset_stats.stats[s]):
                    self.train_set.remove(s)
        
        if "val" in splits:
            for s in self.val_set.copy():
                if not predicate(self.dataset_stats.stats[s]):
                    self.val_set.remove(s)
        
        if "test" in splits:
            for s in self.test_set.copy():
                if not predicate(self.dataset_stats.stats[s]):
                    self.test_set.remove(s)
        
        print(f"successfully filtered {splits} splits:")
        print("train samples:", len(self.train_set))
        print("val samples:", len(self.val_set))
        print("test samples:", len(self.test_set))

    @classmethod
    def construct_paths(cls, dataset_location: str | Path, sample_names: list[str]):

        output_path = Path(dataset_location) / "gprmax_output_files"
        paths = []
        for sample_name in sample_names:
                sample_folder = output_path / sample_name
                sample_paths = {
                    "dir": sample_folder,
                    "snapshots": sample_folder / "snapshots.npz",
                    "geometry": sample_folder / (sample_name + "_geometry.npy"),
                    "output": sample_folder / (sample_name + "_merged.out"),
                    "view": sample_folder / (sample_name + "_view.vti")
                }
                if not sample_paths["snapshots"].exists():
                    raise ValueError()
                if not sample_paths["geometry"].exists():
                    raise ValueError()
                if not sample_paths["output"].exists():
                    raise ValueError()
                if not sample_paths["view"].exists():
                    raise ValueError()
                paths.append(sample_paths)
        
        return paths

    def get_splits_paths(self) -> tuple[list[dict[str, Path]], list[dict[str, Path]], list[dict[str, Path]]]:
        """
        Constructs and returns the samples paths for geometry, raw output, snapshots and view file.
        """

        dataset_paths = []

        for i, split in enumerate([self.train_set, self.val_set, self.test_set]):
            paths = self.construct_paths(self.dataset_location, split)
            dataset_paths.append(paths)

        return tuple(dataset_paths)

class GPRDataset(ABC, Dataset):
    def __init__(self, samples_dict: list[dict[str, Path]]):
        super().__init__()
        self.samples_dict = samples_dict

    @abstractmethod
    def __len__(self): pass

    @abstractmethod
    def __getitem__(self, index): pass

class InMemoryDataset(GPRDataset):
    def __init__(self, samples_dict: list[dict[str, Path]]):
        super().__init__(samples_dict)
        
        # load all the dataset into memory:
        snapshots = []
        times = []
        geometries = []
        print("Loading into memory...")
        for d in tqdm(samples_dict):
            snaps = np.load(d["snapshots"])["00000_E"]
            t = np.load(d["snapshots"])["00000_times"]
            geom = np.load(d["geometry"])

            snapshots.append(snaps)
            times.append(t)
            geometries.append(geom)
        
        snapshots = np.asarray(snapshots, dtype=np.float32)
        times = np.asarray(times, dtype=np.float32)
        geometries = np.asarray(geometries, dtype=np.float32)

        self.snapshots = torch.from_numpy(snapshots)
        self.times = torch.from_numpy(times)
        self.geometries = torch.from_numpy(geometries)

        print("snapshots:", snapshots.shape)
        print("times:", times.shape)
        print("geometries:", geometries.shape)

    def __len__(self):
        return math.prod(self.snapshots.shape)

    def __getitem__(self, index):

        t_shape = self.snapshots.shape[1]
        y_shape = self.snapshots.shape[2]
        x_shape = self.snapshots.shape[3]

        x_index = index % x_shape
        y_index = (index // x_shape) % y_shape
        t_index = index // (x_shape * y_shape) % t_shape
        n_index = index // (x_shape * y_shape * t_shape)

        # one cell is spatially 0.006m both in x and y directions
        x = torch.tensor([x_index], dtype=torch.float32) * 0.006
        y = torch.tensor([y_index], dtype=torch.float32) * 0.006
        t = self.times[n_index][t_index]
        u = self.snapshots[n_index][t_index][y_index][x_index]

        return x.squeeze(), y.squeeze(), t, u, n_index

class StorageDataset(GPRDataset):
    def __init__(self, samples_dict: list[dict[str, Path]], snapshot_shapes = [24, 284, 250]):
        super().__init__(samples_dict)
        self.snapshot_shapes = snapshot_shapes
        
    def __len__(self):
        return math.prod(self.snapshot_shapes) * len(self.samples_dict)
    
    def __getitem__(self, index):
        
        t_shape = self.snapshot_shapes[0]
        y_shape = self.snapshot_shapes[1]
        x_shape = self.snapshot_shapes[2]

        x_index = index % x_shape
        y_index = (index // x_shape) % y_shape
        t_index = index // (x_shape * y_shape) % t_shape
        n_index = index // (x_shape * y_shape * t_shape)

        d = self.samples_dict[n_index]

        snaps = np.load(d["snapshots"])["00000_E"]
        times = np.load(d["snapshots"])["00000_times"]
        # geom = np.load(d["geometry"])

        # geometries have not been block reduced
        # geom = block_reduce(geom, block_size=(1, 3, 3), func=np.mean)
        
        x = torch.tensor([x_index], dtype=torch.float32) * 0.006
        y = torch.tensor([y_index], dtype=torch.float32) * 0.006
        t = times[t_index]
        u = snaps[t_index][y_index][x_index]

        return x, y, t, u, n_index

    def load_geometries(self) -> torch.Tensor:
        geometries = []
        print("Loading geometries...")
        for d in tqdm(self.samples_dict):
            geom = np.load(d["geometry"])
            geometries.append(geom)
        
        geometries = np.asarray(geometries, dtype=np.float32)
        self.geometries = torch.from_numpy(geometries)
        return self.geometries




if __name__ == "__main__":

    creator = DatasetCreator("dataset")
    DEVICE = "cuda:0"

    filter = lambda x : x.track_type == "AC_rail"
    creator.filter("train", filter)

    train_paths = creator.get_splits_paths()[0]

    dataset = InMemoryDataset(train_paths[:100])
    # dataset = StorageDataset(train_paths[:100])

    loader = DataLoader(dataset, batch_size = 8192, num_workers=64)

    from src.pinns.models import PINN4GPR
    model = PINN4GPR().to(DEVICE)
    # 
    geometry_embeddings = torch.randn((100, 128)).to(DEVICE)

    for batch in tqdm(loader):
        xs, ys, ts, us, geometry_indexes = batch

        xs = xs.to(DEVICE)
        ys = ys.to(DEVICE)
        ts = ts.to(DEVICE)
        us = us.to(DEVICE)
        geometry_indexes = geometry_indexes.to(DEVICE)

        geometries = geometry_embeddings[geometry_indexes]

        mlp_inputs = torch.stack([xs, ys, ts], dim=-1)

        output = model.cnn_embedding_forward(mlp_inputs, geometries)
