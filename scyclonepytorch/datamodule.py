from typing import Optional
from os import cpu_count

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data import random_split, DataLoader
from pytorch_lightning import LightningDataModule
from .audioDataset import audioDataset


class DataLoaderPerformance:
    """PyTorch DataLoader performance configs.

    All attributes which affect performance of [torch.utils.data.DataLoader][^DataLoader] @ v1.6.0
    [^DataLoader]:https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    """

    def __init__(self, num_workers: Optional[int] = None, pin_memory: bool = True) -> None:
        """Default: num_workers == cpu_count & pin_memory == True
        """

        # Design Note:
        #   Current performance is for single GPU training.
        #   cpu_count() is not appropriate under the multi-GPU condition.

        if num_workers is None:
            c = cpu_count()
            num_workers = c if c is not None else 0
        self.num_workers = 2
        self.pin_memory = pin_memory


class NonParallelSpecDataModule(LightningDataModule):
    def __init__(
        self,
        sampling_rate,
        batch_size = 64,
        performance = None,
        adress_data_root = None,
    ):
        super().__init__()
        self._sampling_rate = sampling_rate
        self.batch_size = batch_size

        if performance is None:
            performance = DataLoaderPerformance()
        self._num_worker = performance.num_workers
        self._pin_memory = performance.pin_memory

        self._adress_dir_corpuses = None
        self._adress_dir_datasets = None

    def prepare_data(self, *args, **kwargs) -> None:
        NonParallelSpecDataset("scyclonepytorch/data/dataset", "scyclonepytorch/data/dataset_jsut", train=True)

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            dataset_full = NonParallelSpecDataset("scyclonepytorch/data/dataset", "scyclonepytorch/data/dataset_jsut", train=True)

            # use modulo for validation (#training become batch*N)
            n_full = len(dataset_full)
            mod = n_full % self.batch_size
            self.dataset_train, self.dataset_val = random_split(
                dataset_full, [n_full - mod, mod]
            )
            self.batch_size_val = mod
        if stage == "test" or stage is None:
            self.dataset_test = NonParallelSpecDataset("scyclonepytorch/data/testset_api", "scyclonepytorch/data/testset_jsut", train=False)
            self.batch_size_test = self.batch_size

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self._num_worker,
            pin_memory=self._pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size_val,
            num_workers=self._num_worker,
            pin_memory=self._pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset_test,
            batch_size=self.batch_size_test,
            num_workers=self._num_worker,
            pin_memory=self._pin_memory,
        )

class NonParallelSpecDataset(Dataset):
    def __init__(self, path_A: str, path_B: str, train: bool):
        self.train = train
        self.datasetA = audioDataset(path_A, train)
        if train == True:
            self.datasetB = audioDataset(path_B, train)

    def __getitem__(self, n: int):
        """Load the n-th sample from the dataset.
        Potential problem: A/B pair
        Current implementation yield fixed A/B pair.
        When batch size is small (e.g. 1), Batch_A and Batch_B has strong correlation.
        If big batch, correlation decrease so little problem.
        We could solve this problem through sampler (e.g. sampler + sampler reset).
        """
        # ignore label
        if self.train == True:
            return (self.datasetA[n], self.datasetB[n])
        else:
            return self.datasetA[n]

    def __len__(self) -> int:
        if self.train == True:
            return min(len(self.datasetA), len(self.datasetB))
        else:
            return len(self.datasetA)


# class NonParallelSpecBigDataset(Dataset):
#     def __init__(self,
#         train: bool,
#         sampling_rate: int,
#         adress_dir_corpuses: Optional[str] = None,
#         adress_dir_datasets: Optional[str] = None,
#     ):
#         """Non-parallel spectrogram dataset with large datasets.
#         """

#         # Design Note:
#         #   Sampling rates of dataset A and B should match, so `sampling_rate` is not a optional, but required argument.

#         if train:
#             subtypes_a = ["basic5000"]
#             subtypes_b = ["short-form/basic5000"]
#         else:
#             subtypes_a = ["voiceactress100"]
#             subtypes_b = ["short-form/voiceactress100"]

#         # adress_dir_corpuses = "s3:machine-learning/corpuses"
#         corpus_archive_JSUT = adress_dir_corpuses + "/JSUT.zip" if adress_dir_corpuses else None
#         corpus_archive_JSSS = adress_dir_corpuses + "/JSSS.zip" if adress_dir_corpuses else None

#         # adress_dir_datasets = "s3:machine-learning/datasets"
#         dataset_dir_JSUT_spec = adress_dir_datasets + "/JSUT_spec" if adress_dir_datasets else None
#         dataset_dir_JSSS_spec = adress_dir_datasets + "/JSSS_spec" if adress_dir_datasets else None

#         self.datasetA = JSUT_spec(train, download_corpus=True, transform=pad_clip, subtypes=subtypes_a,
#             corpus_adress=corpus_archive_JSUT,
#             dataset_dir_adress=dataset_dir_JSUT_spec,
#             resample_sr=sampling_rate
#         )
#         self.datasetB = JSSS_spec(train, download_corpus=True, transform=pad_clip, subtypes=subtypes_b,
#             corpus_adress=corpus_archive_JSSS,
#             dataset_dir_adress=dataset_dir_JSSS_spec,
#             resample_sr=sampling_rate
#         )

#     def __getitem__(self, n: int):
#         """Load the n-th sample from the dataset.

#         Potential problem: A/B pair
#         Current implementation yield fixed A/B pair.
#         When batch size is small (e.g. 1), Batch_A and Batch_B has strong correlation.
#         If big batch, correlation decrease so little problem.
#         We could solve this problem through sampler (e.g. sampler + sampler reset).
#         """
#         # ignore label
#         return (self.datasetA[n][0], self.datasetB[n][0])

#     def __len__(self) -> int:
#         return min(len(self.datasetA), len(self.datasetB))

if __name__ == "__main__":
    # test for clip
    i = torch.zeros(2, 2, 190, 200)
    o = pad_clip(i)
    print(o.size())