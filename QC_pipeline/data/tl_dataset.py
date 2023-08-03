from pytorch_lightning import LightningDataModule
from .torch_datasets import TextClassificationDataset
from torch.utils.data import DataLoader
from typing import List, Optional, Union
from os.path import join as opj


class E5Dataset(LightningDataModule):
    """
    PyTorch Lightning data module 

    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer_name: str,
        max_len: int,

        train_batch_size: int = 10,
        val_batch_size: int = 10,
        num_workers: int = 1,
        pin_memory: bool = False,
        transform=None,
        **kwargs,
    ):
        super().__init__()

        self.data_path = data_path
        self.tokenizer_name = tokenizer_name
        self.max_len = max_len

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.transform = transform

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = TextClassificationDataset(
            data_path=opj(self.data_path, 'train.csv'),
            tokenizer_name=self.tokenizer_name,
            max_len=self.max_len,
        )
        self.val_dataset = TextClassificationDataset(
            data_path=opj(self.data_path, 'val.csv'),
            tokenizer_name=self.tokenizer_name,
            max_len=self.max_len,
        )
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )
    
    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )