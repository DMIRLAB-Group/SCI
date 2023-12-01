r"""
This data module includes 11 GOOD datasets and a dataloader for an organized data loading process.
"""
from GOOD.data.dataset_manager import load_dataset, create_dataloader, read_meta_info
from .good_datasets import *
from .good_loaders import *
