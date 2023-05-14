# **
# * ����
# *
# * @author 雁楚
# * @edit 雁楚

from .toy_data_loader import is_builtin
from .toy_data_loader import load_builtin
from .synthetic_data_generator import generate_datasets
from .easydataset import EasyDataset

__all__ = ['EasyDataset', 'is_builtin', 'load_builtin', 'generate_datasets']
