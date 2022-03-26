
from parser import parse_args

hyper_params = {}
args = parse_args()
for arg in vars(args):
    hyper_params[arg] = getattr(args, arg)

from Models.Teacher import Teacher as Teacher

# ********************* model select *****************************
from Dataset.dataset import Dataset as Dataset
from Dataset.dataset import TestDataset as TestDataset