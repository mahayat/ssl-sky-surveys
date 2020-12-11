import os
import time
import numpy as np
import torch
from YParams import YParams
from data_loader import get_data_loader

params = YParams(os.path.abspath('config/photoz.yaml'), 'default')
params.train_data_path = "/global/cscratch1/sd/mustafa/SDSS/data_20200901/train.h5"
params.num_data_workers = 1
params.batch_size = 4
params.valid_batch_size_per_gpu = 4

dataset, sampler = get_data_loader(params, params.train_data_path, False, is_train=False)

for epoch in range(1):
  for data in dataset:
    print(data[0].shape, data[0].dtype, torch.sum(data[0]))
