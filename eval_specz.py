import os
import argparse
from collections import OrderedDict

import torch
import logging
logging.basicConfig(format='%(levelname)s - %(message)s', level=logging.INFO)

import models.resnet
from utils.YParams import YParams
from utils.data_loader import get_data_loader
from utils.load_trained_model import load_model

def eval_specz(model, data_loader):
  model.eval()

  device = torch.cuda.current_device()

  correct = 0.0
  with torch.no_grad():
    for data in data_loader:
      images, labels = map(lambda x: x.to(device), data)
      outputs = model(images)
      _, preds = outputs.max(1)
      correct += preds.eq(labels).sum().float()/labels.shape[0]

  logs = {'acc1': correct/len(data_loader)}

  logging.info('Eval acc1={}'.format(logs['acc1']))

  return logs

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--yaml_config", default='./config/photoz.yaml', type=str)
  parser.add_argument("--config", default='default', type=str)
  args = parser.parse_args()

  model, params = load_model(os.path.abspath(args.yaml_config), args.config)
  data_loader, _  = get_data_loader(params, params.valid_data_path, distributed=False, load_specz=True, is_train=False)

  eval_specz(model, data_loader)
