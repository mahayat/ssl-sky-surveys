import os
import argparse
from collections import OrderedDict

import torch
import logging
logging.basicConfig(format='%(levelname)s - %(message)s', level=logging.INFO)

import models.resnet
from utils.YParams import YParams
from utils.data_loader import get_data_loader

def load_model(yaml_config_file='./config/photoz.yaml', config='default', device=torch.cuda.current_device()):
  params = YParams(yaml_config_file, config)

  # setup output directory
  expDir = os.path.join('./expts', config)
  if not os.path.isdir(expDir):
    logging.error("%s not found"%expDir)
    exit(1)

  params['experiment_dir'] = os.path.abspath(expDir)
  params['checkpoint_path'] = os.path.join(expDir, 'checkpoints/ckpt.tar')

  if not os.path.isfile(params.checkpoint_path):
    logging.error("%s not found"%params.checkpoint_path)
    exit(1)

  params.log()
  params['log_to_screen'] = True

  return load(params, device), params

def load(params, device=torch.cuda.current_device()):

  model = models.resnet.resnet50(num_channels=params.num_channels, num_classes=params.num_classes).to(device)

  logging.info("Loading checkpoint %s"%params.checkpoint_path)
  restore_checkpoint(model, params.checkpoint_path)

  return model

def restore_checkpoint(model, checkpoint_path):
  checkpoint = torch.load(checkpoint_path, map_location='cuda:0')

  new_model_state = OrderedDict()
  for key in checkpoint['model_state'].keys():
    if 'module.' in key:
      name = str(key).replace('module.', '')
      new_model_state[name] = checkpoint['model_state'][key]
    else:
      new_model_state[key] = checkpoint['model_state'][key]

  model.load_state_dict(new_model_state)
  logging.info("Chekpoint loaded. Checkpoint epoch %d"%checkpoint['epoch'])
