import os
import argparse
from collections import OrderedDict
import numpy as np
from astropy.stats import mad_std

import torch
import logging
logging.basicConfig(format='%(levelname)s - %(message)s', level=logging.INFO)

import models.resnet
from utils.YParams import YParams
from utils.data_loader import get_data_loader
from utils.load_trained_model import load_experiment

def astro_stats(params, pdfs, speczs):
  bin_width = params.specz_upper_lim/params.num_classes
  span = (bin_width/2) + bin_width*np.arange(0, params.num_classes)

  photozs = np.sum((pdfs*span), axis = 1)
  delzs = (photozs-speczs)/(1+speczs)
  madstd = mad_std(delzs)
  th = 0.05
  eta = np.sum(abs(delzs)>th)/delzs.shape[0]

  logging.info('Mean of delz: {}'.format(np.float32(np.mean(delzs))))
  logging.info('sigma_MAD: {}'.format(np.float32(madstd)))
  logging.info('eta (|del z|>0.05): {}'.format(100*eta))

def eval_specz(params, model, data_loader, save_pdfs=False):
  model.eval()

  device = torch.cuda.current_device()

  softmax = torch.nn.Softmax(dim = 1)
  batch_size = params.valid_batch_size_per_gpu
  n_samples = batch_size * len(data_loader)
  pdfs = np.ndarray((n_samples, params.num_classes), dtype=np.float32)
  speczs = np.ndarray(n_samples, dtype=np.float32)

  correct = 0.0
  with torch.no_grad():
    for idx, data in enumerate(data_loader):
      images = data[0].to(device)
      specz_bin = data[1]
      specz = data[2]
      outputs = model(images)

      pdfs[idx*batch_size:(idx+1)*batch_size,:] = softmax(outputs).cpu().detach().numpy()
      speczs[idx*batch_size:(idx+1)*batch_size] = specz.numpy()

      _, preds = outputs.max(1)
      preds = preds.detach().cpu()
      correct += preds.eq(specz_bin).sum().float()/specz_bin.shape[0]

  logs = {'acc1': correct/len(data_loader)}

  logging.info('>> Performance >>')
  logging.info('Eval acc1={}'.format(logs['acc1']))
  astro_stats(params, pdfs, speczs)

  if save_pdfs:
    outdir = os.path.join(params.experiment_dir, 'pdfs_specz/')
    if not os.path.isdir(outdir):
      os.makedirs(outdir)

    np.save(os.path.join(outdir, 'pdfs.npy'), pdfs)
    np.save(os.path.join(outdir, 'speczs.npy'), speczs)

  return logs

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--yaml_config", default='./config/photoz.yaml', type=str)
  parser.add_argument("--config", default='default', type=str)
  args = parser.parse_args()

  model, params = load_experiment(os.path.abspath(args.yaml_config), args.config)
  data_loader, _  = get_data_loader(params, params.valid_data_path, distributed=False, load_specz=True, is_train=False)

  eval_specz(params, model, data_loader, True)
