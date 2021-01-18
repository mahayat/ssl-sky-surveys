import logging
import glob
import torch
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as transforms
import skimage.transform
import h5py
from utils.sdss_dr12_galactic_reddening import SDSSDR12Reddening

class RandomRotate:
  def __call__(self, image):
    # print("RR", image.shape, image.dtype)
    return (skimage.transform.rotate(image, np.float32(360*np.random.rand(1)))).astype(np.float32)

class JitterCrop:
  def __init__(self, outdim, jitter_lim=None):
    self.outdim = outdim
    self.jitter_lim = jitter_lim
    self.offset = self.outdim//2

  def __call__(self, image):
    # print("JC", image.shape, image.dtype)
    center_x = image.shape[0]//2
    center_y = image.shape[0]//2
    if self.jitter_lim:
      center_x += int(np.random.randint(-self.jitter_lim, self.jitter_lim+1, 1))
      center_y += int(np.random.randint(-self.jitter_lim, self.jitter_lim+1, 1))

    return image[(center_x-self.offset):(center_x+self.offset), (center_y-self.offset):(center_y+self.offset)]

def get_data_loader(params, files_pattern, distributed, is_train, load_specz):
  if is_train:
    transform = transforms.Compose([SDSSDR12Reddening(deredden=True),
                                    RandomRotate(),
                                    JitterCrop(outdim=params.crop_size, jitter_lim=params.jc_jit_limit),
                                    transforms.ToTensor()])
  else:
    transform = transforms.Compose([SDSSDR12Reddening(deredden=True),
                                    JitterCrop(outdim=params.crop_size),
                                    transforms.ToTensor()])

  dataset = SDSSDataset(params.num_classes, files_pattern, transform, load_specz, True, params.specz_upper_lim)
  sampler = DistributedSampler(dataset, shuffle=True) if distributed else None
  dataloader = DataLoader(dataset,
                          batch_size=int(params.batch_size) if is_train else int(params.valid_batch_size_per_gpu),
                          num_workers=params.num_data_workers,
                          shuffle=(sampler is None),
                          sampler=sampler,
                          drop_last=True,
                          pin_memory=torch.cuda.is_available())

  return dataloader, sampler

class SDSSDataset(Dataset):
  def __init__(self, num_classes, files_pattern, transform, load_specz, load_ebv, specz_upper_lim=None):
    self.num_classes = num_classes
    self.files_pattern = files_pattern
    self.transform = transform
    self.load_specz = load_specz
    self.load_ebv = load_ebv
    self.specz_upper_lim = specz_upper_lim
    self._get_files_stats()

  def _get_files_stats(self):
    self.files_paths = glob.glob(self.files_pattern)
    self.n_files = len(self.files_paths)
    with h5py.File(self.files_paths[0], 'r') as _f:
      self.n_samples_per_file = _f['images'].shape[0]
    self.n_samples = self.n_files * self.n_samples_per_file

    self.files = [None for _ in range(self.n_files)]
    logging.info("Found {} at path {}. Number of examples: {}".format(self.n_files, self.files_pattern, self.n_samples))

  def _open_file(self, ifile):
    self.files[ifile] = h5py.File(self.files_paths[ifile], 'r')

  def __len__(self):
    return self.n_samples

  def __getitem__(self, global_idx):
    ifile = int(global_idx/self.n_samples_per_file)
    local_idx = int(global_idx%self.n_samples_per_file)

    if not self.files[ifile]:
      self._open_file(ifile)

    if self.load_specz:
      specz = self.files[ifile]['specz_redshift'][local_idx]
      # hard-coded numbers are specific to the dataset used in this tutorial
      if specz >= self.specz_upper_lim:
        specz = self.specz_upper_lim - 1e-6
      specz_bin = torch.tensor(int(specz//(self.specz_upper_lim/self.num_classes)))

    # we flip channel axis because all tranforms we use assume HWC input,
    # last transform, ToTensor, reverts this operation
    image = np.swapaxes(self.files[ifile]['images'][local_idx], 0, 2)

    if self.load_ebv:
      ebv = self.files[ifile]['e_bv'][local_idx]
      out = [image, ebv]
    else:
      out = image

    if self.load_specz:
      return self.transform(out), specz_bin, torch.tensor(specz)
    else:
      return self.transform(out)
