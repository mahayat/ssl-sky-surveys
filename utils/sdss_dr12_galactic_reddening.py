import numpy as np
import logging

class SDSSDR12Reddening:
  def __init__(self, deredden = False, redden_aug = False, ebv_max = 0.5):
    self.R = np.array([4.239, 3.303, 2.285, 1.698, 1.263]) 
    self.R_dr12 = np.array([5.155, 3.793, 2.751, 2.086, 1.479])
    self.deredden = deredden
    self.redden_aug = redden_aug # apply reddening augmentation
    self.ebv_max = ebv_max

  def __call__(self, data):

    if type(data)==list:
      image = data[0]
      if self.deredden:
        dr12_ext = data[1]
        sfd_ebv = np.mean(dr12_ext/self.R_dr12)
        true_ext = self.R*sfd_ebv
        image = np.float32(image*(10.**(true_ext/2.5))) # deredden image
    else:
      image = data
      if self.deredden:
        logging.error("Dereddening requested but no ebv value passed from dataset loader")
        exit(1)

    if self.redden_aug:
      new_ebv = np.random.uniform(0, self.ebv_max)
      image = np.float32(image*(10.**(-self.R*new_ebv/2.5)))

    return image
