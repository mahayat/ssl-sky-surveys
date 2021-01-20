### Self-Supervised Representation Learning for Astronomical Images

This repo contains a PyTorch implementation to produce the results in "Self-Supervised Representation Learning for Astronomical Images" paper ([arXiv:2012.13083](https://arxiv.org/abs/2012.13083)).

```
@Article{hayat2020selfsupervised,
      author={Md Abul Hayat and George Stein and Peter Harrington and Zarija Luki\'{c} and Mustafa Mustafa},
      title={Self-Supervised Representation Learning for Astronomical Images},
      year={2020},
      eprint={2012.13083},
      archivePrefix={arXiv}
}
```

**Code Status**: you can use this code to load pretrained checkpoints and train supervised baselines. Pretraining code will be added soon.  

#### Pretrained checkpoints:

To download pretrained checkpoints:  

1. Pretrained model:
```bash
wget https://portal.nersc.gov/project/dasrepo/self-supervised-learning-sdss/checkpoints/pretrained_paper_model.pth.tar
```

2. Fine-tuned photo-z model:
```bash
wget https://portal.nersc.gov/project/dasrepo/self-supervised-learning-sdss/checkpoints/photoz_finetuned_model.pth.tar
```

3. Supervised baseline photo-z model:
```bash
wget https://portal.nersc.gov/project/dasrepo/self-supervised-learning-sdss/checkpoints/photoz_supervised_baseline_model.pth.tar
```

#### Data:

Our database of galaxies is assembled from Data Release 12 (DR12; [Eisenstein et al. 2011](https://ui.adsabs.harvard.edu/abs/2011AJ....142...72E/abstract)) of The Sloan Digital Sky Survey: Mapping the Universe ([SDSS](https://www.sdss.org/)). Data puuling and pre-processing procedure is described in the paper appendix.  

You can download the validation dataset by following:

```bash
wget https://portal.nersc.gov/project/dasrepo/self-supervised-learning-sdss/datasets/sdss_w_specz_valid.h5
```

The full training dataset will be available soon.

>  Acknowledgment: Funding for SDSS-III has been provided by the Alfred P. Sloan Foundation, the Participating Institutions, the National Science Foundation, and the U.S. Department of Energy Office of Science. The SDSS-III web site is http://www.sdss3.org/. DSS-III is managed by the Astrophysical Research Consortium for the Participating Institutions of the SDSS-III Collaboration including the University of Arizona, the Brazilian Participation Group, Brookhaven National Laboratory, Carnegie Mellon University, University of Florida, the French Participation Group, the German Participation Group, Harvard University, the Instituto de Astrofisica de Canarias, the Michigan State/Notre Dame/JINA Participation Group, Johns Hopkins University, Lawrence Berkeley National Laboratory, Max Planck Institute for Astrophysics, Max Planck Institute for Extraterrestrial Physics, New Mexico State University, New York University, Ohio State University, Pennsylvania State University, University of Portsmouth, Princeton University, the Spanish Participation Group, University of Tokyo, University of Utah, Vanderbilt University, University of Virginia, University of Washington, and Yale University.  

#### Examples:
1. [Load pretrained representations](notebooks/example_load_pretrained_representations.ipynb)  
1. [Photo-z example](notebooks/example_photoz_model.ipynb)  
