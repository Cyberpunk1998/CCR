# CCR

This repository contains the dataset link and the code for our paper [CCR: A Counterfactual Causal Reasoning-based Method for Cross-view Geo-localization](https://ieeexplore.ieee.org/document/link链接), IEEE Transactions on Circuits and Systems for Video Technology. Thank you for your interest in this work.

The article has been accepted and is awaiting formal publication.


## Requirement
1. Download the [University-1652](https://github.com/layumi/University1652-Baseline) dataset
2. Download the [SUES-200](https://github.com/Reza-Zhu/SUES-200-Benchmark) dataset
3. Configuring the environment
   * First you need to configure the torch and torchision from the [pytorch](https://pytorch.org/) website
   * ```shell
     pip install -r requirement.txt
     ```

## About dataset
The organization of the dataset.

More detailed about Univetsity-1652 dataset structure:
```
├── University-1652/
│   ├── train/
│       ├── drone/                   /* drone-view training images 
│           ├── 0001
│           ├── 0002
│           ...
│       ├── street/                  /* street-view training images 
│       ├── satellite/               /* satellite-view training images       
│       ├── google/                  /* noisy street-view training images (collected from Google Image)
│   ├── test/
│       ├── query_drone/  
│       ├── gallery_drone/  
│       ├── query_street/  
│       ├── gallery_street/ 
│       ├── query_satellite/  
│       ├── gallery_satellite/ 
│       ├── 4K_drone/
```
More detailed about SUES-200 dataset structure:
```
├── SUES-200/
│   ├── train/
│       ├── 150/
│           ├── drone/                   /* drone-view training images 
│               ├── 0001
│               ├── 0002
│               ...
│           ├── satellite/               /* satellite-view training images       
│       ├── 200/                  
│       ├── 250/  
│       ├── 300/  
│   ├── test/
│       ├── 150/  
│           ├── query_drone/  
│           ├── gallery_drone/  
│           ├── query_satellite/  
│           ├── gallery_satellite/ 
│       ├── 200/  
│       ├── 250/  
│       ├── 300/  
```

## Model weights
You can find our model weights in the **model/CCR_Model_University** folder. The weighting applies only to University-1652.

If you want to use this weight, replace the **models** folder with the **model/CCR_Model_University/models** folder.

## Train and Test
We provide scripts to complete CCR training and testing
* Change the **data_dir** and **test_dir** paths in **run.sh** and then run:

For University-1652:
```shell
bash run_university.sh
```
For SUES-200:
```shell
bash run_sues.sh
```

* Change the **data_dir** and **test_dir** paths in **test.sh** and then test:
```shell
bash test_university.sh
```


## Citation

```bibtex
CCR: A Counterfactual Causal Reasoning-based Method for Cross-view Geo-localization}}
```
