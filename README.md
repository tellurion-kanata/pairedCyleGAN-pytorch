#PairedCycleGAN-PyTorch-Implementation

Personal implementation of paper [PairedCycleGAN: Asymmetric Style Transfer for Applying and Removing Makeup](https://gfx.cs.princeton.edu/pubs/Chang_2018_PAS/Chang-CVPR-2018.pdf)

Please organize your training/testing dataset like:  
├─ dataroot  
│  ├─ sketch (real A)  
│  ├─ reference (real B)  

Or you can change the dataset code (in datasets/dataset.py) to fit your training dataset.  

$\mathcal{L}_{p}(G,D_{s})$ introduced in the paper is not adopted and implemented according to my task.

Basic training command:
```
python3 train.py --name [your_project_name] --dataroot [your_training_dataset_path]
```

See options/options.py for more information regarding the command.

Checkpoint files is saved in ./checkpoints/\[project_name\] in default settings.