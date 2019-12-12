### Learning Binding Affinities from Docking 
Docking is a small and cheap (0.1 - 0.01 GPU hours simulation) that provides pose and interaction energy for 2 bodies. 
The inspiration paper: Ultra-large library docking for discovering new chemotypes, Jiankun Lyi and others, Nature 2019

### Structure
d4 - dopamine receptor protein files
   
   dock6 files - files for docking new molecules to D4 with dock6
   
   raw         - raw parquet files with dockscores for learning
   
   processed   - graphs processed from raw files
   
   d4_100k_mine_model001 - trained model for energy prediction

### Join large files if provided as parts
https://linoxide.com/linux-how-to/split-large-text-file-smaller-files-linux/

```split: split -b 100MB ampc_100k_optim.parquet```

To make the dataset:

```cat x* > ampc_100k_optim.parquet```
