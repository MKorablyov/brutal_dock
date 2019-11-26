https://linoxide.com/linux-how-to/split-large-text-file-smaller-files-linux/
split: split -b 100MB ampc_100k_optim.parquet

To make the dataset:
cat x* > ampc_100k_optim.parquet




