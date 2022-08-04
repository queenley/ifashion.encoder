# Datasets

Download SCHP extracted Lookbook training dataset on PC0 at:

`
/home/nttai/Cerebro/fashion.encoder/tfrecords-lookbook-atr.zip
`

For embedding database, download unparse mixed FashionIQ and Lookbook datasets
on PC0 at:

`
/home/nttai/Cerebro/fashion.encoder/tfrecords-fashioniq-lookbook.zip
`

SCP could be utilize to download to your local machine, for example:

`
scp -P 10022 <USERNAME>@pc0.cerebro.host:<PATH>
`


# Training

Extract training data, modify respective paths to tfrecods in config file
and run:

`
python3 train.py --cfg_path ./configs/arc_res50_lookbook.yaml
`

An `encoder.h5` file will be saved at root dir after training done
