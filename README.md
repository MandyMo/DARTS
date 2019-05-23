# DARTS
This is a pytorch implementation of the DARTS.

Since the [offical released code](https://github.com/quark0/darts.git) does not support the pytorch 0.4 or later, here we reimplement the DARTS with pytorch 1.0.

# search model
    nohup python3 train_search.py > train_search.log &

# training searched model
    nohup python3 train.py > train_model.log &

# costs
We train the search model via a single Tesla V100 GPU for 4 days, while we train the model on cifar-10 for 2 days with two Telsa v100 GPUS. The normal.pdf and reduce.pdf are the normal cell structure and the reduce structure respectively.
