# DARTS
This is a pytorch implementation of the DARTS.

Since the [offical released code](https://github.com/quark0/darts.git) does not support the pytorch 0.4 or later, here we reimplement the DARTS with pytorch 1.0.

# search model
    nohup python3 train_search.py > train_search.log &

# training searched model
    nohup python3 train.py > train_model.log &
