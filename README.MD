# NeuMIP: multi-resolution neural materials

https://cseweb.ucsd.edu//~viscomp/projects/NeuMIP/

## Setup and prerequisites 

The code was developed on Ubuntu 18.04, using Python 3.7 and PyTorch 1.7.
Although, we don't use any unique features of those systems, running on different configuration might require some small adjustments.

First you need to setup your data path, where all the datasets, models, etc. reside. Go to `path_config.py` and at line 9 change:
```path_base = "$YOUR_PATH``` to your path. 

## License
The code is licensed under BSD license. 


## Training and Running

```python3.7 ./neural_rendering.py 
--dataset dataset_name.hdf5  
--outm  my_model.pth  
 --loss l1```

