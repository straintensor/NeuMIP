# NeuMIP: Multi-Resolution Neural Materials
## SIGGRAPH 2021



https://cseweb.ucsd.edu/~viscomp/projects/NeuMIP

## Setup and prerequisites 

The code was developed on Ubuntu 18.04, using Python 3.7 and PyTorch 1.7.
Although, we don't use any unique features of those systems, running on different configuration might require some small adjustments.

First you need to setup your data path, where all the datasets, models, etc. reside. Go to `path_config.py` and at line 9 change:
```path_base = "$YOUR_PATH``` to your path. 

## License
The code is licensed under BSD license. 


## Dataset
Here's the link to the datasets and models:
https://drive.google.com/drive/folders/1EbzMlpmE7f49jTs2IY7oMQKZAGAfl3M5

## Training and Running

To train:
```
./rung.sh 0 ./neural_rendering.py --dataset DATASET_NAME  --max_iter 30000 --outm  MODEL_NAME  --experiment  StandardRawLongShadowMaskOnly  --batch 4  --loss comb2
 ```

 To visualize it:
```
./run.sh neural_rendering.py --inm wool2.pth --vis 
```


The main code is in the neural_rendering.py file. Beware, the code supports different configurations/architectures. Please, use  -experiment  StandardRawLongShadowMaskOnly 
For loss, use l1 or comb2, which is better for specular.