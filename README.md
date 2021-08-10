# FrictGAN: Image-to-Friction Generation 

This is the implementation of the paper [GAN-based Image-to-Friction Generation for Tactile Simulation of Fabric Material](https://github.com/shaoyuca)
![image](https://github.com/shaoyuca/Image-to-Friction-Generation/blob/main/dataset/img.png) 

## Setup

We run the program on a Linux desktop using python.

Environment requirements: 

- tensorflow 2.1.0  
- tensorlfow-addons 0.12.0  
- tensorlfow-io 0.17.0  
- librosa 0.8.0  
- scipy 1.4.1  
- opencv 4.5.1  

## Usage

- Train the model:
```bash
python FrictganNet.py --train --epoch <number>
```

- Test the model:
```bash
python FrictganNet.py --test
```

- Visualize the generated frictional signals:
```bash
python FrictganNet.py --visualize
```

- Visualize the training processing:
```bash
cd logs
tensorboard --logdir=./
```

- Data: Training and testing data can be downloaded [here](https://drive.google.com/drive/folders/1ZA7aDgw1AYa85aXPJWPKKvTodIKZU97B?usp=sharing). After extracting the compressed file, put all the folders (from the downloaded file) in the project directory './dataset' (the same directory where the main file locates in).

- Original dataset from [HapTex](http://haptic.buaa.edu.cn/English_FabricDatabase.htm) developed by BUAA Human-machine Interaction lab, thanks for sharing the physical fabirc samples for our experiment.  

## Acknowledgement
This code is based on the implementation of sketch2normal worked by Wanchao Su from [sketch2normal](https://github.com/Ansire/sketch2normal). Thanks for the great work!
