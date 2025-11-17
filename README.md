# HPDS_EAA_INTERHAND

This repository contains a pytorch implementation of "__Efficient Visual Attention-Based Lightweight 3D Hand Reconstruction.__" 

$Minxuan\; Hu^{1*}, Wenji\; Yang^{2*}$

$^{1}\;School\;of\;Computer\;and\;Information\;Engineering,\;Jiangxi\;Agricultural\;University,\;Jiangxi\;330045,\;China$

$^{2}\;School\;of\;Software,\;Jiangxi\;Agricultural\;University,\;Jiangxi\;330045,\;China$



## Requirements

- Tested with python3.9 on Ubuntu 20.04.6 LTS, CUDA 12.2.

### packages

- pytorch (tested on 2.1.0)

- torchvision (tested on 0.16.0)

- pytorch3d (tested on 0.7.5)

- numpy

- OpenCV

- tqdm

- yacs >= 0.1.8

- gradio

### Pre-trained model and data

- Register and download [MANO](https://mano.is.tue.mpg.de/)  data. Put `MANO_LEFT.pkl` and `MANO_RIGHT.pkl` in `misc/mano`

After collecting the above necessary files, the directory structure of `./misc` is expected as follows:

```
./misc
├── mano
│   └── MANO_LEFT.pkl
│   └── MANO_RIGHT.pkl
├── model
│   └── config.yaml
│   └── interhand.pth
│   └── wild_demo.pth
├── graph_left.pkl
├── graph_right.pkl
├── mesh_left.pkl
├── mesh_right.pkl
├── upsample.pkl
├── v_color.pkl

```

## DEMO

1. Real-time demo :

```
python apps/demo.py --live_demo
```
2. Single-image reconstruction  :

```
python apps/demo.py --img_path demo/ --save_path demo/
```
Results will be stored in folder `./demo`

3. Web application  :
```
python apps/gradio_demo.py 
```
![p](demo/image.png)

**Noted**: We don't operate hand detection, so hands are expected to be roughly at the center of image and take approximately 70-90% of the image area.

## Training

1. Download [InterHand2.6M](https://mks0601.github.io/InterHand2.6M/) dataset and unzip it. (**Noted**: we used the `v1.0_5fps` version and `H+M` subset for training and evaluating)

2. Process the dataset by :
```
python dataset/interhand.py --data_path PATH_OF_INTERHAND2.6M --save_path ./interhand2.6m/
```
Replace `PATH_OF_INTERHAND2.6M` with your own store path of [InterHand2.6M](https://mks0601.github.io/InterHand2.6M/) dataset. 

3. Try the training code:
```
python apps/train.py utils/defaults.yaml
```

The output model and TensorBoard log file would be store in `./output`.
If you have multiple GPUs on your device, set `--gpu` to use them.  For example, use:

```
python apps/train.py utils/defaults.yaml --gpu 0,1,2,3
```
to train model on 4 GPUs.

4. We highly recommend you to try different loss weight and fine-turn the model with lower learning rate to get better result. The training configuration can be modified in `utils/defaults.yaml`.

## Evaluation

1. Download [InterHand2.6M](https://mks0601.github.io/InterHand2.6M/) dataset and unzip it. (**Noted**: we used the `v1.0_5fps` version and `H+M` subset for training and evaluating)

2. Process the dataset by :
```
python dataset/interhand.py --data_path PATH_OF_INTERHAND2.6M --save_path ./interhand2.6m/
```
Replace `PATH_OF_INTERHAND2.6M` with your own store path of [InterHand2.6M](https://mks0601.github.io/InterHand2.6M/) dataset. 

3. Run evaluation:
```
python apps/eval_interhand.py --data_path ./interhand2.6m/
```

You would get following output :

```
joint mean error:
    left: 9.097068570554256 mm
    right: 8.561082184314728 mm
    all: 8.829075377434492 mm
vert mean error:
    left: 9.311042726039886 mm
    right: 8.786577731370926 mm
    all: 9.048810228705406 mm
```


## Acknowledgement

The pytorch implementation of MANO is based on [manopth](https://github.com/hassony2/manopth). The GCN network is based on [hand-graph-cnn](https://github.com/3d-hand-shape/hand-graph-cnn). The heatmap generation and inference is based on [DarkPose](https://github.com/ilovepose/DarkPose). We thank the authors for their great job!


