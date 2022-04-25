# Text-to-Image-Synthesis 

## Intoduction

This is a pytorch implementation of [Generative Adversarial Text-to-Image Synthesis paper](https://arxiv.org/abs/1605.05396), we train a conditional generative adversarial network, conditioned on text descriptions, to generate images that correspond to the description. The network architecture is shown below (Image from [1]). This architecture is based on DCGAN.

<img width="1143" alt="image" src="https://user-images.githubusercontent.com/16959405/165022019-c1a75dba-4031-45c5-80c2-c0fbd4ba8a43.png">


## Requirements

- pytorch 
- visdom
- h5py
- PIL
- numpy

This implementation currently only support running with GPUs.

## Implementation details

This implementation follows the Generative Adversarial Text-to-Image Synthesis paper [1], however it works more on training stablization and preventing mode collapses by implementing:
- Feature matching [2]
- One sided label smoothing [2]
- minibatch discrimination [2] (implemented but not used)
- WGAN [3]
- WGAN-GP [4] (implemented but not used)
- SRCNN
- GAN-BERT (implemented but not integrated)

## Datasets

We used [Flowers](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/) datasets, we converted each dataset (images, text embeddings) to hd5 format. 

We used the [text embeddings](https://github.com/reedscot/icml2016) provided by the paper authors

**To use this code you can either:**

- Use the converted hd5 datasets, [flowers](https://drive.google.com/open?id=1EgnaTrlHGaqK5CCgHKLclZMT_AMSTyh8)
- Convert the data youself
  1. download the dataset as described [here](https://github.com/reedscot/cvpr2016)
  2. Add the paths to the dataset to `config.yaml` file.
  3. Use [convert_flowers_to_hd5_script](convert_flowers_to_hd5_script.py) script to convert the dataset.
  
**Hd5 file taxonomy**
`
 - split (train | valid | test )
    - example_name
      - 'name'
      - 'img'
      - 'embeddings'
      - 'class'
      - 'txt'
      
## Usage
### Training

```
python runtime.py

```

**Arguments:**
- `type` : GAN archiecture to use `(gan | wgan | vanilla_gan | vanilla_wgan)`. default = `gan`. Vanilla mean not conditional
- `dataset`: Dataset to use `(birds | flowers)`. default = `flowers`
- `split` : An integer indicating which split to use `(0 : train | 1: valid | 2: test)`. default = `0`
- `lr` : The learning rate. default = `0.0002`
- `diter` :  Only for WGAN, number of iteration for discriminator for each iteration of the generator. default = `5`
- `vis_screen` : The visdom env name for visualization. default = `gan`
- `save_path` : Path for saving the models.
- `l1_coef` : L1 loss coefficient in the generator loss fucntion for gan and vanilla_gan. default=`50`
- `l2_coef` : Feature matching coefficient in the generator loss fucntion for gan and vanilla_gan. default=`100`
- `pre_trained_disc` : Discriminator pre-tranined model path used for intializing training.
- `pre_trained_gen` Generator pre-tranined model path used for intializing training.
- `batch_size`: Batch size. default= `64`
- `num_workers`: Number of dataloader workers used for fetching data. default = `8`
- `epochs` : Number of training epochs. default=`200`
- `cls`: Boolean flag to whether train with cls algorithms or not. default=`False`

## Image Super Resolution

We have used SRCNN (Super-Resolution Convolutional Neural Network) and Bicubic  interpolation inorder to enhance the generated image.

Command:

```
cd SRCNN
```
```
cd SRCNN-pytorch
```

```
python test.py --weights-file "BLAH_BLAH/srcnn_x3.pth" --image-file "data/a flower that has violet petals that are surrounding a cluster of stamen_.jpg” —scale 4

```

```
python test.py --weights-file "BLAH_BLAH/srcnn_x3.pth" --image-file "data/this flower is white and trumpet shaped with yellow-green lines running from the center of the flowe.jpg" --scale 4

```

## GAN-Bert

We have added GAN-BERT (implement, but not integrated) 
```
GANBERT_pytorch.ipynb
```

## Result

We have received an inception score of 3.73 

<img width="903" alt="image" src="https://user-images.githubusercontent.com/16959405/165020421-5c7bdb00-42be-4dfd-b444-24a0461a7105.png">

![image](https://user-images.githubusercontent.com/16959405/165020146-50f44691-0240-44c0-a0a3-9fd299d11cac.png)

![image](https://user-images.githubusercontent.com/16959405/165020085-1090e101-0e26-4455-bfb1-ed401af9fa68.png)

<img width="667" alt="image" src="https://user-images.githubusercontent.com/16959405/165020519-a4a25d9e-a5d9-4a10-b825-e463677bff3a.png">

![plot_epoch_w_cores (1) (1)](https://user-images.githubusercontent.com/16959405/165020715-a0951c31-5c47-4175-94bb-4eb44fa32460.png)



## References
[1]  Generative Adversarial Text-to-Image Synthesis https://arxiv.org/abs/1605.05396 

[2]  Improved Techniques for Training GANs https://arxiv.org/abs/1606.03498

[3]  Wasserstein GAN https://arxiv.org/abs/1701.07875

[4] Improved Training of Wasserstein GANs https://arxiv.org/pdf/1704.00028.pdf

[5] Image Super-Resolution Using DeepConvolutional Networks https://arxiv.org/pdf/1501.00092.pdf

[6] https://github.com/crux82/ganbert-pytorch

