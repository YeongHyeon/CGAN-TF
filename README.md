[TensorFlow] Conditional Generative Adversarial Nets (CGAN)
=====

TensorFlow implementation of Conditional Generative Adversarial Nets (CGAN) with MNIST dataset.  

## Architecture

### Training algorithm
<div align="center">
  <img src="./figures/algorithm.png" width="500">  
  <p>The algorithm for training CGAN [1].</p>
</div>

### GAN architecture
<div align="center">
  <img src="./figures/cgan.png" width="500">  
  <p>The architecture of CGAN [1].</p>
</div>

### Graph in TensorBoard
<div align="center">
  <img src="./figures/graph.png" width="650">  
  <p>Graph of CGAN.</p>
</div>

## Results

### Training Procedure
<div align="center">
  <p>
    <img src="./figures/CGAN_loss_d.svg" width="300">
    <img src="./figures/CGAN_loss_g.svg" width="300">
  </p>
  <p>Loss graph in the training procedure. </br> Each graph shows loss of the discriminator and loss of the generator respectively.</p>
</div>

### Test Procedure

## From random noise without conditions
<div align="center">

|z:2|z:64|z:128|
|:---:|:---:|:---:|
|<img src="./figures/z02_n.png" width="200">|<img src="./figures/z64_n.png" width="200">|<img src="./figures/z128_n.png" width="200">|

</div>

## From random noise with conditions
<div align="center">

|z:2|z:64|z:128|
|:---:|:---:|:---:|
|<img src="./figures/z02_c.png" width="200">|<img src="./figures/z64_c.png" width="200">|<img src="./figures/z128_c.png" width="200">|

</div>

## Latent space walking with conditions
<div align="center">

|Class-0 (z:2)|Class-1 (z:2)|Class-2 (z:2)|Class-3 (z:2)|Class-4 (z:2)|
|:---:|:---:|:---:|:---:|:---:|
|<img src="./figures/z02_0.png" width="150">|<img src="./figures/z02_1.png" width="150">|<img src="./figures/z02_2.png" width="150">|<img src="./figures/z02_3.png" width="150">|<img src="./figures/z02_4.png" width="150">|

|Class-5 (z:2)|Class-6 (z:2)|Class-7 (z:2)|Class-8 (z:2)|Class-9 (z:2)|
|:---:|:---:|:---:|:---:|:---:|
|<img src="./figures/z02_5.png" width="150">|<img src="./figures/z02_6.png" width="150">|<img src="./figures/z02_7.png" width="150">|<img src="./figures/z02_8.png" width="150">|<img src="./figures/z02_9.png" width="150">|

</div>

<div align="center">

|Class-0 (z:64)|Class-1 (z:64)|Class-2 (z:64)|Class-3 (z:64)|Class-4 (z:64)|
|:---:|:---:|:---:|:---:|:---:|
|<img src="./figures/z64_0.png" width="150">|<img src="./figures/z64_1.png" width="150">|<img src="./figures/z64_2.png" width="150">|<img src="./figures/z64_3.png" width="150">|<img src="./figures/z64_4.png" width="150">|

|Class-5 (z:64)|Class-6 (z:64)|Class-7 (z:64)|Class-8 (z:64)|Class-9 (z:64)|
|:---:|:---:|:---:|:---:|:---:|
|<img src="./figures/z64_5.png" width="150">|<img src="./figures/z64_6.png" width="150">|<img src="./figures/z64_7.png" width="150">|<img src="./figures/z64_8.png" width="150">|<img src="./figures/z64_9.png" width="150">|

</div>

<div align="center">

|Class-0 (z:128)|Class-1 (z:128)|Class-2 (z:128)|Class-3 (z:128)|Class-4 (z:128)|
|:---:|:---:|:---:|:---:|:---:|
|<img src="./figures/z128_0.png" width="150">|<img src="./figures/z128_1.png" width="150">|<img src="./figures/z128_2.png" width="150">|<img src="./figures/z128_3.png" width="150">|<img src="./figures/z128_4.png" width="150">|

|Class-5 (z:128)|Class-6 (z:128)|Class-7 (z:128)|Class-8 (z:128)|Class-9 (z:128)|
|:---:|:---:|:---:|:---:|:---:|
|<img src="./figures/z128_5.png" width="150">|<img src="./figures/z128_6.png" width="150">|<img src="./figures/z128_7.png" width="150">|<img src="./figures/z128_8.png" width="150">|<img src="./figures/z128_9.png" width="150">|

</div>

## Environment
* Python 3.7.4  
* Tensorflow 1.14.0  
* Numpy 1.17.1  
* Matplotlib 3.1.1  
* Scikit Learn (sklearn) 0.21.3  


## Reference
[1] Mehdi Mirza and Simon Osindero. (2014). <a href="https://arxiv.org/abs/1411.1784">Conditional Generative Adversarial Nets</a>. arXiv preprint arXiv:1411.1784.   
