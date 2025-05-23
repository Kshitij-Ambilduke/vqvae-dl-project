# Implementation of Vector Quantized Variational Auto-encoder (VQ-VAE)

### Report: [[Link]](https://drive.google.com/file/d/19C-hUncvtCInUwkIhJ7ArrRQYpZTtlNl/view?usp=sharing)

### For training VQ-VAE:
```bash
python train_vqvae.py --output_dir vqvae_output_dir
```
other arguments:
```
--dataset           : "mnist" / "fashionmnist" / "cifar10"
--embedding_dim     : Dimension of codebook vectors
--num_embeddings    : Number of codebook vectors
--output_dir        : directory to save artifacts
```

### For training pixelCNN 
```bash
python pixelcnn\train_pixel_cnn.py --vqvae_path vqvae_output_dir\vqvae_model_3.pth --dataset mnist --output_dir pixel_cnn_output_dir 
```

Other arguments:
```
--dataset           :"mnist" / "fashionmnist" / "cifar10"
--num_embeddings    : Should be same as the value used for vqvae training
--embedding_dim     : Should be same as the value used for vqvae training
--in_channels       : 1 for "mnist"/"fashionmnist", 3 for "cifar10"
```

### Sample new images
```bash
python sample_new_images.py --pixelcnn_path pixel_cnn_output_dir\pixelcnn_30.pth --vqvae_path vqvae_output_dir\vqvae_model_3.pth --output_dir vqvae_output_dir
```

> change the `pixelcnn_path` and `vqvae_path` according to your training settings. 

Arguments:
```
--pixelcnn_path     : path of pixelcnn model. It will be saved in output_dir stated in train_pixel_cnn.py

--vqvae_path        : path of the vqvae model. It will be saved in output_dir stated in train_vqvae.py

--output_dir        : Output directory, the new images will be saved at "output_dir/newly_sample_images"
```
