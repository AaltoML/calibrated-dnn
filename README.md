# Fixing Overconfidence in Dynamic Neural Networks

This repository provides instructions and code on how to replicate the results predented in the paper *Fixing Overconfidence in Dynamic Neural Networks*.


## Installing required Python packages in a virtual environment:

* Start by installing Python version 3.7.4
* Create and start a virtual environment named `MSDNet`:
```setup
python -m venv MSDNet
source MSDNet/bin/activate
```
* Install the required packages into the newly created virtual environment:
```setup
python -m pip install -r requirements.txt
```


## Obtaining the CIFAR-100, ImageNet, and Caltech-256 data sets


### CIFAR-100

* The CIFAR-100 and Caltech-256 data sets will be automatically downloaded by the training script if you don't already have them (also CIFAR-10 if you wish to experiment on that). Note that on Caltech-256 the dowloaded image folders may contain some additional non-image files that need to be manually removed from the folders for the training scripts to run.

### ImageNet

* The ImageNet data set can be downloaded at image-net.org. You should download the `Training images (Task 1 & 2)`(138GB) and `Validation images (all tasks)`(6.3GB) from the ILSVRC2012 version.
* After this you should have `ILSVRC2012_img_train.tar` and `ILSVRC2012_img_val.tar`. Extract `ILSVRC2012_img_train.tar` once to obtain a folder `ILSVRC2012_img_train` containing 1000 .tar subfolders.
* Make sure `ILSVRC2012_img_train` and `ILSVRC2012_img_val.tar` are in the same folder, we refer to this as `/path_to_imagenet`. Go to this folder to perform the following data set extraction.
* The complete training set can be extracted from the .tar files for example by doing the following:
```setup
mkdir train && cd train
find /path_to_imagenet/ILSVRC2012_img_train -name "*.tar" | while read NAME ; do SUBSTRING=$(echo $NAME| cut -d'/' -f 7) ; mkdir -p "${SUBSTRING%.tar}"; tar -xf "${NAME}" -C "${SUBSTRING%.tar}"; done
cd ..
```

* The validation set can be extracted as follows:
```setup
mkdir val && cd val && tar -xf /path_to_imagenet/ILSVRC2012_img_val.tar -C /path_to_imagenet/val
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
cd ..
```

* After these steps you should have folders `/path_to_imagenet/train` and `/path_to_imagenet/val` which both contain 1000 subfolders, each containing the sample images for one class.
                
## Training MSDNet backbone for CIFAR100

* To train the small model for CIFAR-100, run the following command:
```
python main.py --data-root /path_to_CIFAR100/ --data cifar100 \
	--save /savepath/MSDNet/cifar100_4  \
    --arch msdnet --batch-size 64 --epochs 300 --nBlocks 4 \
    --stepmode lin_grow --step 1 --base 1 --nChannels 16 --use-valid \
    -j 1 --var0 2.0 --laplace_temperature 1.0
```

* To run the medium and large models, you need to change the `--nBlocks` argument to 6 and 8 respectively. Note that you also need to change the path in `--save` to not overwrite previously trained models.
    
## Training MSDNet backbone for ImageNet

* To train the small model for ImageNet, run the following command:
```
python main.py --data-root /path_to_imagenet --data ImageNet \
	--save /savepath/MSDNet/imagenet_base4 \
    --arch msdnet --batch-size 256 --epochs 90 --nBlocks 5 \
    --stepmode even --step 4 --base 4 --nChannels 32 --use-valid \
    --growthRate 16 --grFactor 1-2-4-4 --bnFactor 1-2-4-4 \
    -j 4 --gpu 0 --var0 2.0 --laplace_temperature 1.0
```
* To run the medium and large models, you need to change both the `--step` and `--base` arguments to 6 for the medium model, and to 7 for the large model. Note that you also need to change the path in `--save` to not overwrite previously trained models.

## Training MSDNet backbone for Caltech-256

* To train the small model for Caltech-256, run the following command:
```
python main.py --data-root /path_to_caltech --data caltech256 \
	--save /savepath/MSDNet/caltech_base4 \
    --arch msdnet --batch-size 128 --epochs 180 --nBlocks 5 \
    --stepmode even --step 4 --base 4 --nChannels 32 --use-valid \
    --growthRate 16 --grFactor 1-2-4-4 --bnFactor 1-2-4-4 \
    -j 4 --gpu 0 --var0 2.0 --laplace_temperature 1.0
```
* To run the medium and large models, you need to change both the `--step` and `--base` arguments to 6 for the medium model, and to 7 for the large model. Note that you also need to change the path in `--save` to not overwrite previously trained models.
                
## Compute Laplace approximation separately after training

* The Laplace approximation is precomputed automatically at the end of training. However, if you wish to separately recompute the Laplace approximation, you can do so as follows:
```
python main.py --data-root /path_to_CIFAR100/ --data cifar100 \
	--save /savepath/MSDNet/cifar100_4 \
    --arch msdnet --batch-size 64 --epochs 300 --nBlocks 4 \
    --stepmode lin_grow --step 1 --base 1 --nChannels 16 --use-valid \
    -j 1 --compute_only_laplace --resume /savepath/MSDNet/cifar100_4/save_models/model_best_acc.pth.tar \
    --var0 2.0
```
* Note that you need to change the arguments `--save` and `--resume` to the correct path that contains the trained model that you want to calculate the Laplace approximation for.
* The example command is for CIFAR-100 but the same can be done for ImageNet or Caltech-256, by adding the `--compute_only_laplace` and `--resume /path_to_saved_model/save_models/model_best_acc.pth.tar` arguments to the ImageNet or Caltech-256 training command.

## Test vanilla MSDNet on CIFAR-100

* To test the small vanilla MSDNet model on CIFAR-100, run the following:
```
python main.py --data-root /path_to_CIFAR100/ --data cifar100 --save /savepath/MSDNet/cifar100_4 \
    --arch msdnet --batch-size 64 --epochs 300 --nBlocks 4 --stepmode lin_grow --step 1 --base 1 \
    --nChannels 16 --use-valid -j 1 --evalmode dynamic \
    --evaluate-from /savepath/MSDNet/cifar100_4/save_models/model_best_acc.pth.tar
```
* Note that the `--save` and `--evaluate-from` arguments have to be the correct paths to the saved model directory.
* For medium and large models you need to again change the `--nBlocks` argument accordingly, as well as the paths in `--save` and `--evaluate-from`.

## Test vanilla MSDNet on ImageNet

* To test the small vanilla MSDNet model on ImageNet, run the following:
```
python main.py --data-root /path_to_imagenet --data ImageNet --save /savepath/MSDNet/imagenet_base4 \
    --arch msdnet --batch-size 256 --epochs 90 --nBlocks 5 --stepmode even --step 4 --base 4 \
    --nChannels 32 --use-valid -j 4 --gpu 0 --evalmode dynamic \
    --growthRate 16 --grFactor 1-2-4-4 --bnFactor 1-2-4-4 \
    --evaluate-from /savepath/MSDNet/imagenet_base4/save_models/model_best_acc.pth.tar
```
* Here again the medium and large models require changing the `--step` and `--base` arguments as described in the model training, and the paths in `--save` and `--evaluate-from` need to be set correctly to utilize the correct saved model that you want to evaluate.

## Test vanilla MSDNet on Caltech-256

* To test the small vanilla MSDNet model on Caltech-256, run the following:
```
python main.py --data-root /path_to_caltech --data caltech256 --save /savepath/MSDNet/caltech_base4 \
    --arch msdnet --batch-size 128 --epochs 180 --nBlocks 5 --stepmode even --step 4 --base 4 \
    --nChannels 32 --use-valid -j 4 --gpu 0 --evalmode dynamic \
    --growthRate 16 --grFactor 1-2-4-4 --bnFactor 1-2-4-4 \
    --evaluate-from /savepath/MSDNet/caltech_base4/save_models/model_best_acc.pth.tar
```
* Here again the medium and large models require changing the `--step` and `--base` arguments as described in the model training, and the paths in `--save` and `--evaluate-from` need to be set correctly to utilize the correct saved model that you want to evaluate.
        
## Test models that use Laplace and/or model-internal ensembling (MIE)

* To use Laplace approximation in the evaluation of a model, add the following arguments to the model testing commands:
```
--laplace --laplace_temperature 1.0 --var0 2.0 --n_mc_samples 50 --optimize_temperature --optimize_var0
```
* To use MIE in the evaluation of a model, add the argument `--MIE` to the model testing commands.
* To test with both Laplace and MIE (our model) add both of the above mentioned arguments into the testing command.
