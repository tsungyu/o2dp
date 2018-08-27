# Second-order Democratic Aggregation 

Created by  [Tsung-Yu Lin](https://people.cs.umass.edu/~tsungyulin/),  [Subhransu Maji](http://people.cs.umass.edu/~smaji/) and [Piotr Koniusz](http://users.cecs.anu.edu.au/~koniusz/).

## Introduction

This repository contains the code for reproducing the results in our ECCV 2018 paper:

```
@inproceedings{lin2018o2dp,
    Author = {Tsung-Yu Lin and Subhransu Maji and Piotr Koniusz},
    Title = {Second-order Democratic Aggregation},
    Booktitle = {European Conference on Computer Vision (ECCV)},
    Year = {2018}
}
```

The paper analyzes various feature aggregators in the context of second-order features and proposes &gamma;-democratic pooling which generalizes sum pooling and democratic aggregation. See the [project page](http://vis-www.cs.umass.edu/o2dp/) and the [paper](https://arxiv.org/abs/1808.07503) for the detail. The code is tested on Ubuntu 14.04 using NVIDIA Titan X GPU and MATLAB R2016a. 

## Prerequisite 

1. [MatConvNet](http://www.vlfeat.org/matconvnet): Our code was developed on the MatConvNet version `1.0-beta24`.
2. [VLFEAT](http://www.vlfeat.org/) 
3. [bcnn-package](https://bitbucket.org/tsungyu/bcnn-package): The package includes our implementation of customized layers.

The packages are set up as the git submodules. Check them out by the following commands and follow the instructions on [MatConvNet](http://www.vlfeat.org/matconvnet) and [VLFEAT](http://www.vlfeat.org/)  project pages to install them.

```
>> git submodule init
>> git submodule update
```



## Datasets

To run the experiments, download the following datasets and edit the `model_setup.m` file to point them to the dataset locations. For instance, you can point to the birds dataset directory by setting `opts.cubDir = 'data/cub'`.

#### Fine-grained classification datasets:

* [Caltech-UCSD Birds](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)
* [FGVC Aircrafts](http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/)
* [Stanford Cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)

####Texture and indoor scene datasets:

* [Describable Textures Dataset (DTD)](https://www.robots.ox.ac.uk/~vgg/data/dtd/)
* [Flickr Material Database (FMD)](https://people.csail.mit.edu/celiu/CVPR2010/FMD/)
* [MIT Indoor](http://web.mit.edu/torralba/www/indoor.html)

### Pre-trained models

* ImageNet LSVRC 2012 pre-trained models: The [`vgg-verydeep-16`](http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-16.mat) and [`reset-101`](http://www.vlfeat.org/matconvnet/models/imagenet-resnet-101-dag.mat) ImageNet pre-trained models are used as our basic models. Download them from MatConvNet pre-trained models page.
* B-CNN fine-tuned models: We also provide the B-CNN fine-tuned models with `vgg-verydeep-16` from which we can extract the CNN features and aggregate them to construct the image descriptor. Download the models for [CUB Birds](http://maxwell.cs.umass.edu/bcnn/models2/bcnn-cub-dd-net.mat), [FGVC Aircrafts](http://maxwell.cs.umass.edu/bcnn/models2/bcnn-aircrafts-dd.mat), or [Stanford Cars](http://maxwell.cs.umass.edu/bcnn/models2/bcnn-cars-dd.mat) to reproduce the accuracy provided in the paper.

### Testing the models:

Solving the coefficients for &gamma;-democratic aggregation involves sinkhorn iteration. The hyperparameters  for the sinkhorn iteration are configurable in the entry codes `run_experiments_o2dp.m` and `run_experiments_sketcho2dp_resnet.m`. See the comment in the code for the detail.

* Second-order &gamma;-democratic aggregation: Point the variable `model_path` to the location of the model in `run_experiments_o2dp.m` and run the command `run_experiments_o2dp(dataset, gamma, gpuidx)`  in matlab terminal.

  * For example:

  ```matlab
  % gamma is the hyper-parameter gamma for &gamma;-democratic aggregation
  % gpuidx is the index of gpu on which you run the experiment
  run_experiments_o2dp('mit_indoor', 0.3, 1) 
  ```
  * Classification results: Sum and democratic aggregation can be achieved by setting the proper values of &gamma;. The optimal &gamma; values are indicated in the parenthesis. In general &gamma;=0.5 performs reasonably well. For `DTD` and `FMD` these numbers are reported on the first split. For the fine-grained recognition datasets (&#8224;) the results are obtained by using the fine-tuned B-CNN models while for the texture and indoor scene datasets the ImageNet pre-trained `vgg-verydeep-16` model is used.

    <table align="center">
        <tr>
          <th><i>Dataset</i></th>
          <th align=center>Sum(&gamma;=1)</th>
          <th align=center>&nbsp;&nbsp; Democratic(&gamma;=0)</th>
          <th align=center>&nbsp;&nbsp; &gamma;-democratic</th>
        </tr>
        <tr>
          <td>Caltech UCSD Birds &#8224;</td>
          <td align=center>84.0</td>
          <td align=center>84.7</td>
          <td align=center>84.9 (0.5)</td>
        </tr>
        <tr>
          <td>Stanford Cars &#8224;</td>
          <td align=center>90.6</td>
          <td align=center>89.7</td>
          <td align=center>90.8 (0.5)</td>
        </tr>
        <tr>
          <td>FGVC Aircrafts &#8224;</td>
          <td align=center>85.7</td>
          <td align=center>86.7</td>
          <td align=center>86.7 (0.0)</td>
        </tr>
        <tr>
          <td>DTD</td>
          <td align=center>71.2</td>
          <td align=center>72.2</td>
          <td align=center>72.3 (0.3)</td>
        </tr>
        <tr>
          <td>FMD</td>
          <td align=center>84.6</td>
          <td align=center>82.8</td>
          <td align=center>84.8 (0.8)</td>
        </tr>
        <tr>
          <td>MIT Indoor</td>
          <td align=center>79.5</td>
          <td align=center>79.6</td>
          <td align=center>80.4 (0.3)</td>
        </tr>
        </table>

* Second-order &gamma;-democratic aggregation in sketch space: Point the variable `model_path` to the location of the model in `run_experiments_sketcho2dp_resnet.m` and run the command `run_experiments_sketcho2dp_resnet(dataset, gamma, d, gpuidx)`  in matlab terminal. 

  * For example:

  ```matlab
  % gamma is the hyper-parameter gamma for &gamma;-democratic aggregation
  % d is the dimension for the sketch space
  % gpuidx is the index of gpu on which you run the experiment
  run_experiments_sketcho2dp_resnet('mit_indoor', 0.5, 8192, 1) 
  ```

  * The script aggregates the second-order ResNet features pre-trained on ImageNet in a 8192-dimensional sketch space with &gamma;-democratic aggregator. With ResNet features the model achieves the following results. For `DTD` and `FMD` the accuracy is averaged over 10 splits.

    <table align="center">
        <tr>
          <th></th>
          <th align=center>DTD</th>
          <th align=center>&nbsp;&nbsp; FMD</th>
          <th align=center>&nbsp;&nbsp; MIT Indoor</th>
        </tr>
        <tr>
          <td>Accuracy</td>
          <td align=center>76.2 &#8723; 0.7</td>
          <td align=center>84.3 &#8723; 1.5</td>
          <td align=center>84.3</td>
        </tr>
    </table>

