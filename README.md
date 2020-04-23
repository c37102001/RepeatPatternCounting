# Repeat Pattern Counting


## Preparation

***From 0310 Updates, no need to download edge images anymore. Just put your images under `data/your_folder_name/image/` and modify `data_name = your_folder_name` in `config.ini`, then run `python main.py --test_all`, all generated edge images will be stored under `data/your_folder_name/edge/` and the results under `output/your_folder_name/`. See sample directory structure below.***

Images are available from https://ppt.cc/ffLrKx

HED, RCF, SF pretrained models are available from https://ppt.cc/fzN82x
* download `HED/network-bsds500.pytorch` model and put it in `HED/pretrained/`
* download `RCF/RCFcheckpoint_epoch12.pth` model and put it in `RCF/pretrained/`
* download `RCF/vgg16convs.mat` model and put it in `RCF/pretrained/`
* download structure forest model from `https://ppt.cc/fk7Xsx` and put it in `SF/`


## How to run
For old version
* `cd legacy_code` and run `python find_repeat_pattern.py`

For new version
* `cd src`
* if test one to several pictures:
    1. modify `img_list` in config.ini like `img_list = IMG_ (39).jpg, IMG_ (10).jpg, IMG_ (16).jpg`
    2. run `python main.py`
* if test all images in a folder:
    1. run `python main.py --test_all`
* add `--draw` to draw all process images
* add `--mark` to plot contour numbers in images


## Update
* 0215-1
  * ***Change obvious color gradient voting decision(in main.py part 4) according to thesis***
  * Fix remove_overlap(contours, 'inner') bug
  * Change obviousity factor name from 'shape' to 'solidity'
  * Add obviousity thresholds into config.ini
  * Remove --test --img arg flags
  * Add title in image

* 0216-1
  * ***Change remove overlap(outer contour) method(in get_group_cnts) and delete keep_list***
  * ***Remove keep_overlap list in config.ini and modify remove_overlap function. now only allow keep inner***
  * Rename images to re-order outputs.

* 0216-2
  * ***Add edge on border (add method in utils)***

* 0216-3
  * ***Move edge usage into config.ini use_edge and remove combine option***
  * ***Fix check_overlap label_weight sorting bug***
    * Note: setting `if more_weight >= label_list.count(label_group[-1]):...` means when weights are equal, choose later edge map in config.ini, so the use_edge order do affect result sometimes. The better edge image should put later.

* 0217-1
  * ***Add sliding_window.py***
  * ***Add saliency in input image dir and config.ini***
  * Change solidity threshold in contour filtering to average solidity.

* 0217-2
  * ***Replace solidity filter to area_over_perimeter in `filter_contours`***

* 0219-1
  * ***Change filter to approxPolyDP, convexHull***
  * Change area featrue extraction method from arcLength to contourArea

* 0224-1
  * Add check overlap condition: any point inside the other contour
  * Area obvious change to 0.25*largest area
  * Add morphological closing

* 0224-2
  * ***Change pipeline to combine contours first, then remove extract features and group.***
  * ***Add clustering threshold***

* 0225-1
  * ***Remove img_avg_color_gradient from obvisity filter***
  * ***Add "empirical" clustering threshold***
  * Add remove size outliers

* 0226-1
  * ***Add HED and RCF edges***
  * ***Set cluster threshold to absolute value***
  * Cancel size outlier removal

* 0302-1
  * ***New pipeline: combine contours >> group >> rm overlap***
  * ***Add Sobel edge detection***
  * New method for extracting shape feature
  * Lower l channel color gradient by 0.5

* 0302-2
  * ***New color gradient algorithm***
  * New remove overlap method

* 0303-1
  * ***Optimize overlap algorithm***
  * Fix color gradient bug
  * Visualize color gradient

* 0303-2
  * ***Use color gradient to filter noisy contour***
  * ***Remove 'area' from obvious voting***

* 0304-1
  * ***Add second clustering***
  * ***Set `color_gradient` as remove overlap criteria when different labels***
  * Filter group with too few contours at the end
  * Add function `clear_output_dir = True` and `img_list = 1-5` in config.ini

* 0304-2
  * ***Do `filter small group` before obviousity voting***
  
* 0309
  * ***Allow producing SF, HED and RCF edge image in procedure***

* 0310
  * Modify directory structure and config to more flexible testing

* 0317
  * Use solidity for filtering, area and color grad for obviousity voting

* 0319
  * Add texture feature, but performance drops when using it. Better not to use.

* 0423
  * Evaluation
  * Remove "Outer contour filter" Algorithm


## Sample directory structure 
Asterisk(*) means auto-generated directory
```
PatternCounting
├── data
│   ├── colony
│   │   ├── edge*
│   │   │   └── SF*
│   │   └── image
│   ├── general
│   │   ├── edge*
│   │   │   ├── HED*
│   │   │   ├── RCF*
│   │   │   └── SF*
│   │   └── image
│   └── pigs
│       ├── edge*
│       │   ├── HED*
│       │   ├── RCF*
│       │   └── SF*
│       └── image
├── output*
│   ├── colony*
│   ├── general*
│   └── pigs*
└── src
```

```
src
├── HED
│   ├── model.py
│   ├── pretrained
│   │   └── network-bsds500.pytorch
│   └── run_hed.py
├── RCF
│   ├── models.py
│   ├── pretrained
│   │   ├── RCFcheckpoint_epoch12.pth
│   │   └── vgg16convs.mat
│   └── run_rcf.py
├── SF
│   ├── model.yml.gz
│   └── run_sf.py
├── get_contours.py
├── get_features.py
...
```