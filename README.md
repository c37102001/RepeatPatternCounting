# Repeat Pattern Counting


## Preparation
Images are available from https://ppt.cc/ffLrKx
* put color imgs(det_imgs) in `input/image/`
* put structure forest edge imgs(det_edges) in `input/edge_image/`
* put HED edge imgs(hed_edges) in `input/hed_edge_image/`
* put RCF edge imgs(rcf_edges) in `input/rcf_edge_image/`
* put saliency imgs (saliency_maps) in `input/saliency_image/`
* modify configs in `config.ini`

HED, RCF pretrained models are available from https://ppt.cc/fzN82x
* download `HED/network-bsds500.pytorch` model and put it in `HED/pretrained/`
* download `RCF/RCFcheckpoint_epoch12.pth` model and put it in `RCF/pretrained/`
* download `RCF/vgg16convs.mat` model and put it in `RCF/pretrained/`


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

To obtain HED (and RCF) edge images:
* cd `HED`
* modify IMG_DIR, OUTPUT_DIR in `run_hed.py` (and test.lst or test_all.lst for RCF)
* run `python run_hed.py`


## Update
* 0215-1
  * [Important] change obvious color gradient voting decision(in main.py part 4) according to thesis
  * fix remove_overlap(contours, 'inner') bug
  * change obviousity factor name from 'shape' to 'solidity'
  * add obviousity thresholds into config.ini
  * remove --test --img arg flags
  * add title in image

* 0216-1
  * [Important] change remove overlap(outer contour) method(in get_group_cnts) and delete keep_list
  * [Important] remove keep_overlap list in config.ini and modify remove_overlap function. now only allow keep inner
  * rename images to re-order outputs.

* 0216-2
  * [Important] add edge on border (add method in utils)

* 0216-3
  * [Important] move edge usage into config.ini use_edge and remove combine option
  * [Important] fix check_overlap label_weight sorting bug
    * Note: setting `if more_weight >= label_list.count(label_group[-1]):...` means when weights are equal, choose later edge map in config.ini, so the use_edge order do affect result sometimes. The better edge image should put later.

* 0217-1
  * [Important] add sliding_window.py
  * [Important] add saliency in input image dir and config.ini
  * change solidity threshold in contour filtering to average solidity.

* 0217-2
  * [Important] replace solidity filter to area_over_perimeter in `filter_contours`

* 0219-1
  * [Important] change filter to approxPolyDP, convexHull
  * change area featrue extraction method from arcLength to contourArea

* 0224-1
  * add check overlap condition: any point inside the other contour
  * area obvious change to 0.25*largest area
  * add morphological closing

* 0224-2
  * [Important] change pipeline to combine contours first, then remove extract features and group.
  * [Important] add clustering threshold

* 0225-1
  * [Important] remove img_avg_color_gradient from obvisity filter
  * [Important] add "empirical" clustering threshold
  * add remove size outliers

* 0226-1
  * [Important] add HED and RCF edges
  * [Important] set cluster threshold to absolute value
  * cancel size outlier removal

* 0302-1
  * [Important] New pipeline: combine contours >> group >> rm overlap
  * [Important] Add Sobel edge detection
  * New method for extracting shape feature
  * Lower l channel color gradient by 0.5

* 0302-2
  * [Important] New color gradient algorithm
  * new remove overlap method

* 0303-1
  * [Important] Optimize overlap algorithm
  * Fix color gradient bug
  * Visualize color gradient

* 0303-2
  * [Important] Use color gradient to filter noisy contour
  * del 'area' from obvious voting. 