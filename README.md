# Repeat Pattern Counting

## Preparation
(Images are available from https://ppt.cc/ffLrKx)

* put color imgs(det_imgs) in `input/image`
* put structure forest edge imgs(det_edges) in `input/edge_image`
* put HED edge imgs(hed_edges) in `input/hed_edge_image`
* modify configs in `config.ini`

## How to run
for old version
* `cd legacy_code` and run `python find_repeat_pattern.py`

for new version
* `cd src`
* if test one to several pictures:
    1. modify `img_list` in config.ini like `img_list = IMG_ (39).jpg, IMG_ (10).jpg, IMG_ (16).jpg`
    2. run `python main.py`
* if test all images in a folder:
    1. modify `input_dir`, `strct_edge_dir` and `hed_edge_dir` in config.ini
    2. run `python main.py --test_all`
* add `--draw` if want to draw all process pictures 
* add `--mark` if want to see every contour number in plots

## Result
* marked contour
![img](./img/IMG_39_marked.jpg)
* some results
![img](./img/IMG_39_result.jpg)

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