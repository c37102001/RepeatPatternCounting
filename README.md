# Repeat Pattern Counting

## Preparation
(Images are available from https://ppt.cc/ffLrKx)

* put color imgs(det_imgs) in `input/image`
* put structure forest edge imgs(det_edges) in `input/edge_image`
* put HED edge imgs(hed_edges) in `input/hed_edge_image`

## How to run
for old version
1. `cd legacy_code`
2. run `python find_repeat_pattern.py`

for new version
1. `cd src`
2. run `python run.py`


## Result
Test on [structure, edge, combine] x [inner, outer, all], and for now [inner, all] is better.
![img](./src/IMG_39.jpg)
