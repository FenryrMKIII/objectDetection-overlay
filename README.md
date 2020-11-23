# About
odsynthds stands for Object Detection Synethtic Dataset.

It provides the user an easy way to produce a synthetic dataset for object detection tasks from a set of background images and a set of objects to be detected. The package handles image augmentation techniques. 

# How to install


# How to use
launch in a cli :
```
python -m main.py -t overlay -o raw_symbols -b backgrounds --no_alpha --save_bboxes --resize 640 -f SSD -n 200 --save trainingset
```
