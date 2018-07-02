# YOLOv2 Object Detection and Tracking w/ Keras 

This repository builds on top of [this](https://github.com/miranthajayatilake/YOLOw-Keras) repository. 
--------------------------------------------------------------------------------

## Quick Start

- Clone this repository to your PC
- Download any Darknet model cfg and weights from the [official YOLO website](http://pjreddie.com/darknet/yolo/). 
- Convert the dowloaded cfg and weights files into a h5 file using YAD2K library. (This is explained step by step below in the more details section)
- Copy the generated h5 file to the model_data folder and edit the name of the pretrained model in yolo.py code to the name of your h5 file.
- Place the input image you want to try object detection in the images folder and copy its file name.
- Assign your input image file name to input_image_name variable in yolo.py.
- Open terminal from the repository directory directly and run the yolo.py file
	
	`python yolo.py`

--------------------------------------------------------------------------------

## More Details

How to convert cfg and weights files to h5 using YAD2k library (Windows)

- Clone the [YAD2K Library](https://github.com/allanzelener/YAD2K) to your PC
- Open terminal from the cloned directory
- Copy and paste the downloaded weights and cfg files to the YAD2K master directory
- Run `python yad2k.py yolo.cfg yolo.weights model_data/yolo.h5` on the terminal and the h5 file will be generated.
- Move the generated h5 file to model_data folder of the simpleYOLOwKeras directory



-------------------------------------------------------------------------------
