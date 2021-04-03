# MAL_deepsort
This repo consists of the detector [Multiple Anchor Learning(MAL)](https://github.com/DeLightCMU/MAL-inference/) detector and tracker [deepsort](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch/tree/a60dcf9300ee7673b7d484e8ebb363b9a6408b81/deep_sort_pytorch) and tests on [VisDrone](https://github.com/VisDrone/VisDrone-Dataset).

## 1. Installation

### Requirements:
Please refer to [Multiple Anchor Learning(MAL)](https://github.com/DeLightCMU/MAL-inference/) and [deepsort](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch/tree/a60dcf9300ee7673b7d484e8ebb363b9a6408b81/deep_sort_pytorch)

Download MAL parameters and deepsort parameters:
```bash
#The path of the MAL parameters (backbone¡ª¡ªResNet-50-FPN) trained on COCO:
./models/resnet50/mal_r-50-fpn_cocomodel_0090000.pth
#The path of the deepsort parameters:
./deep_sort/deep/checkpoint/ckpt.t7
```
## 2. Any two videos from VisDrone (Task 4: Multi-Object Tracking) 
We test on valset([BaiduYun](https://pan.baidu.com/s/1_gLvMxkMKb3RZjGyZv7btQ)|[GoogleDrive](https://drive.google.com/file/d/1rqnKe9IgU_crMaxRoel9_nuUsMEBBVQu/view?usp=sharing)) of VisDrone2019-MOT dataset. Two videos we track are:
```bash
#The path of the first video:
./VisDrone2019-MOT-val/sequences/uav0000137_00458_v
#The annotation of the first video:
./VisDrone2019-MOT-val/annotations/uav0000137_00458_v.txt
#The path of the second video:
./VisDrone2019-MOT-val/sequences/uav0000305_00000_v
#The annotation of the first video:
./VisDrone2019-MOT-val/annotations/uav0000305_00000_v.txt
```
If you want to try other videos, please replace the annotation path in L152 of ./retinanet/infer.py to your own ones. You can also change the output path by replacing L238 of  ./retinanet/infer.py to your own one.


## 3. Running (Pytorch)
```bash
CUDA_VISIBLE_DEVICES=0 retinanet infer --images "path to your video data"  --batch=1

```
## 4. Video output from image sequence
```bash
#Please replace L7 to your output image sequence and replace L18 to your output images' name.
python video.py

```
