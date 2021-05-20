# 说明

**此项目基于openpose，在openpose上添加安全帽于人体匹配策略，新增文件有base64_func.py input.json output.json server.py**

**base64_func.py用于图片解码编码，保持opencv读取和pillow读取接口，input.json outpu.json用于演示输入输出结果,server.json提供了一个http服务，可用于客户端调用接口**

# 描述

**该请求用于识别一张图片，即对于输入的一张图片（可正常解码，且长宽比较合适），输出人体边框与是否佩戴安全帽与否的标志信息。具体细节请看请求说明**

# 请求说明

## 请求示例

HTTP 方法：`POST`

请求URL： `http://0.0.0.0:24417/`

Header如下：

| 参数         | 值               |
| ------------ | ---------------- |
| Content-Type | application/json |

Body中放置请求参数，参数详情如下：

## 请求参数

| 参数名称     | 是否必选 | 类型   | 说明                                                         |
| ------------ | -------- | ------ | ------------------------------------------------------------ |
| img          | 必选     | string | 图像数据，base64编码，支持jpg/png/bmp格式。**注意：图片需要base64编码、去掉编码头** |
| hat_location | 必选     | list   | 包含安全帽的坐标信息，具体描述请看实例                       |
| +height      | 必选     | int    | 安全帽高低                                                   |
| +left        | 必选     | int    | 安全帽x轴坐标                                                |
| +top         | 必选     | int    | 安全帽y轴坐标                                                |
| +width       | 必选     | int    | 安全帽宽度                                                   |

## 请求代码示例

```json
{
    "hat_location": [
      {
     			"height": 16,
                "left": 398,
                "top": 110,
                "width": 22
      },
      {
      			"height": 16,
                "left": 155,
                "top": 102,
                "width": 23
      }
    ],
    "img":[
    			"img base64 code"
    ]
}
```



## 返回说明

### 返回参数

| 参数    | 类型   | 是否必须 | 说明                                                         |
| ------- | ------ | -------- | ------------------------------------------------------------ |
| code    | uint64 | 是       | 状态码                                                       |
| data    | list   | 是       | 人体坐标数据                                                 |
| +flag   | bool   | 是       | 是否佩戴安全帽标识符                                         |
| +rate   | float  | 是       | 人体在图片中的面积占比（在后期可以根据占比，去除占比太小的人体） |
| +x1     | uint64 | 是       | 左上角x坐标                                                  |
| +y1     | uint64 | 是       | 左上角y坐标                                                  |
| +x2     | uint64 | 是       | 右下角x坐标                                                  |
| +y2     | uint64 | 是       | 右下角y坐标                                                  |
| message | string | 是       | 提示完成检测                                                 |

### 返回值示例

```json
{
    "code": "200",
    "data": [
        {
            "flag": true,
            "rate": 6.324000000000001,
            "x1": 171,
            "x2": 249,
            "y1": 109,
            "y2": 244
        },
        {
            "flag": true,
            "rate": 4.625,
            "x1": 327,
            "x2": 382,
            "y1": 122,
            "y2": 262
        },
        {
            "flag": true,
            "rate": 3.108,
            "x1": 85,
            "x2": 130,
            "y1": 98,
            "y2": 213
        },
        {
            "flag": true,
            "rate": 5.58,
            "x1": 262,
            "x2": 319,
            "y1": 117,
            "y2": 280
        },
        {
            "flag": false,
            "rate": 1.934,
            "x1": 148,
            "x2": 176,
            "y1": 111,
            "y2": 226
        }
    ],
    "message": "Success"
}
```

**以下为原项目说明**

# Real-time 2D Multi-Person Pose Estimation on CPU: Lightweight OpenPose

This repository contains training code for the paper [Real-time 2D Multi-Person Pose Estimation on CPU: Lightweight OpenPose](https://arxiv.org/pdf/1811.12004.pdf). This work heavily optimizes the [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) approach to reach real-time inference on CPU with negliable accuracy drop. It detects a skeleton (which consists of keypoints and connections between them) to identify human poses for every person inside the image. The pose may contain up to 18 keypoints: ears, eyes, nose, neck, shoulders, elbows, wrists, hips, knees, and ankles. On COCO 2017 Keypoint Detection validation set this code achives 40% AP for the single scale inference (no flip or any post-processing done). The result can be reproduced using this repository. *This repo significantly overlaps with https://github.com/opencv/openvino_training_extensions, however contains just the necessary code for human pose estimation.*

<p align="center">
  <img src="data/preview.jpg" />
</p>

:fire: Check out our [new work](https://github.com/Daniil-Osokin/gccpm-look-into-person-cvpr19.pytorch) on accurate (and still fast) single-person pose estimation, which ranked 10<sup>th</sup> on CVPR'19 [Look-Into-Person](http://47.100.21.47:9999/index.php) challenge.

:fire::fire: Check out our lightweight [3D pose estimation](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation-3d-demo.pytorch), which is based on [Single-Shot Multi-Person 3D Pose Estimation From Monocular RGB](https://arxiv.org/pdf/1712.03453.pdf) paper and this work.

## Table of Contents

* [Requirements](#requirements)
* [Prerequisites](#prerequisites)
* [Training](#training)
* [Validation](#validation)
* [Pre-trained model](#pre-trained-model)
* [C++ demo](#cpp-demo)
* [Python demo](#python-demo)
* [Citation](#citation)

### Other Implementations

* TensorFlow by [murdockhou](https://github.com/murdockhou/lightweight_openpose).
* OpenVINO by [Pavel Druzhkov](https://github.com/openvinotoolkit/open_model_zoo/pull/1718/).

## Requirements

* Ubuntu 16.04
* Python 3.6
* PyTorch 0.4.1 (should also work with 1.0, but not tested)

## Prerequisites

1. Download COCO 2017 dataset: [http://cocodataset.org/#download](http://cocodataset.org/#download) (train, val, annotations) and unpack it to `<COCO_HOME>` folder.
2. Install requirements `pip install -r requirements.txt`

## Training

Training consists of 3 steps (given AP values for full validation dataset):
* Training from MobileNet weights. Expected AP after this step is ~38%.
* Training from weights, obtained from previous step. Expected AP after this step is ~39%.
* Training from weights, obtained from previous step and increased number of refinement stages to 3 in network. Expected AP after this step is ~40% (for the network with 1 refinement stage, two next are discarded).

1. Download pre-trained MobileNet v1 weights `mobilenet_sgd_68.848.pth.tar` from: [https://github.com/marvis/pytorch-mobilenet](https://github.com/marvis/pytorch-mobilenet) (sgd option). If this doesn't work, download from [GoogleDrive](https://drive.google.com/file/d/18Ya27IAhILvBHqV_tDp0QjDFvsNNy-hv/view?usp=sharing).

2. Convert train annotations in internal format. Run `python scripts/prepare_train_labels.py --labels <COCO_HOME>/annotations/person_keypoints_train2017.json`. It will produce `prepared_train_annotation.pkl` with converted in internal format annotations.

   [OPTIONAL] For fast validation it is recommended to make *subset* of validation dataset. Run `python scripts/make_val_subset.py --labels <COCO_HOME>/annotations/person_keypoints_val2017.json`. It will produce `val_subset.json` with annotations just for 250 random images (out of 5000).

3. To train from MobileNet weights, run `python train.py --train-images-folder <COCO_HOME>/train2017/ --prepared-train-labels prepared_train_annotation.pkl --val-labels val_subset.json --val-images-folder <COCO_HOME>/val2017/ --checkpoint-path <path_to>/mobilenet_sgd_68.848.pth.tar --from-mobilenet`

4. Next, to train from checkpoint from previous step, run `python train.py --train-images-folder <COCO_HOME>/train2017/ --prepared-train-labels prepared_train_annotation.pkl --val-labels val_subset.json --val-images-folder <COCO_HOME>/val2017/ --checkpoint-path <path_to>/checkpoint_iter_420000.pth --weights-only`

5. Finally, to train from checkpoint from previous step and 3 refinement stages in network, run `python train.py --train-images-folder <COCO_HOME>/train2017/ --prepared-train-labels prepared_train_annotation.pkl --val-labels val_subset.json --val-images-folder <COCO_HOME>/val2017/ --checkpoint-path <path_to>/checkpoint_iter_280000.pth --weights-only --num-refinement-stages 3`. We took checkpoint after 370000 iterations as the final one.

We did not perform the best checkpoint selection at any step, so similar result may be achieved after less number of iterations.

#### Known issue

We observe this error with maximum number of open files (`ulimit -n`) equals to 1024:

```
  File "train.py", line 164, in <module>
    args.log_after, args.val_labels, args.val_images_folder, args.val_output_name, args.checkpoint_after, args.val_after)
  File "train.py", line 77, in train
    for _, batch_data in enumerate(train_loader):
  File "/<path>/python3.6/site-packages/torch/utils/data/dataloader.py", line 330, in __next__
    idx, batch = self._get_batch()
  File "/<path>/python3.6/site-packages/torch/utils/data/dataloader.py", line 309, in _get_batch
    return self.data_queue.get()
  File "/<path>/python3.6/multiprocessing/queues.py", line 337, in get
    return _ForkingPickler.loads(res)
  File "/<path>/python3.6/site-packages/torch/multiprocessing/reductions.py", line 151, in rebuild_storage_fd
    fd = df.detach()
  File "/<path>/python3.6/multiprocessing/resource_sharer.py", line 58, in detach
    return reduction.recv_handle(conn)
  File "/<path>/python3.6/multiprocessing/reduction.py", line 182, in recv_handle
    return recvfds(s, 1)[0]
  File "/<path>/python3.6/multiprocessing/reduction.py", line 161, in recvfds
    len(ancdata))
RuntimeError: received 0 items of ancdata
```

To get rid of it, increase the limit to bigger number, e.g. 65536, run in the terminal: `ulimit -n 65536`

## Validation

1. Run `python val.py --labels <COCO_HOME>/annotations/person_keypoints_val2017.json --images-folder <COCO_HOME>/val2017 --checkpoint-path <CHECKPOINT>`

## Pre-trained model <a name="pre-trained-model"/>

The model expects normalized image (mean=[128, 128, 128], scale=[1/256, 1/256, 1/256]) in planar BGR format.
Pre-trained on COCO model is available at: https://download.01.org/opencv/openvino_training_extensions/models/human_pose_estimation/checkpoint_iter_370000.pth, it has 40% of AP on COCO validation set (38.6% of AP on the val *subset*).

#### Conversion to OpenVINO format

1. Convert PyTorch model to ONNX format: run script in terminal `python scripts/convert_to_onnx.py --checkpoint-path <CHECKPOINT>`. It produces `human-pose-estimation.onnx`.
2. Convert ONNX model to OpenVINO format with Model Optimizer: run in terminal `python <OpenVINO_INSTALL_DIR>/deployment_tools/model_optimizer/mo.py --input_model human-pose-estimation.onnx --input data --mean_values data[128.0,128.0,128.0] --scale_values data[256] --output stage_1_output_0_pafs,stage_1_output_1_heatmaps`. This produces model `human-pose-estimation.xml` and weights `human-pose-estimation.bin` in single-precision floating-point format (FP32).

## C++ Demo <a name="cpp-demo"/>

To run the demo download Intel&reg; OpenVINO&trade; Toolkit [https://software.intel.com/en-us/openvino-toolkit/choose-download](https://software.intel.com/en-us/openvino-toolkit/choose-download), install it and [build the samples](https://software.intel.com/en-us/articles/OpenVINO-InferEngine) (*Inferring Your Model with the Inference Engine Samples* part). Then run `<SAMPLES_BIN_FOLDER>/human_pose_estimation_demo -m <path_to>/human-pose-estimation.xml -i <path_to_video_file>` for the inference on `CPU`.

## Python Demo <a name="python-demo"/>

We provide python demo just for the quick results preview. Please, consider c++ demo for the best performance. To run the python demo from a webcam:
* `python demo.py --checkpoint-path <path_to>/checkpoint_iter_370000.pth --video 0`

## Citation:

If this helps your research, please cite the paper:

```
@inproceedings{osokin2018lightweight_openpose,
    author={Osokin, Daniil},
    title={Real-time 2D Multi-Person Pose Estimation on CPU: Lightweight OpenPose},
    booktitle = {arXiv preprint arXiv:1811.12004},
    year = {2018}
}
```