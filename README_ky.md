# SECOND for KITTI/NuScenes object detection (1.6.0 Alpha)
SECOND detector.

"Alpha" means there may be many bugs, config format may change, spconv API may change.

ONLY support python 3.6+, pytorch 1.0.0+. Tested in Ubuntu 16.04/18.04/Windows 10.

## Install

### 1. Clone code

```bash
git clone https://github.com/traveller59/second.pytorch.git
cd ./second.pytorch/second
```

### 2. Install dependence python packages

It is recommend to use Anaconda package manager.

```bash
conda install scikit-image scipy numba pillow matplotlib
```

```bash
pip install fire tensorboardX protobuf opencv-python
```

If you don't have Anaconda:

```bash
pip install numba scikit-image scipy pillow
```

Follow instructions in [spconv](https://github.com/traveller59/spconv) to install spconv. 

### 3. Setup cuda for numba (will be removed in 1.6.0 release)

you need to add following environment variable for numba.cuda, you can add them to ~/.bashrc:

```bash
export NUMBAPRO_CUDA_DRIVER=/usr/lib/x86_64-linux-gnu/libcuda.so
export NUMBAPRO_NVVM=/usr/local/cuda/nvvm/lib64/libnvvm.so
export NUMBAPRO_LIBDEVICE=/usr/local/cuda/nvvm/libdevice
```

### 4. add second.pytorch/ to PYTHONPATH

同样是`vim ~/.bashrc`

```shell
export PYTHONPATH=xxx/second.pytorch/:$PYTHONPATH
```



## Prepare dataset

* KITTI Dataset preparation

Download KITTI dataset and create some directories first:

```plain
└── KITTI_DATASET_ROOT
       ├── training    <-- 7481 train data
       |   ├── image_2 <-- for visualization
       |   ├── calib
       |   ├── label_2
       |   ├── velodyne
       |   └── velodyne_reduced <-- empty directory  （手动创建）
       └── testing     <-- 7580 test data
           ├── image_2 <-- for visualization
           ├── calib
           ├── velodyne
           └── velodyne_reduced <-- empty directory   (手动创建)
```

Then run
```bash
python create_data.py kitti_data_prep --root_path=KITTI_DATASET_ROOT
```

* Modify config file

There is some path need to be configured in config file:

```bash
train_input_reader: {
  ...
  database_sampler {
    database_info_path: "/path/to/dataset_dbinfos_train.pkl"
    ...
  }
  dataset: {
    dataset_class_name: "DATASET_NAME"
    kitti_info_path: "/path/to/dataset_infos_train.pkl"
    kitti_root_path: "DATASET_ROOT"
  }
}
...
eval_input_reader: {
  ...
  dataset: {
    dataset_class_name: "DATASET_NAME"
    kitti_info_path: "/path/to/dataset_infos_val.pkl"
    kitti_root_path: "DATASET_ROOT"
  }
}
```

## Usage

### train

I recommend to use script.py to train and eval. see script.py for more details.

#### train with single GPU

```bash
python ./pytorch/train.py train --config_path=./configs/car.fhd.config --model_dir=/path/to/model_dir
```

#### train with multiple GPU (need test, I only have one GPU)

Assume you have 4 GPUs and want to train with 3 GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,3 python ./pytorch/train.py train --config_path=./configs/car.fhd.config --model_dir=/path/to/model_dir --multi_gpu=True
```

Note: The batch_size and num_workers in config file is per-GPU, if you use multi-gpu, they will be multiplied by number of GPUs. Don't modify them manually.

You need to modify total step in config file. For example, 50 epochs = 15500 steps for car.lite.config and single GPU, if you use 4 GPUs, you need to divide ```steps``` and ```steps_per_eval``` by 4.

### evaluate

```bash
python ./pytorch/train.py evaluate --config_path=./configs/car.fhd.config --model_dir=/path/to/model_dir --measure_time=True --batch_size=1
```

* detection result will saved as a result.pkl file in model_dir/eval_results/step_xxx or save as official KITTI label format if you use --pickle_result=False.

  

## Try Kitti Viewer Web

### Major step

1. run ```python ./kittiviewer/backend/main.py main --port=xxxx``` in your server/local.

2. run ```cd ./kittiviewer/frontend && python -m http.server``` to launch a local web server.

3. open your browser and enter your frontend url (e.g. http://127.0.0.1:8000, default]).

4. input backend url (e.g. http://127.0.0.1:16666)

5. input root path, info path and det path (optional)

6. click load, loadDet (optional), input image index in center bottom of screen and press Enter.

### Inference step

Firstly the load button must be clicked and load successfully.

1. input checkpointPath and configPath.

2. click buildNet.

3. click inference.

![GuidePic](https://raw.githubusercontent.com/traveller59/second.pytorch/master/images/viewerweb.png)

## Concepts


* Kitti lidar box

A kitti lidar box is consist of 7 elements: [x, y, z, w, l, h, rz], see figure.

![Kitti Box Image](https://raw.githubusercontent.com/traveller59/second.pytorch/master/images/kittibox.png)

All training and inference code use kitti box format. So we need to convert other format to KITTI format before training.

* Kitti camera box

A kitti camera box is consist of 7 elements: [x, y, z, l, h, w, ry].
