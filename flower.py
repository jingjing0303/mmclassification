# Load the base config file
from mmcv import Config
from mmcls.utils import auto_select_device
from config import *

# Specify the path of the config file and checkpoint file.
checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/mobilenet_v2/mobilenet_v2_batch256_imagenet_20200708-3b2dc3af.pth'

cfg = Config.fromfile('configs/mobilenet_v2/mobilenet-v2_8xb32_in1k.py')
cfg.device = auto_select_device()

# Modify the number of classes in the head.
cfg.model.head.num_classes = 5
cfg.model.head.topk = (1, )

# Load the pre-trained model's checkpoint.
cfg.model.backbone.init_cfg = dict(type='Pretrained', checkpoint=checkpoint_file, prefix='backbone')

# Specify sample size and number of workers.
cfg.data.samples_per_gpu = samples_per_gpu
cfg.data.workers_per_gpu = workers_per_gpu

# Specify the path and meta files of training dataset
cfg.data.train.data_prefix = root + 'train/'
cfg.data.train.ann_file = root + 'train.txt'
cfg.data.train.classes = root + 'classes.txt'

# Specify the path and meta files of validation dataset
cfg.data.val.data_prefix = root + 'val/'
cfg.data.val.ann_file = root + 'val.txt'
cfg.data.val.classes = root + 'classes.txt'

# Specify the path and meta files of test dataset
cfg.data.test.data_prefix = root + 'val/'
cfg.data.test.ann_file = root + 'test.txt'
cfg.data.test.classes = root + 'classes.txt'

# Specify the normalization parameters in data pipeline
normalize_cfg = dict(type='Normalize', mean=[124.508, 116.050, 106.438], std=[58.577, 57.310, 57.437], to_rgb=True)
cfg.data.train.pipeline[3] = normalize_cfg
cfg.data.val.pipeline[3] = normalize_cfg
cfg.data.test.pipeline[3] = normalize_cfg

# Modify the evaluation metric
cfg.evaluation['metric_options']={'topk': (1, )}

# Specify the optimizer
cfg.optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
cfg.optimizer_config = dict(grad_clip=None)

# Specify the learning rate scheduler
cfg.lr_config = dict(policy='step', step=1, gamma=0.1)

# cfg.runner.max_iters = 200
# cfg.log_config.interval = 10
# cfg.evaluation.interval = 200
# cfg.checkpoint_config.interval = 200

cfg.runner = dict(type='EpochBasedRunner', max_epochs=max_epochs)

# Specify the work directory
cfg.work_dir = './work_dirs/flower'

# Output logs for every 10 iterations
cfg.log_config.interval = 10

# Set the random seed and enable the deterministic option of cuDNN
# to keep the results' reproducible.
from mmcls.apis import set_random_seed
cfg.seed = 0
set_random_seed(0, deterministic=True)

cfg.gpu_ids = range(1)


import time
import mmcv
import os.path as osp

from mmcls.datasets import build_dataset
from mmcls.models import build_classifier
from mmcls.apis import train_model

# Create the work directory
mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
# Build the classifier
model = build_classifier(cfg.model)
model.init_weights()
# Build the dataset
datasets = [build_dataset(cfg.data.train)]
# Add `CLASSES` attributes to help visualization
model.CLASSES = datasets[0].CLASSES
# Start fine-tuning
train_model(
    model,
    datasets,
    cfg,
    distributed=False,
    validate=True,
    timestamp=time.strftime('%Y%m%d_%H%M%S', time.localtime()),
    meta=dict())


