Oncloud = True

if Oncloud:
    root = '/cephfs/PERSONAL/project/wangjing_data/mmseg/flower_dataset/'
    samples_per_gpu = 128
    workers_per_gpu = 32
    max_epochs = 500
else:
    root = '/media/jingjing/Seagate/Segmentation/flower_dataset/'
    samples_per_gpu = 32
    workers_per_gpu = 16
    max_epochs = 2
