from glob import glob
from config import root

split = 'val'

source_dir = root  # '/media/jingjing/Seagate/Segmentation/flower_dataset/'
folders = ["daisy", "dandelion", "rose", "sunflower", "tulip"]

f = open(source_dir + split + '.txt', "w")

for i in range(len(folders)):
    for file in glob(source_dir + split + '/' + folders[i] + '/*'):
        f.writelines(file + ' ' + str(i)+'\n')

f.close()
print('done')
