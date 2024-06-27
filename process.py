############################################################
# LIBRARIES
############################################################
import numpy as np
import glob
import tqdm
import rasterio as rio
import numpy as np
import os
import skimage.io as sio
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import time
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import pandas as pd

%matplotlib inline
plt.style.use('fast')


############################################################
# VARS
############################################################
DATA_DIR = os.getenv('DATA_DIR') #set as environment variable
if DATA_DIR is None:DATA_DIR = '.'
CHIP_DIR  = DATA_DIR + '/chips'
CHIP_SIZE = 224

blue_images  = sorted(glob.glob(DATA_DIR + '/*/T*B02*.jp2'))  # blue in any .SAFE dir
green_images = sorted(glob.glob(DATA_DIR + '/*/T*B03*.jp2'))  # green in any .SAFE dir
red_images   = sorted(glob.glob(DATA_DIR + '/*/T*B04*.jp2'))  # red in any .SAFE dir
all_bands    = blue_images + green_images + red_images 

############################################################
# LIBRARIES
############################################################
# SET DEVICE
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps') #mac
else:
    device = torch.device('cpu')
print(f'Using {device} for inference')

############################################################
# FUNCTIONS
############################################################
#--------------------------------------------------
# FUNCTION Def
#--------------------------------------------------
def chip_and_normalize(input_path,output_dir):
    #input variables
    src        = rio.open(input_path,'r')
    raster_max = src.statistics(1).max
    raster_min = src.statistics(1).min
    basename   = os.path.basename(input_path).split('.')[0]

    #output variables
    kwargs     = src.meta.copy()
    kwargs.update({'driver':'GTiff','height':CHIP_SIZE,'width':CHIP_SIZE,'dtype':np.float32})
    
    #Loop thru chips (by i row, j column)
    for i,x in enumerate(range(0,src.height-CHIP_SIZE,CHIP_SIZE)):
        for j,y in enumerate(range(0,src.width-CHIP_SIZE,CHIP_SIZE)):

            #output path
            output_path = f"{output_dir}/{basename}_CHIP_{i:02d}_{j:02d}.tif"

            #Read window from input and normalize
            win  = rio.windows.Window(col_off=y,row_off=x,width=CHIP_SIZE,height=CHIP_SIZE)
            chip = src.read(1,window=win)
            chip = (chip - raster_min)/(raster_max - raster_min) #Normalize to [0,1]
            # chip = (chip * 65535).astype(int) #Stretch to [0,65535]

            #write to output file
            with rio.open(output_path, 'w',**kwargs) as dst:
                dst.write(chip,indexes=1)

    #feedback and close
    print(f'{input_path} done.')
    src.close()
    return

############################################################
# CLASSES
############################################################

#--------------------------------------------------
# MULTI-THREADED PASS of above
#--------------------------------------------------
if not os.path.isdir(CHIP_DIR):
    os.mkdir(CHIP_DIR)

start_time = time.time()

with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor: 
    for raster_path in all_bands:
        executor.submit(chip_and_normalize,raster_path,CHIP_DIR)
        
end_time = time.time()
print(f"Execution time: {round(end_time-start_time,4)} seconds.")

#--------------------------------------------------
# # Sequential (single-thread) processing in case above code does not work.
#--------------------------------------------------
# start_time = time.time()
# for i,raster_path in enumerate(all_bands):
#     print(f"[{i+1}/{len(all_bands)}] ",end='')
#     chip_and_normalize(raster_path,CHIP_DIR)
# end_time = time.time()
# print(f"Execution time: {round(end_time-start_time,4)} seconds.")

############################################################
# LIBRARIES
############################################################
b2_paths = sorted(glob.glob(CHIP_DIR + '/T*B02*.tif')) #--- paths, len is 2401 * nr.of images
sample_b2_path = b2_paths[1500]
sample_b3_path = sample_b2_path.replace('_B02_','_B03_')
sample_b4_path = sample_b2_path.replace('_B02_','_B04_')

sample_b = sio.imread(sample_b2_path) #--- read images and stack
sample_g = sio.imread(sample_b3_path)
sample_r = sio.imread(sample_b4_path)
rgb = np.stack([sample_r,sample_g,sample_b],axis=2)

normalize = lambda x: (x-x.min())/(x.max()-x.min())
plt.imshow(normalize(rgb)*2) #--- Plot with range set to [0,1] * 2 for higher brightness
#< ------change to plt.savefig()

############################################################
# LIBRARIES
############################################################
class SentinelDataset(torch.utils.data.Dataset):
    def __init__(self):
        # self.images_b = sio.imread_collection(f'{CHIP_DIR}/*_B02_10m_CHIP_*_*.tif')
        # self.images_g = sio.imread_collection(f'{CHIP_DIR}/*_B03_10m_CHIP_*_*.tif')
        # self.images_r = sio.imread_collection(f'{CHIP_DIR}/*_B04_10m_CHIP_*_*.tif')
        self.images_b = sorted(glob.glob(f'{CHIP_DIR}/*_B02_10m_CHIP_*_*.tif'))
        self.images_g = sorted(glob.glob(f'{CHIP_DIR}/*_B03_10m_CHIP_*_*.tif'))
        self.images_r = sorted(glob.glob(f'{CHIP_DIR}/*_B04_10m_CHIP_*_*.tif'))
    
    def __getitem__(self, index):
        b = sio.imread(self.images_b[index])
        g = sio.imread(self.images_g[index])
        r = sio.imread(self.images_r[index])
        rgb_img = torch.from_numpy(np.stack([r,g,b], axis=0))
        return rgb_img

    def __len__(self):
        return len(self.images_b)


dataset    = SentinelDataset()
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)


resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained = True,trust_repo=True,verbose=True,skip_validation=True)
resnet50.fc = torch.nn.Identity() # Remove final classification layer
resnet50 = resnet50.eval()
# resnet50 = resnet50.to(device)
# -------
# In case above model does not load, use torchvision resnet:
# resnet50 = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
# resnet50.fc = torch.nn.Identity() # Remove final classification layer
# resnet50 = resnet50.eval()
# resnet50 = resnet50.to(device)

FEATURES_CSV  = f'{DATA_DIR}/features.csv'
features_list = []

start_time = time.time()

with torch.no_grad():
    for _,x in tqdm.tqdm(enumerate(dataloader), total=(len(dataset))):
        x = x.to(device)
        y = resnet50(x)
        y = y.cpu().numpy()
        features_list.append(y)

stop_time  = time.time()
execution_time = stop_time - start_time
print(f"Execution time: {execution_time} seconds")

#SAVE FEATURES TO DISK
np.savetxt(FEATURES_CSV,np.array(features_list),delimiter=',')