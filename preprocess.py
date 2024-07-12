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
# GLOBAL VARS
############################################################
DATA_DIR = os.getenv('DATA_DIR') #set as environment variable
if DATA_DIR is None:DATA_DIR = './data'
CHIP_DIR  = DATA_DIR + '/chips'
CHIP_SIZE = 224
FEATURES_CSV  = f'{DATA_DIR}/features.csv'

# IMAGE LISTDIR
blue_images  = sorted(glob.glob(DATA_DIR + '/*/T*B02*.jp2'))  # blue in any .SAFE dir
green_images = sorted(glob.glob(DATA_DIR + '/*/T*B03*.jp2'))  # green in any .SAFE dir
red_images   = sorted(glob.glob(DATA_DIR + '/*/T*B04*.jp2'))  # red in any .SAFE dir
all_bands    = blue_images + green_images + red_images 

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
    '''        
    Just checking times here....
    src = rio.open(all_bands[0],'r')
    start_time = time.time()
    windows = []
    for i,x in enumerate(range(0,src.height-CHIP_SIZE,CHIP_SIZE)):
        for j,y in enumerate(range(0,src.width-CHIP_SIZE,CHIP_SIZE)):
            windows.append(((i,j),rio.windows.Window(y,x,CHIP_SIZE,CHIP_SIZE)))
    end_time = time.time()
    print(f"Execution time: {round(end_time-start_time,4)} seconds.")  

    start_time = time.time()
    n_chip_rows = src.height // CHIP_SIZE
    n_chip_cols = src.width // CHIP_SIZE
    n_chip = n_chip_rows * n_chip_cols
    windows = []
    for k in range(n_chip):
        i = k // n_chip_cols
        j = k % n_chip_cols
        x = i * CHIP_SIZE
        y = j * CHIP_SIZE
        windows.append(((i,j),rio.windows.Window(y,x,CHIP_SIZE,CHIP_SIZE)))
    end_time = time.time()
    print(f"Execution time: {round(end_time-start_time,4)} seconds.")   <-- 1/2 but both <0.01 sec
    '''

def chip_and_normalize_single_band(input_path,index):
    #some feedback
    filename = input_path.split("/")[-1]
    print(f'[{index}/{N}] Processing {filename}')
    
    #input variables
    src        = rio.open(input_path,'r')
    raster_max = src.statistics(1).max
    raster_min = src.statistics(1).min
    basename   = os.path.basename(input_path).split('.')[0]

    #output variables
    kwargs = src.meta.copy()
    kwargs.update({'driver':'GTiff','height':CHIP_SIZE,'width':CHIP_SIZE,'dtype':np.float32})

    windows = []     
    for i,x in enumerate(range(0,src.height-CHIP_SIZE,CHIP_SIZE)):
        for j,y in enumerate(range(0,src.width-CHIP_SIZE,CHIP_SIZE)):
            windows.append(((i,j),rio.windows.Window(col_off=y,row_off=x,width=CHIP_SIZE,height=CHIP_SIZE)))
    
    for (i,j),w in windows:
        #read and normalize
        chip = src.read(1,window=w)
        chip = (chip-raster_min)/(raster_max-raster_min) #Normalize to [0,1]

        #write to output file
        output_path = f"{CHIP_DIR}/{basename}_CHIP_{i:02d}_{j:02d}.tif"        
        with rio.open(output_path, 'w',**kwargs) as dst:
            dst.write(chip,indexes=1)

    #signal done and close
    print(f'{filename} done.')
    src.close()


def chip_and_normalize(input_path_b,index):
    #some feedback
    filename = input_path_b.split("/")[-1][0:-12]
    print(f'[{index}/{N}] Processing {filename}')
    
    #input variables
    input_path_g = input_path_b.replace('_B02_','_B03_')
    input_path_r = input_path_b.replace('_B02_','_B04_')
    src_b      = rio.open(input_path_b,'r')
    src_g      = rio.open(input_path_g,'r')
    src_r      = rio.open(input_path_r,'r')
    b_max = src_b.statistics(1).max
    b_min = src_b.statistics(1).min
    g_max = src_g.statistics(1).max
    g_min = src_g.statistics(1).min
    r_max = src_r.statistics(1).max
    r_min = src_r.statistics(1).min    
    basename   = os.path.basename(input_path_b).split('.')[0].replace('_B02_','_')

    #output variables
    kwargs = src_b.meta.copy()
    kwargs.update({'driver':'GTiff','height':CHIP_SIZE,'width':CHIP_SIZE,'dtype':np.float32,'count':3,'photometric':'RGB'})

    windows = []     
    for i,x in enumerate(range(0,src_b.height-CHIP_SIZE,CHIP_SIZE)):
        for j,y in enumerate(range(0,src_b.width-CHIP_SIZE,CHIP_SIZE)):
            windows.append(((i,j),rio.windows.Window(col_off=y,row_off=x,width=CHIP_SIZE,height=CHIP_SIZE)))
    
    for (i,j),w in windows:
        #read and normalize
        chip_b = src_b.read(1,window=w)
        chip_g = src_g.read(1,window=w)
        chip_r = src_r.read(1,window=w)        
        chip_b = (chip_b-b_min)/(b_max-b_min) #Normalize to [0,1]
        chip_g = (chip_g-g_min)/(g_max-g_min) #Normalize to [0,1]
        chip_r = (chip_r-r_min)/(r_max-r_min) #Normalize to [0,1]
        chip = np.stack([chip_r,chip_g,chip_b],axis=0)
        
        #write to output file
        output_path = f"{CHIP_DIR}/{basename}_CHIP_{i:02d}_{j:02d}.tif"        
        with rio.open(output_path, 'w',**kwargs) as dst:
            dst.write(chip,indexes=[1,2,3])

    # clean up and signal done
    src_b.close()
    src_g.close()
    src_r.close()    
    for path in [input_path_b,input_path_g,input_path_r]:
        if os.path.isfile(path+'.aux.xml'):
            os.remove(path+'.aux.xml')
    print(f'{filename} done.')


def print_sample_chip_bands():
    sample_n = 1500
    b2_paths = sorted(glob.glob(CHIP_DIR + '/T*B02*.tif')) 
    sample_b2_path = b2_paths[sample_n]
    sample_b3_path = sample_b2_path.replace('_B02_','_B03_')
    sample_b4_path = sample_b2_path.replace('_B02_','_B04_')

     # read images and stack
    sample_b = sio.imread(sample_b2_path)
    sample_g = sio.imread(sample_b3_path)
    sample_r = sio.imread(sample_b4_path)
    rgb = np.stack([sample_r,sample_g,sample_b],axis=2)

    normalize = lambda x: (x-x.min())/(x.max()-x.min()) #brightness
    plt.imsave('./figs/sample_chip.png',normalize(rgb)*2)
    print("Sample chip saved to ./figs/sample_chip.png")




############################################################
# CLASSES
############################################################
class SentinelDataset(torch.utils.data.Dataset):
    def __init__(self):
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

############################################################
# MAIN
############################################################
if __name__ == '__main__':

    chip_images      = True
    extract_features = True

    # II.SPLIT/CHIP IMAGES
    #-----------------------------------------------------------
    if chip_images is True:
        # Set output dir if not set
        if not os.path.isdir(CHIP_DIR):
            os.mkdir(CHIP_DIR)

        # Set the number of threads/workers
        if os.cpu_count() > 16:
            n_workers = 16
        else:
            n_workers = os.cpu_count()

        # Multi-threaded pass (throw rasters to each thread)
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=n_workers) as ex: 
            ex.map(chip_and_normalize,all_bands,range(1,N+1))
        end_time = time.time()
        print(f"Execution time: {round(end_time-start_time,4)} seconds.")

        # Print a sample
        print_sample_chip()


    # III.PASS CHIPS THRU NETWORK (AND SAVE FEATURES)
    #-----------------------------------------------------------
    if extract_features is True:
        # DATASET AND DATALOADER
        dataset    = SentinelDataset()
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

        # LOAD RESNET-50
        resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub',
            'nvidia_resnet50',
            pretrained=True,
            trust_repo=True,
            verbose=True,
            skip_validation=True
            )
        resnet50.fc = torch.nn.Identity() # Remove final classification layer
        resnet50 = resnet50.eval()
        resnet50 = resnet50.to(device)
        # -------
        # In case above model does not load, use torchvision resnet:
        # resnet50 = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        # resnet50.fc = torch.nn.Identity() # Remove final classification layer
        # resnet50 = resnet50.eval()
        # resnet50 = resnet50.to(device)

        # PASS CHIPS THRU NET
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

        # SAVE FEATURES TO DISK
        np.savetxt(FEATURES_CSV,np.array(features_list),delimiter=',')