

import numpy as np
import sys
from utils.kitti import load_dt_annos, load_gt_annos
from utils.eval import KittiAveragePrecision
from utils.eval import apMethod

from sklearn.metrics import mean_squared_error
import time

def main():
    RESULT_FILES_PATH=sys.argv[1]
    GT_ANNOS_PATH = './data/kitti_3d/training/label_2/'
    DATA_SPLIT_FILE = './data/kitti_3d/split/val.txt'

    with open(DATA_SPLIT_FILE, 'r') as f:
        lines = f.readlines()
    filelist = [int(line) for line in lines]

    CLASSES = ('Car', 'Pedestrian', 'Cyclist')
    dt_annos = load_dt_annos(RESULT_FILES_PATH, filelist)
    gt_annos = load_gt_annos(GT_ANNOS_PATH, filelist)

    ap = KittiAveragePrecision(gt_annos, dt_annos, apMethod.interpHyb41)
    reth41 = ap.kitti_eval()

    print(reth41[:,0,:,0].flatten()*100)
    print(reth41[:,1,:,0].flatten()*100)
    print(reth41[:,2,:,0].flatten()*100)
    
    

if __name__ == '__main__':
    main()
