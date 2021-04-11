

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
    
    ######
    ap = KittiAveragePrecision(gt_annos, dt_annos, apMethod.interpHyb40)
    rettmp = ap.kitti_eval()

    ap.set_eval_method(apMethod.interp11)
    t_start = time.perf_counter()
    ret11 = ap.kitti_eval()
    t_end = time.perf_counter()
    t_11 = t_end - t_start

    ap.set_eval_method(apMethod.interpAll)
    t_start = time.perf_counter()
    retAll = ap.kitti_eval()
    t_end = time.perf_counter()
    t_ALL = t_end - t_start

    ap.set_eval_method(apMethod.interp40)
    t_start = time.perf_counter()
    ret40 = ap.kitti_eval()
    t_end = time.perf_counter()
    t_40 = t_end - t_start

    ap.set_eval_method(apMethod.interp41)
    t_start = time.perf_counter()
    ret41 = ap.kitti_eval()
    t_end = time.perf_counter()
    t_41 = t_end - t_start

    ap.set_eval_method(apMethod.interpHyb11)
    t_start = time.perf_counter()
    reth11 = ap.kitti_eval()
    t_end = time.perf_counter()
    t_h11 = t_end - t_start

    ap.set_eval_method(apMethod.interpHyb40)
    t_start = time.perf_counter()
    reth40 = ap.kitti_eval()
    t_end = time.perf_counter()
    t_h40 = t_end - t_start

    ap.set_eval_method(apMethod.interpHyb41)
    t_start = time.perf_counter()
    reth41 = ap.kitti_eval()
    t_end = time.perf_counter()
    t_h41 = t_end - t_start

    print('='*20)
    print(f'All-point Time:{t_ALL}')
    print(f'11-point Time:{t_11}')
    print(f'40-point Time:{t_40}')
    print(f'41-point Time:{t_41}')
    print(f'h11-point Time:{t_h11}')
    print(f'h40-point Time:{t_h40}')
    print(f'h41-point Time:{t_h41}')


    print('='*20)
    car11 = ret11[:,0,:,0].flatten()*100
    car40 = ret40[:,0,:,0].flatten()*100
    car41 = ret41[:,0,:,0].flatten()*100
    carall = retAll[:,0,:,0].flatten()*100
    carhyb11 = reth11[:,0,:,0].flatten()*100
    carhyb40 = reth40[:,0,:,0].flatten()*100
    carhyb41 = reth41[:,0,:,0].flatten()*100
    print('Car 11P MSE:{}'.format(mean_squared_error(car11,carall)))
    print('Car 40P MSE:{}'.format(mean_squared_error(car40,carall)))
    print('Car 41P MSE:{}'.format(mean_squared_error(car41,carall)))
    print('Car h11P MSE:{}'.format(mean_squared_error(carhyb11,carall)))
    print('Car h40P MSE:{}'.format(mean_squared_error(carhyb40,carall)))
    print('Car h41P MSE:{}'.format(mean_squared_error(carhyb41,carall)))

    print('='*20)
    ped11 = ret11[:,1,:,0].flatten()*100
    ped40 = ret40[:,1,:,0].flatten()*100
    ped41 = ret41[:,1,:,0].flatten()*100
    pedall = retAll[:,1,:,0].flatten()*100
    pedhyb11 = reth11[:,1,:,0].flatten()*100
    pedhyb40 = reth40[:,1,:,0].flatten()*100
    pedhyb41 = reth41[:,1,:,0].flatten()*100
    print('PED 11 MSE:{}'.format(mean_squared_error(ped11,pedall)))
    print('PED 40 MSE:{}'.format(mean_squared_error(ped40,pedall)))
    print('PED 41 MSE:{}'.format(mean_squared_error(ped41,pedall)))
    print('PED hyb11 MSE:{}'.format(mean_squared_error(pedhyb11,pedall)))
    print('PED hyb40 MSE:{}'.format(mean_squared_error(pedhyb40,pedall)))
    print('PED hyb41 MSE:{}'.format(mean_squared_error(pedhyb41,pedall)))

    print('='*20)
    cyc11 = ret11[:,2,:,0].flatten()*100
    cyc40 = ret40[:,2,:,0].flatten()*100
    cyc41 = ret41[:,2,:,0].flatten()*100
    cycall = retAll[:,2,:,0].flatten()*100
    cychyb11 = reth11[:,2,:,0].flatten()*100
    cychyb40 = reth40[:,2,:,0].flatten()*100
    cychyb41 = reth41[:,2,:,0].flatten()*100
    print('CYC 11 MSE:{}'.format(mean_squared_error(cyc11,cycall)))
    print('CYC 40 MSE:{}'.format(mean_squared_error(cyc40,cycall)))
    print('CYC 41 MSE:{}'.format(mean_squared_error(cyc41,cycall)))
    print('CYC hyb11 MSE:{}'.format(mean_squared_error(cychyb11,cycall)))
    print('CYC hyb40 MSE:{}'.format(mean_squared_error(cychyb40,cycall)))
    print('CYC hyb41 MSE:{}'.format(mean_squared_error(cychyb41,cycall)))

if __name__ == '__main__':
    main()
