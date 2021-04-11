import numpy as np
from enum import Enum 
# import pickle
from .overlap import calculate_iou_partly
from .overlap import image_box_overlap
np.set_printoptions(precision=2,suppress=True)

class apMethod(Enum):
    interp11 = 1
    interp40 = 2
    interp41 = 9
    interpAll = 3
    interpHyb11 = 4
    interpHyb40 = 5
    interpHyb41 = 6

class KittiAveragePrecision:
    def __init__(self, 
                gt_annos, 
                dt_annos, 
                method=apMethod.interp11,
                current_classes=['Car', 'Pedestrian', 'Cyclist'], 
                eval_types=['bbox', 'bev', '3d']):
        
        self.class_to_name = {0: 'Car', 1: 'Pedestrian', 2: 'Cyclist', 3: 'Van', 4: 'Person_sitting'}
        self.gt_annos = gt_annos
        self.dt_annos = dt_annos
        self.eval_types = eval_types
        self.current_classes = current_classes
        self.min_overlaps = None
        self.eval_method = None
        self.set_eval_method(method)
        self.difficultys = [0, 1, 2]
        self.mAPbbox = None
        self.mAPbev = None
        self.mAP3d = None
        self.mAPaos = None

    def set_eval_method(self, method):
        assert method in apMethod
        self.eval_method = method
        print(f'Set Evaluation Method to {self.eval_method}')

    def set_difficulty(self, diff):
        assert isinstance(diff, list)
        self.difficultys = diff

    def kitti_eval(self):
        self._eval_type_check()
        self._get_current_class()
        self._get_min_overlap()
        ret = self._do_eval()
        # print(ret)
        # ap_str, ap_dict = self._format_ap(ret)
        return ret
    
    def _eval_type_check(self):
        assert len(self.eval_types) > 0, 'must contain at least one evaluation type'
        if 'aos' in self.eval_types:
            assert 'bbox' in self.eval_types, 'must evaluate bbox when evaluating aos'
        
        compute_aos = False
        pred_alpha = False
        valid_alpha_gt = False
        for anno in self.dt_annos:
            if anno['alpha'].shape[0] != 0:
                pred_alpha = True
                break
        for anno in self.gt_annos:
            if anno['alpha'][0] != -10:
                valid_alpha_gt = True
                break
        compute_aos = (pred_alpha and valid_alpha_gt)
        if compute_aos:
            self.eval_types.append('aos')

    def _get_current_class(self):
        name_to_class = {v: n for n, v in self.class_to_name.items()}
        if not isinstance(self.current_classes, (list, tuple)):
            self.current_classes = [self.current_classes]
        current_classes_int = []
        for curcls in self.current_classes:
            if isinstance(curcls, str):
                current_classes_int.append(name_to_class[curcls])
            else:
                current_classes_int.append(curcls)
        self.current_classes = current_classes_int

    def _get_min_overlap(self):
        overlap_0_7 = np.array([[0.7, 0.5, 0.5, 0.7,
                                0.5], [0.7, 0.5, 0.5, 0.7, 0.5],
                                [0.7, 0.5, 0.5, 0.7, 0.5]])
        overlap_0_5 = np.array([[0.7, 0.5, 0.5, 0.7, 0.5],
                                [0.5, 0.25, 0.25, 0.5, 0.25],
                                [0.5, 0.25, 0.25, 0.5, 0.25]])
        self.min_overlaps = np.stack([overlap_0_7, overlap_0_5], axis=0)[:, :, self.current_classes]

    def _do_eval(self):
        
        ap_info = []
        if 'bbox' in self.eval_types:
            ap_info.append(self._eval_class(0))

        if 'bev' in self.eval_types:
            ap_info.append(self._eval_class(1))
        
        if '3d' in self.eval_types:
            ap_info.append(self._eval_class(2))
            # ap_dict['3d'] = self._format_ap(ap_info)
        ap_info = self._format_ap(ap_info)
        return ap_info

    def _eval_class(self, metric, num_parts=200):
        '''
        Args:
            metric (int): Eval type. 0: bbox, 1: bev, 2: 3d
        '''
        compute_aos = ('aos' in self.eval_types) and (metric==0)
        
        rets = calculate_iou_partly(self.dt_annos, self.gt_annos, metric)
        overlaps, total_dt_num, total_gt_num = rets
        ap_list = []
        save_infomat_list = []
        for m, current_class in enumerate(self.current_classes):
            # print(self.class_to_name[current_class])
            for idx_l, difficulty in enumerate(self.difficultys):
                rets = self._prepare_data(current_class, difficulty)
                (gt_datas_list, dt_datas_list, ignored_gts, ignored_dets,
                dontcares, total_dc_num, total_num_valid_gt) = rets
                for k, min_overlap in enumerate(self.min_overlaps[:, metric, m]):
                    ttp, tfn, tfp = 0, 0, 0
                    infosmat = []
                    # print(f'metric:{metric}, class:{current_class}, difficulty:{difficulty}, min_overlap:{min_overlap}')
                    thresh = 1.0
                    for i in range(len(self.gt_annos)):
                        info = (overlaps[i], gt_datas_list[i], dt_datas_list[i], ignored_gts[i], ignored_dets[i], dontcares[i])
                        rets = self.firstloop(info, metric, min_overlap)
                        if thresh > rets: thresh = rets
                    for i in range(len(self.gt_annos)):
                        info = (overlaps[i], gt_datas_list[i], dt_datas_list[i], ignored_gts[i], ignored_dets[i], dontcares[i])
                        rets = self._get_results_list_v1(info, metric, min_overlap, thresh, compute_aos)
                        tp, fn, fp, mat = rets
                        ttp += tp
                        tfn += fn
                        tfp += fp
                        for ix, x in enumerate(mat):
                            if x == -1: continue
                            infosmat.append([dt_datas_list[i][:, -1][ix], x])
                    infosmat = np.array(infosmat)
                    infosmat = infosmat[np.lexsort(-infosmat.T[0,None])]
                    save_infomat_list.append(infosmat)
                    # print(total_num_valid_gt)
                    # print(ttp, tfp, tfn)
                    ret = self._get_AP(infosmat, misc=(ttp,tfn, total_num_valid_gt))
                    ap_list.append(ret)
                    # break
                    # return 0
        # with open(f'infomat_{metric}.pkl','wb') as f:
        #     pickle.dump(save_infomat_list, f)
        ap_list = np.array(ap_list).reshape((3,3,2))
        return ap_list

    @staticmethod
    def _get_results_list_v1(info, metric, min_overlap, thresh, compute_aos=False):
        '''
        Args:
        return:
            res:    FP:0, TP:1, IGNORE:-1
        '''
        (overlaps, gt_datas, dt_datas, ignored_gt, ignored_det, dc_bboxes) = info

        gt_size = gt_datas.shape[0]
        det_size = dt_datas.shape[0]
        dt_scores = dt_datas[:, -1]
        dt_bboxes = dt_datas[:, :4]

        # -1: ignore, 0:false positive, 1:true positive
        assigned_detection = [0] * det_size
        ignored_threshold = [False] * det_size
        for i in range(det_size):
            if (dt_scores[i] < thresh):
                ignored_threshold[i] = True
        tp, fp, fn, similarity = 0, 0, 0, 0
        for i in range(gt_size):
            if ignored_gt[i] == -1: continue
            det_idx = -1
            valid_detection = False
            max_overlap = 0
            assigned_ignored_det = False
            for j in range(det_size):
                if ignored_det[j] == -1: continue
                if assigned_detection[j] != 0: continue
                if ignored_threshold[j]: continue

                overlap = overlaps[j,i]
                dt_score = dt_scores[j]

                if ((overlap > min_overlap)
                    and (overlap > max_overlap or assigned_ignored_det)
                    and (ignored_det[j] == 0)):
                    max_overlap = overlap
                    det_idx = j 
                    valid_detection = True
                    assigned_ignored_det = False
                elif ((overlap > min_overlap)
                    and (valid_detection == False)
                    and (ignored_det[j] == 1)):
                    det_idx = j 
                    valid_detection = True
                    assigned_ignored_det = True
            
            if ((valid_detection == False)
                and (ignored_gt[i] == 0)):
                fn += 1
            elif ((valid_detection == True)
                and (ignored_gt[i] == 1 or ignored_det[det_idx] == 1)):
                assigned_detection[det_idx] = -1
            elif (valid_detection == True):
                assigned_detection[det_idx] = 1
                tp += 1
        for j in range(det_size):
            if ignored_det[j] != 0:
                assigned_detection[j] = -1
                continue
            if ignored_threshold[j]:
                assigned_detection[j] = -1
                continue
            if (assigned_detection[j] == 0):
                fp += 1
                assigned_detection[j] = 0
        
        if metric == 0:
            overlaps_dt_dc = image_box_overlap(dt_bboxes, dc_bboxes, 0)        
            for i in range(dc_bboxes.shape[0]):
                for j in range(det_size):
                    if assigned_detection[j] != 0: continue
                    if (ignored_threshold[j]): continue
                    if overlaps_dt_dc[j,i] > min_overlap:
                        assigned_detection[j] = -1
                        fp -= 1
        return tp, fn, fp, assigned_detection
    
    @staticmethod
    def firstloop(info, metric, min_overlap):
        
        (overlaps, gt_datas, dt_datas, ignored_gt, ignored_det, dc_bboxes) = info

        gt_size = gt_datas.shape[0]
        det_size = dt_datas.shape[0]
        dt_scores = dt_datas[:, -1]
        dt_bboxes = dt_datas[:, :4]

        assigned_detection = [False] * det_size
        min_score = 100.0
        for i in range(gt_size):
            if ignored_gt[i] == -1: continue
            det_idx = -1
            valid_detection = False
            max_overlap = -1
            max_score = -1
            for j in range(det_size):
                if ignored_det[j] == -1: continue
                if assigned_detection[j]: continue

                overlap = overlaps[j,i]
                dt_score = dt_scores[j]
                
                if overlap <= min_overlap: continue
                if dt_score > max_score: 
                    det_idx = j
                    valid_detection = True
                    max_score = dt_score

            if ((valid_detection == True)
                and (ignored_gt[i] ==0)
                and (ignored_det[det_idx] == 0)):
                if min_score > dt_scores[det_idx]:
                    min_score = dt_scores[det_idx]
        return min_score

    def _get_results_list_v2(self, info, metric, min_overlap, compute_aos=False):
        '''
        NO GARANTEE!!!
        NEED TO CONSIDER BEST OVERLAP MATCH.
        Args:
        return:
            res:    FP:0, TP:1, IGNORE:-1
        '''
        (overlaps, gt_datas, dt_datas, ignored_gt, ignored_det, dc_bboxes) = info

        gt_size = gt_datas.shape[0]
        det_size = dt_datas.shape[0]
        dt_scores = dt_datas[:, -1]
        dt_bboxes = dt_datas[:, :4]
        
        # -1: ignore, 0:false positive, 1:true positive
        assigned_detection = [False] * det_size
        retmat = [-1] * det_size

        tp, fp, fn, similarity = 0, 0, 0, 0
        for i in range(gt_size):
            if ignored_gt[i] == -1: continue
            det_idx = -1
            valid_detection = False
            max_overlap = -1
            max_score = -1
            for j in range(det_size):
                if ignored_det[j] == -1: continue
                if assigned_detection[j]: continue

                overlap = overlaps[j,i]
                dt_score = dt_scores[j]
                
                if overlap <= min_overlap: continue
                
                # if ignored_det[j] == 0:
                #     if overlap > max_overlap:
                #         max_overlap = overlap
                #         det_idx = j
                #         valid_detection = True
                # elif ignored_det[j] == 1:
                #     if valid_detection == False:
                #         det_idx = j
                #         valid_detection = True

                if ignored_det[j] == 0:
                    if dt_score > max_score:
                        max_score = dt_score
                        max_overlap = overlap
                        det_idx = j
                        valid_detection = True
                    elif ((dt_score == max_score)
                        and (overlap > max_overlap)):
                        max_overlap = overlap
                        det_idx = j
                elif ignored_det[j] == 1:
                    if valid_detection == False:
                        det_idx = j
                        valid_detection = True
            
            if valid_detection == True:
                if (ignored_gt[i] == 0 and ignored_det[det_idx] == 0):
                    assigned_detection[det_idx] = True
                    retmat[det_idx] = 1
                    tp += 1
                else:
                    assigned_detection[det_idx] = True
                    retmat[det_idx] = -1
            elif valid_detection == False:
                if ignored_gt[i] == 0:
                    fn += 1

        for j in range(det_size):
            if assigned_detection[j]: continue
            if ignored_det[j] != 0: continue
            fp += 1
            retmat[j] = 0
        
        if metric == 0:
            overlaps_dt_dc = image_box_overlap(dt_bboxes, dc_bboxes, 0)        
            for i in range(dc_bboxes.shape[0]):
                for j in range(det_size):
                    if assigned_detection[j]: continue
                    if ignored_det[j] != 0: continue
                    if overlaps_dt_dc[j,i] > min_overlap:
                        assigned_detection[j] = True
                        retmat[j] = -1
                        fp -= 1
        return tp, fn, fp, retmat

    def _get_mAP(self, infosmat, misc):
        tp_num, fn_num, gt_num = misc
        # gt_num = tp_num + fn_num
        
        # pr[:,0]: tp, pr[:,1]: fp
        pr = np.zeros((tp_num,2),dtype=np.float64)
        tp, fp, sc, ix = 0, 0, 0, 0
        tpflag = False
        for info in infosmat:
            if ((info[1] == 1)):
                tp += 1
                sc = info[0]
                pr[ix,0] = tp
                pr[ix,1] = fp
                ix += 1
                tpflag = True
            elif info[1] == 0:
                if tpflag and info[0]<sc:
                    tpflag = False
                fp += 1
                if tpflag and (not(info[0]<sc)):
                    pr[ix-1,1] = fp

        indexlist = []
        if self.eval_method == apMethod.interp11:
            N = 11
            P = 10.0
            indexlist = [0]
        elif self.eval_method == apMethod.interp40:
            N = 40
            P = N
        elif self.eval_method == apMethod.interpAll:
            N = gt_num
            P = N
        elif self.eval_method == apMethod.interpHyb11:
            N = 11
            P = 10.0
            indexlist = [0]
        elif self.eval_method == apMethod.interpHyb40:
            N = 40
            P = N
        elif self.eval_method == apMethod.interpHyb41:
            N = 41
            P = 40.0
            indexlist = [0]

        interval = gt_num / P
        t = int(tp_num / interval)        
        # BASE
        for i in range(1,t+1):
            indexlist.append(int(i * interval+0.4999)-1)
        # print(len(indexlist))
        precision = np.zeros(N)
        for ix, ixpr in enumerate(indexlist):
            precision[ix] = pr[ixpr,0] / (pr[ixpr,0] + pr[ixpr,1])
        
        ap = 0
        if ((self.eval_method == apMethod.interp11)
            or (self.eval_method == apMethod.interp40)):
            if indexlist==[] or indexlist[-1]+1 < tp_num:
                precision[len(indexlist)] = pr[-1][0] / (pr[-1][0] + pr[-1][1])
            for ix,_ in enumerate(precision):
            # for ix,_ in enumerate(precision[:-1]):
                precision[ix] = np.max(precision[ix:])
                ap += precision[ix]
            # print(precision)            
            ap /= N
            # print(ap)

        elif ((self.eval_method == apMethod.interpHyb11)
            or (self.eval_method == apMethod.interpHyb40)
            or (self.eval_method == apMethod.interpHyb41)):
            for ix in range(len(indexlist)):
                precision[ix] = np.max(precision[ix:])
                ap += precision[ix]
            ap = ap / N
            ap_tail = 0
            prec_tail = []
            for ix in range(indexlist[-1],tp_num):
                prec_tail.append(pr[ix][0] / (pr[ix][0] + pr[ix][1]))
            prec_tail = np.array(prec_tail)
            for ix in range(prec_tail.shape[0]):
                prec_tail[ix] = np.max(prec_tail[ix:])
                ap_tail += prec_tail[ix]
            ap_tail /= gt_num
            ap = ap + ap_tail
        elif (self.eval_method == apMethod.interpAll):
            interval = gt_num / 10
            t = int(tp_num / interval)
            stop_11point = int(t * interval+0.4999)-1
            interval = gt_num / 40
            t = int(tp_num / interval)
            stop_40point = int(t * interval+0.4999)-1

            # ap = 0
            # for ix in range(stop_11point):
            #     precision[ix] = np.max(precision[ix:])
            #     ap += precision[ix]
            # ap /= gt_num
            # print(f'11points:{ap}')

            # ap = 0
            # for ix in range(stop_40point):
            #     precision[ix] = np.max(precision[ix:])
            #     ap += precision[ix]
            # ap /= gt_num
            # print(f'40points:{ap}')

            ap = 0
            for ix in range(tp_num):
                precision[ix] = np.max(precision[ix:])
                ap += precision[ix]
            ap /= gt_num
            # # print(f'Allpoints:{ap}')

        # print(ap)

        return ap
    
    def _get_AP(self, infosmat, misc):
        tp_num, fn_num, gt_num = misc
        pr = self._get_pr(infosmat, tp_num)
        ap = 0
        if self.eval_method == apMethod.interp11:
            ap = self._get_11P(pr, (tp_num,gt_num))
        elif self.eval_method == apMethod.interp40:
            ap = self._get_40P(pr, (tp_num,gt_num))
        elif self.eval_method == apMethod.interp41:
            ap = self._get_41P(pr, (tp_num,gt_num))
        elif self.eval_method == apMethod.interpAll:
            ap = self._get_allP(pr, (tp_num,gt_num))
        elif self.eval_method == apMethod.interpHyb11:
            ap = self._get_h11P(pr, (tp_num,gt_num))
        elif self.eval_method == apMethod.interpHyb40:
            ap = self._get_h40P(pr, (tp_num,gt_num))
        elif self.eval_method == apMethod.interpHyb41:
            ap = self._get_h41P(pr, (tp_num,gt_num))
        return ap

    @staticmethod
    def _get_pr(infosmat, tp_num):
        pr = np.zeros((tp_num,3),dtype=np.float64)
        tp, fp, sc, ix = 0, 0, 0, 0
        tpflag = False
        for info in infosmat:
            if ((info[1] == 1)):
                tp += 1
                sc = info[0]
                pr[ix,0] = tp
                pr[ix,1] = fp
                ix += 1
                tpflag = True
            elif info[1] == 0:
                if tpflag and info[0]<sc:
                    tpflag = False
                fp += 1
                if tpflag and (not(info[0]<sc)):
                    pr[ix-1,1] = fp
                    
        pr[:,2] = pr[:,0] / (pr[:,0] + pr[:,1])
        return pr
    
    @staticmethod
    def _get_11P(pr, misc):
        tp_num, gt_num = misc
        
        num_interval = 11.0
        interP = np.linspace(0,1,num=11,endpoint=True)
        tp_per_ip = gt_num / 10
        num_ip = int(tp_num / tp_per_ip)
        
        indexlist = [0]
        for i in range(1,num_ip+1):
            indexlist.append(int(interP[i]*gt_num + 0.4999)-1)
        
        if indexlist[-1] != tp_num-1: indexlist.append(tp_num-1)    
        
        prmax = []
        for index in indexlist:
            prmax.append(pr[index][2])
        
        prmax = np.array(prmax)
        ap = 0    
        for index,_ in enumerate(prmax):
            ap += np.max(prmax[index:])
        ap /= num_interval
            
        return ap

    @staticmethod
    def _get_40P(pr, misc):
        tp_num, gt_num = misc
        
        num_interval = 40.0
        interP = np.linspace(1/40,1,num=40,endpoint=True)
        tp_per_ip = gt_num / 40
        num_ip = int(tp_num / tp_per_ip)
        
        indexlist = []
        for i in range(num_ip):
            indexlist.append(int(interP[i]*gt_num + 0.4999)-1)
            
        if indexlist==[] or indexlist[-1] != tp_num-1: indexlist.append(tp_num-1)    
        
        prmax = []
        for index in indexlist:
            prmax.append(pr[index][2])
        
        prmax = np.array(prmax)
        ap = 0    
        for index,_ in enumerate(prmax):
            ap += np.max(prmax[index:])
        ap /= num_interval
            
        return ap

    @staticmethod
    def _get_41P(pr, misc):
        tp_num, gt_num = misc
        
        num_interval = 41.0
        interP = np.linspace(0,1,num=41,endpoint=True)
        tp_per_ip = gt_num / 41
        num_ip = int(tp_num / tp_per_ip)
        
        indexlist = []
        for i in range(num_ip):
            indexlist.append(int(interP[i]*gt_num + 0.4999)-1)
            
        if indexlist==[] or indexlist[-1] != tp_num-1: indexlist.append(tp_num-1)    
        
        prmax = []
        for index in indexlist:
            prmax.append(pr[index][2])
        
        prmax = np.array(prmax)
        ap = 0    
        for index,_ in enumerate(prmax):
            ap += np.max(prmax[index:])
        ap /= num_interval
            
        return ap

    @staticmethod
    def _get_allP(pr, misc):
        tp_num, gt_num = misc
        
        ap = 0
        for index,_ in enumerate(pr):
            ap += np.max(pr[index:,2])
        ap /= gt_num
            
        return ap

    @staticmethod
    def _get_h11P(pr, misc):
        tp_num, gt_num = misc
        
        num_interval = 11.0
        interP = np.linspace(0,1,num=11,endpoint=True)
        tp_per_ip = gt_num / 11
        num_ip = int(tp_num / tp_per_ip)
        
        indexlist = []
        for i in range(num_ip):
            indexlist.append(int(interP[i]*gt_num + 0.4999)-1)
        
        prmax = []
        for index in indexlist:
            prmax.append(pr[index][2])
        prmax = np.array(prmax)
        ap = 0    
        for index,_ in enumerate(prmax):
            ap += np.max(prmax[index:])
        ap /= num_interval
        
        ap2 = 0
        lp = -1 + int(num_ip * tp_per_ip + 0.4999)    
        if lp != tp_num:
            for index in range(lp+1,tp_num):
                ap2 += np.max(pr[index:,2])
            ap2 /= gt_num
        ap += ap2
        return ap

    @staticmethod
    def _get_h40P(pr, misc):
        tp_num, gt_num = misc
        
        num_interval = 40.0
        interP = np.linspace(1/40,1,num=40,endpoint=True)
        tp_per_ip = gt_num / 40
        num_ip = int(tp_num / tp_per_ip)
        
        indexlist = []
        for i in range(num_ip):
            indexlist.append(int(interP[i]*gt_num + 0.4999)-1)
        
        prmax = []
        for index in indexlist:
            prmax.append(pr[index][2])
        prmax = np.array(prmax)
        ap = 0    
        for index,_ in enumerate(prmax):
            ap += np.max(prmax[index:])
        ap /= num_interval
        
        ap2 = 0
        lp = -1 + int(num_ip * tp_per_ip + 0.4999)    
        if lp != tp_num:
            for index in range(lp+1,tp_num):
                ap2 += np.max(pr[index:,2])
            ap2 /= gt_num
        ap += ap2
        return ap

    @staticmethod
    def _get_h41P(pr, misc):
        tp_num, gt_num = misc
        
        num_interval = 41.0
        interP = np.linspace(0,1,num=41,endpoint=True)
        tp_per_ip = gt_num / 41
        num_ip = int(tp_num / tp_per_ip)
        
        indexlist = []
        for i in range(num_ip):
            indexlist.append(int(interP[i]*gt_num + 0.4999)-1)
        
        prmax = []
        for index in indexlist:
            prmax.append(pr[index][2])
        prmax = np.array(prmax)
        ap = 0    
        for index,_ in enumerate(prmax):
            ap += np.max(prmax[index:])
        ap /= num_interval
        
        ap2 = 0
        lp = -1 + int(num_ip * tp_per_ip + 0.4999)    
        if lp != tp_num:
            for index in range(lp+1,tp_num):
                ap2 += np.max(pr[index:,2])
            ap2 /= gt_num
        ap += ap2
        return ap

    def _prepare_data(self, current_class, difficulty):
        gt_datas_list = []
        dt_datas_list = []
        total_dc_num = []
        ignored_gts, ignored_dets, dontcares = [], [], []
        total_num_valid_gt = 0
        for i in range(len(self.gt_annos)):
            rets = self._clean_data(self.gt_annos[i], self.dt_annos[i], current_class, difficulty)
            num_valid_gt, ignored_gt, ignored_det, dc_bboxes = rets
            ignored_gts.append(np.array(ignored_gt, dtype=np.int64))
            ignored_dets.append(np.array(ignored_det, dtype=np.int64))

            if len(dc_bboxes) == 0:
                dc_bboxes = np.zeros((0, 4)).astype(np.float64)
            else:
                dc_bboxes = np.stack(dc_bboxes, 0).astype(np.float64)

            total_dc_num.append(dc_bboxes.shape[0])
            dontcares.append(dc_bboxes)
            total_num_valid_gt += num_valid_gt
            gt_datas = np.concatenate(
                [self.gt_annos[i]['bbox'], self.gt_annos[i]['alpha'][..., np.newaxis]], 1)
            dt_datas = np.concatenate([
                self.dt_annos[i]['bbox'], self.dt_annos[i]['alpha'][..., np.newaxis],
                self.dt_annos[i]['score'][..., np.newaxis]
            ], 1)
            gt_datas_list.append(gt_datas)
            dt_datas_list.append(dt_datas)

        total_dc_num = np.stack(total_dc_num, axis=0)
        return (gt_datas_list, dt_datas_list, ignored_gts, ignored_dets, dontcares, total_dc_num, total_num_valid_gt)

    @staticmethod
    def _clean_data(gt_anno, dt_anno, current_class, difficulty):
        CLASS_NAMES = ['car', 'pedestrian', 'cyclist']
        MIN_HEIGHT = [40, 25, 25]
        MAX_OCCLUSION = [0, 1, 2]
        MAX_TRUNCATION = [0.15, 0.3, 0.5]
        dc_bboxes, ignored_gt, ignored_dt = [], [], []
        current_cls_name = CLASS_NAMES[current_class].lower()
        num_gt = len(gt_anno['name'])
        num_dt = len(dt_anno['name'])
        num_valid_gt = 0
        for i in range(num_gt):
            bbox = gt_anno['bbox'][i]
            gt_name = gt_anno['name'][i].lower()
            height = bbox[3] - bbox[1]
            valid_class = -1
            if (gt_name == current_cls_name):
                valid_class = 1
            elif (current_cls_name == 'Pedestrian'.lower()
                and 'Person_sitting'.lower() == gt_name):
                valid_class = 0
            elif (current_cls_name == 'Car'.lower() and 'Van'.lower() == gt_name):
                valid_class = 0
            else:
                valid_class = -1

            ignore = False
            if ((gt_anno['occluded'][i] > MAX_OCCLUSION[difficulty])
                    or (gt_anno['truncated'][i] > MAX_TRUNCATION[difficulty])
                    or (height <= MIN_HEIGHT[difficulty])):
                ignore = True

            if valid_class == 1 and not ignore:
                ignored_gt.append(0)
                num_valid_gt += 1
            elif (valid_class == 0 or (ignore and (valid_class == 1))):
                ignored_gt.append(1)
            else:
                ignored_gt.append(-1)

        # for i in range(num_gt):
            if gt_anno['name'][i] == 'DontCare':
                dc_bboxes.append(gt_anno['bbox'][i])
        
        for i in range(num_dt):
            if (dt_anno['name'][i].lower() == current_cls_name):
                valid_class = 1
            else:
                valid_class = -1

            height = abs(dt_anno['bbox'][i, 3] - dt_anno['bbox'][i, 1])
            if height < MIN_HEIGHT[difficulty]:
                ignored_dt.append(1)
            elif valid_class == 1:
                ignored_dt.append(0)
            else:
                ignored_dt.append(-1)

        return num_valid_gt, ignored_gt, ignored_dt, dc_bboxes

    def _format_ap(self, ap_info):
        # for im, metric in enumerate(self.eval_types):
        #     if metric == 'aos': continue
        #     for m, current_class in enumerate(self.current_classes):
        #         for idx_l, difficulty in enumerate(self.difficultys):
        #             for k, min_overlap in enumerate(self.min_overlaps[:, im, m]):
        #                 print(metric,current_class,difficulty,min_overlap)
        ap_info = np.array(ap_info)


        # print(ap_info[:,0,:,0].flatten()*100)
        # print(ap_info[:,1,:,0].flatten()*100)
        # print(ap_info[:,2,:,0].flatten()*100)
        return ap_info
