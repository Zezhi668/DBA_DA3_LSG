import torch
import numpy as np
import copy
from loop.loop_detect import LoopDetector
from loop.loop_rectify import LoopRectifier
from lietorch import SE3
import cv2
import os

class LoopModel:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg['device']['mapper']
        
        self.detector  = LoopDetector(cfg)
        self.rectifier = LoopRectifier(cfg)
        
        self.record_pair = []

    def _loop_max_depth(self):
        return float(
            self.cfg.get('looper', {}).get(
                'candidate_max_depth',
                self.cfg.get('looper', {}).get('max_depth', 8.0),
            )
        )
    
    def accept_thisloop(self, kfid_loop_start, kfid_loop_end):
        if len(self.record_pair) == 0:
            return True
        if not isinstance(kfid_loop_end, int): kfid_loop_end = kfid_loop_end.item()
        if not isinstance(kfid_loop_start, int): kfid_loop_start = kfid_loop_start.item()
            
        if kfid_loop_end - kfid_loop_start > 35:
            # 圈内的不要
            left_mask  = np.array(self.record_pair)[:, 0] < kfid_loop_start
            right_mask = np.array(self.record_pair)[:, 1] > kfid_loop_end
            if np.logical_and(left_mask, right_mask).sum() > 0: return False
            # 和之前圈距离之和小于20的不要
            MIN_ACCEPT_DISTANCE_TO_BEFORE = 20
            distance = np.abs(kfid_loop_start-np.array(self.record_pair)[:, 0]) + np.abs(kfid_loop_end-np.array(self.record_pair)[:, 1])
            if distance.min() < MIN_ACCEPT_DISTANCE_TO_BEFORE: return False
            return True    
        else:
            return False    
    
    def get_search_sorted_history_kf_index(self, history_kf_dict, curr_frame_dict):
        search_num         = self.cfg['looper']['search_num']
        loop_radius        = self.cfg['looper']['loop_radius']
        nms = 2
        distance_to_curr = torch.norm(curr_frame_dict['c2w'][:3,-1].unsqueeze(0) - history_kf_dict['c2w'][:, :3, -1], dim=-1) # (N, )
        sorted_distance_to_curr, sorted_history_kf_index = torch.sort(distance_to_curr)
        filtered_sorted_history_kf_index = sorted_history_kf_index[sorted_history_kf_index<(curr_frame_dict['idx']-loop_radius)] # This is index for curr_frame_dict.
        
        if filtered_sorted_history_kf_index.shape[0] == 0:
            return filtered_sorted_history_kf_index
        
        # normal_vec_to_curr = (history_kf_dict['c2w'][:, :3, -1] - curr_frame_dict['c2w'][:3,-1].unsqueeze(0)) # (N, 3)
        # normal_vec_to_curr = normal_vec_to_curr / (1e-4+torch.norm(normal_vec_to_curr, dim=-1, keepdim=True)) # (N, 3)
        
        # search_sorted_history_kf_index   = torch.concat([filtered_sorted_history_kf_index[:nearest_k_frame], sorted_history_kf_index[nearest_k_frame:-1:loop_search_interp]]) # (n, )
        search_sorted_history_kf_index    = torch.zeros(search_num, device=filtered_sorted_history_kf_index.device, dtype=filtered_sorted_history_kf_index.dtype)
        search_sorted_history_kf_index[0] = filtered_sorted_history_kf_index[0]
        cur_search_num = 1
        for idx in range(1, filtered_sorted_history_kf_index.shape[0]):
            curr_idx = filtered_sorted_history_kf_index[idx]
            # if  torch.abs(curr_idx - search_sorted_history_kf_index[:cur_search_num]).min() > nms and\
                # (normal_vec_to_curr[curr_idx].unsqueeze(0)*(normal_vec_to_curr[search_sorted_history_kf_index[:cur_search_num]])).sum(dim=1).max()<0.9999:
            if  torch.abs(curr_idx - search_sorted_history_kf_index[:cur_search_num]).min() > nms:
                search_sorted_history_kf_index[cur_search_num] = curr_idx
                cur_search_num += 1
            if cur_search_num >= search_num:
                break
        search_sorted_history_kf_index = search_sorted_history_kf_index[:cur_search_num]
        
        return search_sorted_history_kf_index        
        

    def detect(self, gaussian_model, history_kf_dict, curr_frame_dict, debug=False):
        '''
            (1) Start Loop Detect when len(history_kf_dict)>10+loop_radius;
            (2) We actually need a global history keyframe list.
        history_kf_dict: a dict contains:{'c2w': (K, 4, 4), 'rgb': (K, 3, H, W), (Optional)'depth': (K, 1, H, W), 'viz_out_idx_to_f_idx': list}, timestamp order.
        cur_frame: single dict, {'idx': int, 'c2w': (4, 4), 'rgb': (3, H, W), 'viz_out_idx_to_f_idx': list}.
        '''
        is_loop_mse_threshold = self.cfg['looper']['is_loop_mse_threshold']
        is_loop_med_times     = self.cfg['looper']['is_loop_med_times']
        
        # Step 1 Match cur_frame with history_kf_list and get error record of history matching results.
        search_sorted_history_kf_index = self.get_search_sorted_history_kf_index(history_kf_dict, curr_frame_dict)
        if search_sorted_history_kf_index.shape[0] == 0: return False, None
        
        # torch.save({'cur_rgb': gt_img2, 'history_rgb': history_kf_dict['rgb'][search_sorted_history_kf_index]}, "/data/wuke/workspace/VINGS-Mono/debug/loop_img.pt")
        error_list = []
        gt_img2    = curr_frame_dict['rgb']
        with torch.no_grad():
            for idx in range(search_sorted_history_kf_index.shape[0]):
                search_history_kf_index = search_sorted_history_kf_index[idx]
                gt_img1, c2w1           = history_kf_dict['rgb'][search_history_kf_index], history_kf_dict['c2w'][search_history_kf_index]
                gt_depth2               = curr_frame_dict['depth'].clone() + 0.0
                gt_depth2[gt_depth2 > self._loop_max_depth()] = 0.0
                # gt_depth2 = None
                c2w2                    = curr_frame_dict['c2w'].clone()
                pred_img1, mask_error   = self.detector.detect_loop(gt_img1, gt_img2, c2w2, gt_depth2, gaussian_model)
                error_list.append(mask_error)
                if mask_error < is_loop_mse_threshold:
                    break
        # torch.save({'cur_rgb': gt_img2, 'history_rgb': history_kf_dict['rgb'][search_sorted_history_kf_index]}, "/data/wuke/workspace/VINGS-Mono/debug/loop_img.pt")
        error_list_npy = np.array(error_list)
        
        # Step 2 Judge if loop exists and get relative pose of loop start to end.
        loopstart_filt_kf_id = np.argmin(error_list_npy)
        loopstart_kf_id      = search_sorted_history_kf_index[loopstart_filt_kf_id]
        is_loop              = (error_list_npy[loopstart_filt_kf_id] < is_loop_mse_threshold) or (error_list_npy[loopstart_filt_kf_id] < np.median(error_list_npy[error_list_npy<1.0])/(is_loop_med_times))
        
        if is_loop:
            if not self.accept_thisloop(loopstart_kf_id, curr_frame_dict['idx']):
                is_loop = False
            
        loopdetect_dict = {}
        if is_loop:
            gt_img1, c2w1 = history_kf_dict['rgb'][loopstart_kf_id], history_kf_dict['c2w'][loopstart_kf_id]
            tstamp1       = history_kf_dict['viz_out_idx_to_f_idx'][loopstart_kf_id]
            gt_depth2               = curr_frame_dict['depth'].clone() + 0.0
            gt_depth2[gt_depth2 > self._loop_max_depth()] = 0.0
            # gt_depth2 = None
            with torch.no_grad():
                c2w2 = curr_frame_dict['c2w']
                pred_img1, debug_dict = self.detector.detect_loop(gt_img1, gt_img2, c2w2, gt_depth2, gaussian_model, debug=True)
                if pred_img1 is None or not isinstance(debug_dict, dict) or 'c2w1' not in debug_dict:
                    return False, None
                kfid_loop_start       = loopstart_kf_id
                kfid_loop_end         = curr_frame_dict['idx']
                tstamp2               = history_kf_dict['viz_out_idx_to_f_idx'][kfid_loop_end]
                # c2w_loop_start2end    = debug_dict['c2w1'].inverse() @ c2w2
                c2w_loop_start2end    = torch.linalg.inv(c2w2) @ debug_dict['c2w1']
                loopdetect_dict['c2w_start2end'] = c2w_loop_start2end
                loopdetect_dict['start_kf_idx']  = kfid_loop_start
                loopdetect_dict['end_kf_idx']    = kfid_loop_end
                loopdetect_dict['pred_img1']     = pred_img1
                
                if not isinstance(kfid_loop_start, int): kfid_loop_start = kfid_loop_start.item()
                if not isinstance(kfid_loop_end, int):   kfid_loop_end = kfid_loop_end.item()
                
                self.record_pair.append([kfid_loop_start, kfid_loop_end])
                    
                # Draw mathing points pair if we detect a loop.
                DRAW_LINES = True
                if DRAW_LINES:
                    concat_image = np.concatenate((gt_img1.permute(1, 2, 0).cpu().numpy()[...,[2,1,0]]*255, gt_img2.permute(1, 2, 0).cpu().numpy()[...,[2,1,0]]*255), axis=1).astype(np.uint8)
                    vu_kp1_numpy ,vu_kp2_numpy = debug_dict['vu_kp1'].cpu().numpy(), debug_dict['vu_kp2'].cpu().numpy()
                    vu_kp2_numpy[:, 0] += gt_img1.shape[2]
                    # 定义直线的颜色和厚度
                    line_color = (0, 255, 0)  # 红色 (BGR 格式)
                    line_thickness = 2
                    # 在图像上绘制每一条直线
                    cv2.imwrite(f"{self.cfg['output']['save_dir']}/ply/{kfid_loop_start}_{kfid_loop_end}.png", concat_image)
                    concat_image = cv2.imread(f"{self.cfg['output']['save_dir']}/ply/{kfid_loop_start}_{kfid_loop_end}.png")
                    
                    for start, end in zip(vu_kp1_numpy, vu_kp2_numpy):
                        start_point = (int(start[0]),int(start[1]))
                        end_point = (int(end[0]),int(end[1]))
                        cv2.line(concat_image, start_point, end_point, line_color, line_thickness)
                
                print(f"Loop detected between {kfid_loop_start} and {kfid_loop_end}.")
                cv2.imwrite(f"{self.cfg['output']['save_dir']}/ply/{kfid_loop_start}_{kfid_loop_end}_error={debug_dict['mask_error'].detach().cpu().item()}.png", (np.concatenate([gt_img1.permute(1, 2, 0).cpu().numpy(), pred_img1.permute(1, 2, 0).cpu().numpy()], axis=1)*255).astype(np.uint8)[...,[2,1,0]])
                cv2.imwrite(f"{self.cfg['output']['save_dir']}/ply/{tstamp1}_{tstamp2}_matches.png", concat_image[...,[2,1,0]])
                

        if debug:
            debug_dict = {}
            debug_dict['error_list'] = error_list_npy
            return is_loop, loopdetect_dict, debug_dict
        else:
            return is_loop, loopdetect_dict
    
    
    def calc_score_v1(self, gaussian_model, intrinsic, old_c2w_list, old_f_list):
        num_gaussians = gaussian_model._xyz.shape[0]
        _scores      = torch.zeros((num_gaussians, 3), dtype=torch.float32, device=gaussian_model.device)
        _globalkfids = torch.zeros((num_gaussians, 3), dtype=torch.int32, device=gaussian_model.device)
        
        for curr_kf_id in range(len(old_f_list)):
            c2w = old_c2w_list[curr_kf_id]
            global_kf_id = old_f_list[curr_kf_id]
            w2c = torch.linalg.inv(c2w)
            
            # curr_scores = gaussian_model.render_scores(w2c, intrinsic) # (N, )
            rets = gaussian_model.render(w2c, intrinsic) # (N, )
            if rets['rgb'].max() < 0.1:
                gaussian_model.optimizer.zero_grad()
                continue
            
            (rets['rgb']*0).sum().backward()
            curr_scores = copy.deepcopy(gaussian_model._zeros.grad[:, 0])
            
            gaussian_model.optimizer.zero_grad()
            
            with torch.no_grad():
                curr_first_mask  = (_scores[:, 0] < curr_scores)
                curr_second_mask = (_scores[:, 0] > curr_scores) & (_scores[:, 1] < curr_scores)
                curr_third_mask  = (_scores[:, 1] > curr_scores) & (_scores[:, 2] < curr_scores)
                
                # First.
                _scores[curr_first_mask, 2] = _scores[curr_first_mask, 1].clone()
                _scores[curr_first_mask, 1] = _scores[curr_first_mask, 0].clone()
                _scores[curr_first_mask, 0] = curr_scores[curr_first_mask]
                _globalkfids[curr_first_mask, 2] = _globalkfids[curr_first_mask, 1].clone()
                _globalkfids[curr_first_mask, 1] = _globalkfids[curr_first_mask, 0].clone()
                _globalkfids[curr_first_mask, 0] = global_kf_id

                # Second.
                _scores[curr_second_mask, 2] = _scores[curr_second_mask, 1].clone()
                _scores[curr_second_mask, 1] = curr_scores[curr_second_mask]
                _globalkfids[curr_second_mask, 2] = _globalkfids[curr_second_mask, 1].clone()
                _globalkfids[curr_second_mask, 1] = global_kf_id

                # Third.
                _scores[curr_third_mask, 2] = curr_scores[curr_third_mask]
                _globalkfids[curr_third_mask, 2] = global_kf_id

        
        return _scores, _globalkfids # (N, 3), (N, 3)

    
    def calc_score_v2(self, gaussian_model, intrinsic, old_c2w_list, old_f_list):
        num_gaussians = gaussian_model._xyz.shape[0]
        _scores       = torch.ones((num_gaussians, 3), dtype=torch.float32, device=gaussian_model.device)
        # _globalkfids = torch.zeros((num_gaussians, 3), dtype=torch.int32, device=gaussian_model.device)
        _globalkfids  = gaussian_model._globalkf_id.reshape(-1, 1)
           
        return _scores, _globalkfids # (N, 3), (N, 3)
    
                
    def calc_score(self, gaussian_model, intrinsic, old_c2w_list, old_f_list):
        _scores, _globalkfids = self.calc_score_v1(gaussian_model, intrinsic, old_c2w_list, old_f_list)
        return _scores, _globalkfids


    def rectify(self, gaussian_model, tracker, history_kf_c2w, loopdetect_dict):
        '''
        history_kf_c2w: (N, 4, 4)
        loopdetect_dict = {'c2w_start2end': (4,4), 'start_kf_idx': int, 'end_kf_idx': int, 'pred_img2': (3, H, W)}
        '''
        loop_start_id, loop_end_id = loopdetect_dict['start_kf_idx'], loopdetect_dict['end_kf_idx']
        loop_start2end_c2w         = loopdetect_dict['c2w_start2end']
        print('start2end_c2w: ', loop_start2end_c2w)
        # Rectify history_kf_list's poses using PGO.
        recitified_history_kf_c2w, rectified_scales = self.rectifier.rectify_poses(history_kf_c2w, loop_start_id, loop_end_id, loop_start2end_c2w)
        # Rectify gaussians.
        self.rectifier.rectify_gaussians(loopdetect_dict, history_kf_c2w, recitified_history_kf_c2w, rectified_scales, self, gaussian_model)
        
        # Rectify tracker.frontend.video.poses, disps, disps_up. 
        #         tracker.frontend.video.poses_save, disps_save, disps_up_save. 
        if not self.cfg['mode'] == 'vo_nerfslam':
            self.rectifier.rectify_tracker(gaussian_model, tracker, loop_start_id, loop_end_id, recitified_history_kf_c2w)
        else:
            self.rectifier.rectify_tracker_nerfslam(gaussian_model, tracker, loop_start_id, loop_end_id, recitified_history_kf_c2w)
            
        # Retrain Gaussians.
        # Dangerous Opiton.
        # gaussian_model.setup_optimizer()
        # self.rectifier.retrain_gaussian(gaussian_model, tracker, loop_start_id, loop_end_id)
        
        # Storage Control.
        
        
        return history_kf_c2w, recitified_history_kf_c2w

    def preprocess_input(self):
        history_kf_dict = {'c2w': None, 'rgb': None}
        curr_frame_dict = {'c2w': None, 'rgb': None}
        return history_kf_dict, curr_frame_dict
        
    def run(self, gaussian_model, tracker, viz_out, idx=None, use_correct=True):
        # TODO, wuke: Decrease frame_num to be matched (<50 frames is great).
        '''
        history_kf_dict is {"c2w": SE3(tracker.frontend.video.images_up_save.poses_save).inv().matrix(), "rgb": tracker.frontend.video.images_up_save}.
        curr_frame_dict is {"idx": viz_out["global_kf_id"][-1], "c2w": viz_out["poses"][-1], "rgb": viz_out["images"][-1]}.
        '''
        if not self.cfg['mode'] == 'vo_nerfslam':
            last_kf = viz_out["global_kf_id"][-1]
        else:
            last_kf = torch.sum(tracker.visual_frontend.cam0_timestamps>0).item() + int(tracker.visual_frontend.cam0_timestamps[0]==0)
            
        if not self.cfg['mode'] == 'vo_nerfslam':
            history_kf_dict = {"c2w": SE3(tracker.frontend.video.poses_save[:last_kf+1]).inv().matrix().to(self.device), "rgb": tracker.frontend.video.images_up_save[:last_kf+1].to(self.device).permute(0, 3, 1, 2)}
            history_kf_dict["viz_out_idx_to_f_idx"] = tracker.frontend.video.tstamp_save[:last_kf+1].cpu().tolist()
            # "c2w": (num_global_kf, 4, 4), "rgb": (num_global_kf, 3, H, W)
        else:
            history_kf_dict = {"c2w": SE3(tracker.visual_frontend.cam0_T_world[:last_kf+1]).inv().matrix().to(self.device), "rgb": tracker.visual_frontend.cam0_images[:last_kf+1].to(self.device)/255.0}
            history_kf_dict["viz_out_idx_to_f_idx"] = tracker.visual_frontend.cam0_timestamps[:last_kf+1].cpu().tolist()
        
        '''
        history_kf_dict["rgb"]'s max value is 1.
        viz_out["images"]'s max value is 1.
        '''
        curr_frame_dict = {"idx": last_kf, "c2w": viz_out["poses"][-1], "rgb": viz_out["images"][-1].permute(2, 0, 1)[[2,1,0],...], "depth": viz_out["depths"][-1].permute(2, 0, 1)}
        curr_frame_dict["timestamp"] = viz_out["global_kf_id"][-1]
        # torch.save({"history_c2ws": history_kf_dict['c2w'], "curr_c2w": curr_frame_dict['c2w']}, "/data/wuke/workspace/VINGS-Mono/debug/loop_img.pt")
        # with torch.no_grad():
        is_loop, loopdetect_dict = self.detect(gaussian_model, history_kf_dict, curr_frame_dict)

        if not self.cfg['mode'] == 'vo_nerfslam':
            poses_before = SE3(tracker.frontend.video.poses_save[:last_kf+1]).inv().matrix()
        else:
            poses_before = SE3(tracker.visual_frontend.cam0_T_world[:last_kf+1]).inv().matrix()
        
        if is_loop:
            '''
            torch.save({'history_kf_dict': history_kf_dict, 'curr_frame_dict': curr_frame_dict, 'loopdetect_dict': loopdetect_dict}, '/data/wuke/workspace/VINGS-Mono/debug/debug_loop.pt')
            '''
            print("Loop detected!")
            
            # -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - 
            # TTD 2024/10/16, Save Tracker and Mapper.
            # Save Mathed KeyPoints.
            
            # torch.save(viz_out, self.cfg['output']['save_dir']+'/ply/viz_out_'+str(idx).zfill(5)+'.pt')
            # tracker.save_pt_ckpt("{}/ply/tracker_{}_beforecorrect.pt".format(self.cfg['output']['save_dir'], str(idx).zfill(5)))
            # gaussian_model.save_pt_ckpt("{}/ply/mapper_{}_beforecorrect.pt".format(self.cfg['output']['save_dir'], str(idx).zfill(5)))
            
            # tstamps = tracker.frontend.video.tstamp_save[:tracker.frontend.video.count_save+tracker.frontend.video.count_save_bias]
            # torch.save({"poses_before": poses_before, "poses_after": poses_after, "tstamps": tstamps, "loopdetect_dict": loopdetect_dict, "history_kf_dict": history_kf_dict}, os.path.join(self.cfg['output']['save_dir'], 'ply', 'poses_beforeafter_'+str(idx).zfill(5)+'.pt'))
            
            # TTD 2024/12/10
            # Save storage_manager's properties.
            
            # -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - 
            
            
            # ⭐ 
            use_correct = True # False # True
            if use_correct:
                recitified_history_kf_c2w, new_xyz_world = self.rectify(gaussian_model, tracker, history_kf_dict['c2w'], loopdetect_dict)
            
            if not self.cfg['mode'] == 'vo_nerfslam':
                poses_after = SE3(tracker.frontend.video.poses_save[:last_kf+1]).inv().matrix()
            else:
                poses_after = SE3(tracker.visual_frontend.cam0_T_world[:last_kf+1]).inv().matrix()
            
            # torch.save({"poses_before": poses_before, "poses_after": poses_after, "loop_start_kfid": loopdetect_dict['start_kf_idx'], "loop_end_kfid": loopdetect_dict['end_kf_idx']}, os.path.join(self.cfg['output']['save_dir'], 'ply', 'poses_beforeafter_'+str(idx).zfill(5)+'.pt'))
            
            # TTD 2024/10/15
            # Update *.txts.
            if not self.cfg['mode'] == 'vo_nerfslam':
                update_file_names = list(map(lambda x: str(x).zfill(6)+'.0.txt', tracker.frontend.video.tstamp_save[:tracker.frontend.video.count_save].cpu().to(torch.int32).tolist()))
                for idx in range(len(update_file_names)):
                    name = update_file_names[idx]
                    new_c2w = SE3(tracker.frontend.video.poses_save[idx]).inv().matrix().numpy()
                    np.savetxt(os.path.join(self.cfg['output']['save_dir'], 'droid_c2w', name), new_c2w)
            else:
                update_file_names = list(map(lambda x: str(x).zfill(6)+'.0.txt', tracker.visual_frontend.cam0_timestamps[:tracker.visual_frontend.kf_idx].cpu().to(torch.int32).tolist()))
                for idx in range(len(update_file_names)):
                    name = update_file_names[idx]
                    new_c2w = SE3(tracker.visual_frontend.cam0_T_world[idx]).cpu().inv().matrix().numpy()
                    np.savetxt(os.path.join(self.cfg['output']['save_dir'], 'droid_c2w', name), new_c2w)
                
        else:
            pass
        
