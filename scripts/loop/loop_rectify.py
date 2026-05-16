import torch
import copy
import os
import random
import gtsam
from lietorch import SE3
import numpy as np
import sys
import os
# sys.path.append('/data/wuke/workspace/VINGS-Mono/scripts')
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from vings_utils.pytorch3d_utils import q2R, R2q
from vings_utils.gtsam_utils import matrix_to_tq
from scipy.optimize import minimize
from gaussian.loss_utils import get_loss

class LoopRectifier:
    def __init__(self, cfg):
        self.cfg = cfg
    
    def rectify_poses_v1(self, poses, loop_start_id, loop_end_id, loop_start2end_c2w):
        '''
        torch.save( {'poses': poses, 'loop_start_id': loop_start_id, 'loop_end_id': loop_end_id, 'loop_start2end_c2w': loop_start2end_c2w}, '/data/wuke/workspace/VINGS-Mono/debug/debug_loop.pth')
        poses: (K, 4, 4)
        '''
        poses_device = poses.device
        if torch.is_tensor(poses): poses = poses.cpu().numpy()
        if torch.is_tensor(loop_start2end_c2w): loop_start2end_c2w = loop_start2end_c2w.cpu().numpy()
        
        raw_trans = torch.tensor(np.matmul(np.linalg.inv(poses[1:]), poses[:-1])[:, :3, -1], dtype=torch.float32)
        
        graph = gtsam.NonlinearFactorGraph()
        initial_estimate = gtsam.Values()
        
        noise_model = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2]))
        
        for kf_idx in range(loop_start_id, loop_end_id+1):
            pose = poses[kf_idx]
            # Add pose node.
            initial_estimate.insert(kf_idx, gtsam.Pose3(gtsam.Rot3(pose[:3, :3]), pose[:3, 3]))
            # Add edge.
            if kf_idx+1 <= poses.shape[0]-1:
                relative_pose = np.linalg.inv(poses[kf_idx]) @ poses[kf_idx+1]
                transform = gtsam.Pose3(gtsam.Rot3(relative_pose[:3, :3]), relative_pose[:3, 3])
                graph.add(gtsam.BetweenFactorPose3(kf_idx, kf_idx+1, transform, noise_model))

        # Add loop constraint.
        loop_transform   = gtsam.Pose3(gtsam.Rot3(loop_start2end_c2w[:3, :3]), loop_start2end_c2w[:3, 3])
        
        loop_noise_model = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2]))
        
        graph.add(gtsam.BetweenFactorPose3(loop_start_id, loop_end_id, loop_transform, loop_noise_model))
        
        # Optimize.
        params = gtsam.LevenbergMarquardtParams()
        params.setVerbosityLM("ERROR")
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
        result = optimizer.optimize()

        # 此处假设loop_start_id的global_pose是不变的，这样是为了保证变换前后能保证前面的pose和现在的一致;
        pose = result.atPose3(loop_start_id)
        pose_matrix = np.identity(4)
        pose_matrix[:3, :3] = pose.rotation().matrix()
        pose_matrix[:3, 3] = pose.translation()
        relative_loop_start_c2w = poses[loop_start_id] @ np.linalg.inv(pose_matrix)
        
        rectified_poses = []
        for kf_idx in range(poses.shape[0]):
            if kf_idx>=loop_start_id and kf_idx<=loop_end_id:
                pose = result.atPose3(kf_idx)
                pose_matrix = np.identity(4)
                pose_matrix[:3, :3] = pose.rotation().matrix()
                pose_matrix[:3, 3] = pose.translation()
                pose_matrix = relative_loop_start_c2w @ pose_matrix
                rectified_poses.append(torch.tensor(pose_matrix, device=poses_device))
            else:
                rectified_poses.append(torch.tensor(poses[kf_idx], device=poses_device))
        rectified_poses = torch.stack(rectified_poses, dim=0).to(torch.float32)
        
        # TTD 2024/11/09
        # rectified_trans = torch.matmul(torch.linalg.inv(rectified_poses[1:]), rectified_poses[:-1])[:, :3, -1]
        # new_scales      = [torch.tensor(1.0)]
        # for kf_idx in range(loop_start_id, loop_end_id):
        #    new_scales.append(torch.norm(rectified_trans[kf_idx].cpu())/torch.norm(raw_trans[kf_idx]))
        # new_scales = torch.stack(new_scales).reshape(-1,1,1).to(torch.float32).to(poses_device)
        new_scales = None
        
        return rectified_poses, new_scales
    
    def rectify_poses_v2(self, poses, loop_start_id, loop_end_id, loop_start2end_c2w):
        prior_rpy_sigma, prior_xyz_sigma = 1, 0.3
        odometry_xyz_sigma, odometry_rpy_sigma = 0.2, 1
        PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([prior_rpy_sigma*np.pi/180,
                                                                prior_rpy_sigma*np.pi/180,
                                                                prior_rpy_sigma*np.pi/180,
                                                                prior_xyz_sigma,
                                                                prior_xyz_sigma,
                                                                prior_xyz_sigma]))
        ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([odometry_rpy_sigma*np.pi/180,
                                                                odometry_rpy_sigma*np.pi/180,
                                                                odometry_rpy_sigma*np.pi/180,
                                                                odometry_xyz_sigma, odometry_xyz_sigma, odometry_xyz_sigma]))

        poses_device = poses.device
        if torch.is_tensor(poses): poses = poses.cpu().numpy()
        if torch.is_tensor(loop_start2end_c2w): loop_start2end_c2w = loop_start2end_c2w.cpu().numpy()
        
        graph = gtsam.NonlinearFactorGraph()
        initial_estimate = gtsam.Values()
        parameters = gtsam.ISAM2Params()
        parameters.setRelinearizeThreshold(0.1)
        parameters.relinearizeSkip = 1
        isam = gtsam.ISAM2(parameters)
        
        true_poses = [gtsam.Pose3(poses[i]) for i in range(poses.shape[0])]
        odometry_tf = [true_poses[i-1].transformPoseTo(true_poses[i]) for i in range(1, len(true_poses))]
        odometry_xyz = [(odometry_tf[i].x(), odometry_tf[i].y(), odometry_tf[i].z()) for i in range(len(odometry_tf))]
        odometry_rpy = [odometry_tf[i].rotation().rpy() for i in range(len(odometry_tf))]

        noisy_measurements = [np.hstack((odometry_rpy[i],odometry_xyz[i])) for i in range(len(odometry_tf))]

        graph.push_back(gtsam.PriorFactorPose3(loop_start_id+1, true_poses[0], PRIOR_NOISE))
        initial_estimate.insert(loop_start_id+1, true_poses[0].compose(gtsam.Pose3(gtsam.Rot3.Rodrigues(0.0, 0.0, 0.0), gtsam.Point3(0.0, 0.0, 0.0))))
        current_estimate = initial_estimate
        
        for i in range(loop_start_id, loop_end_id):
            noisy_odometry = noisy_measurements[loop_start_id]
            noisy_tf = gtsam.Pose3(gtsam.Rot3.RzRyRx(noisy_odometry[:3]), noisy_odometry[3:6].reshape(-1,1))   
            graph.push_back(gtsam.BetweenFactorPose3(i+1, i+2, noisy_tf, ODOMETRY_NOISE))
            noisy_estimate = current_estimate.atPose3(i+1).compose(noisy_tf)
            initial_estimate.insert(i + 2, noisy_estimate)
            isam.update(graph, initial_estimate)    
            current_estimate = isam.calculateEstimate()
            initial_estimate.clear()
        
        graph.push_back(gtsam.BetweenFactorPose3(loop_start_id+1, loop_end_id+1, gtsam.Pose3(loop_start2end_c2w), ODOMETRY_NOISE))        
        isam.update(graph, initial_estimate)    
        current_estimate = isam.calculateEstimate()
        
        pose = current_estimate.atPose3(loop_start_id+1)
        pose_matrix = np.identity(4)
        pose_matrix[:3, :3] = pose.rotation().matrix()
        pose_matrix[:3, 3] = pose.translation()
        relative_loop_start_c2w = poses[loop_start_id] @ np.linalg.inv(pose_matrix)
        
        rectified_poses = []
        for kf_idx in range(poses.shape[0]):
            if kf_idx>=loop_start_id and kf_idx<=loop_end_id:
                pose = current_estimate.atPose3(kf_idx+1)
                pose_matrix = np.identity(4)
                pose_matrix[:3, :3] = pose.rotation().matrix()
                pose_matrix[:3, 3] = pose.translation()
                pose_matrix = relative_loop_start_c2w @ pose_matrix
                rectified_poses.append(torch.tensor(pose_matrix, device=poses_device))
            else:
                rectified_poses.append(torch.tensor(poses[kf_idx], device=poses_device))
        '''
        Return Param[1]: rectified_poses就是全部的pose的shape;
        Return Param[2]: optimized_scales的shape是 loopdetect_dict['start_kf_idx']:loopdetect_dict['end_kf_idx']+1
        '''
        return torch.stack(rectified_poses, dim=0).to(torch.float32), None
    
    def rectify_poses_v3(self, poses, loop_start_id, loop_end_id, loop_start2end_c2w):
        poses_device = poses.device
        
        c2w_start2end_numpy = loop_start2end_c2w.cpu().numpy() # (4, 4)
        poses_before_old = poses.cpu().numpy()[loop_start_id:loop_end_id+1]
        w2cs_old         = np.linalg.inv(poses_before_old)
        relative_c2cs    = np.matmul(w2cs_old[1:], poses_before_old[:-1])
        num_inloop_poses = w2cs_old.shape[0]
        def loss_function(scales):
            # 计算当前位姿
            relative_c2cs_copy = relative_c2cs.copy()
            relative_c2cs_copy[:, :3, -1] *= scales.reshape(-1, 1)
            c1_to_ck_new = np.eye(4)
            for i in range(num_inloop_poses-1):
                c1_to_ck_new = relative_c2cs_copy[i] @ c1_to_ck_new
            reconstruction_error = np.sum((c1_to_ck_new[:3, -1] - c2w_start2end_numpy[:3, -1])**2)
            smoothness_penalty   = 0.05 * np.sum(np.diff(scales)**2) # 对相邻尺度差异进行惩罚
            # near_one_error       = np.abs(scales-1).mean()*0.2 # 防止都收敛成小数
            return reconstruction_error + smoothness_penalty # 总损失
        
        initial_scales = np.ones(num_inloop_poses - 1) + np.random.uniform(0.0, 0.5, num_inloop_poses - 1)
        result = minimize(loss_function, initial_scales, bounds=[(0.5, 3.0)] * (num_inloop_poses - 1),  method='L-BFGS-B', options={'maxiter': 5000, 'gtol': 2e-3})
        optimized_scales, error = result.x, result.fun
        c2ws_new = self.forward_new_c2ws(poses_before_old, relative_c2cs, optimized_scales)
        
        # print(f"error: {error}")
        
        rectified_poses  = []
        loop_end_c2w_old = poses_before_old[-1]
        for kf_idx in range(poses.shape[0]):
            if kf_idx>=loop_start_id and kf_idx<=loop_end_id+1:
                rectified_poses.append(torch.tensor(c2ws_new[kf_idx-loop_start_id], device=poses_device))
                if kf_idx == (loop_end_id+1):
                    loop_end_c2w_new = rectified_poses[-1]
            elif kf_idx>=loop_end_id+2:
                rectified_poses.append( np.linalg.inv(np.linalg.inv(poses[kf_idx]) @ loop_end_c2w_old @ np.linalg.inv(loop_end_c2w_new)) )
            else:
                rectified_poses.append(poses[kf_idx])
        '''
        Return Param[1]: rectified_poses就是全部的pose的shape;
        Return Param[2]: optimized_scales的shape是 loopdetect_dict['start_kf_idx']:loopdetect_dict['end_kf_idx']+1
        '''
        # return torch.stack(rectified_poses, dim=0).to(torch.float32), torch.tensor(np.concatenate(([1], optimized_scales)), device=poses.device, dtype=torch.float32).reshape(-1,1,1)
        return torch.stack(rectified_poses, dim=0).to(torch.float32), None 
    
    # TTD 2024/12/11
    def rectify_poses_v4(self, poses, loop_start_id, loop_end_id, loop_start2end_c2w):
        poses_device = poses.device
        
        c2w_start2end_numpy = loop_start2end_c2w.cpu().numpy() # (4, 4)
        poses_before_old = poses.cpu().numpy()[loop_start_id:loop_end_id+1]
        w2cs_old         = np.linalg.inv(poses_before_old)
        relative_c2cs    = np.matmul(w2cs_old[1:], poses_before_old[:-1])
        num_inloop_poses = w2cs_old.shape[0]
        def loss_function(scales):
            # 计算当前位姿
            relative_c2cs_copy = relative_c2cs.copy()
            relative_c2cs_copy[:, :3, -1] *= scales.reshape(-1, 3)
            c1_to_ck_new = np.eye(4)
            for i in range(num_inloop_poses-1):
                c1_to_ck_new = relative_c2cs_copy[i] @ c1_to_ck_new
            reconstruction_error = np.sum((c1_to_ck_new[:3, -1] - c2w_start2end_numpy[:3, -1])**2)
            smoothness_penalty   = 0.01 * np.sum(np.diff(scales.reshape(-1, 3), axis=0)**2) # 对相邻尺度差异进行惩罚
            # near_one_error       = np.abs(scales-1).mean()*0.2 # 防止都收敛成小数
            return reconstruction_error + smoothness_penalty # 总损失
        
        initial_scales = np.ones(3*(num_inloop_poses-1))
        result = minimize(loss_function, initial_scales, bounds=[(0.5, 2.0)]*initial_scales.shape[0],  method='L-BFGS-B', options={'maxiter': 5000, 'gtol': 2e-3})
        optimized_scales, error = result.x, result.fun
        
        c2ws_new = self.forward_new_c2ws_v4(poses_before_old, relative_c2cs, optimized_scales.reshape(-1, 3))
        
        # print(f"error: {error}")
        
        rectified_poses  = []
        loop_end_c2w_old = poses_before_old[-1]
        for kf_idx in range(poses.shape[0]):
            if kf_idx>=loop_start_id and kf_idx<=loop_end_id+1:
                rectified_poses.append(torch.tensor(c2ws_new[kf_idx-loop_start_id], device=poses_device))
                if kf_idx == (loop_end_id+1):
                    loop_end_c2w_new = rectified_poses[-1]
            elif kf_idx>=loop_end_id+2:
                rectified_poses.append( np.linalg.inv(np.linalg.inv(poses[kf_idx]) @ loop_end_c2w_old @ np.linalg.inv(loop_end_c2w_new)) )
            else:
                rectified_poses.append(poses[kf_idx])
        '''
        Return Param[1]: rectified_poses就是全部的pose的shape;
        Return Param[2]: optimized_scales的shape是 loopdetect_dict['start_kf_idx']:loopdetect_dict['end_kf_idx']+1
        '''
        # return torch.stack(rectified_poses, dim=0).to(torch.float32), torch.tensor(np.concatenate(([1], optimized_scales)), device=poses.device, dtype=torch.float32).reshape(-1,1,1)
        return torch.stack(rectified_poses, dim=0).to(torch.float32), None 
    
    
    
    # -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  
    def forward_new_c2ws(self, c2ws_old, relative_c2cs, optimized_scales):
        relative_c2cs_new_copy = relative_c2cs.copy()
        relative_c2cs_new_copy[:, :3, -1] *= optimized_scales.reshape(-1, 1)
        w2cs_old = np.linalg.inv(c2ws_old)
        w2cs_new = [np.linalg.inv(c2ws_old[0])]
        for idx in range(relative_c2cs_new_copy.shape[0]):
            w2cs_new.append(relative_c2cs_new_copy[idx] @ w2cs_new[-1])
        w2cs_new_numpy = np.array(w2cs_new)
        c2ws_new_numpy = np.linalg.inv(w2cs_new_numpy)
        return c2ws_new_numpy
    
    
    def forward_new_c2ws_v4(self, c2ws_old, relative_c2cs, optimized_scales):
        relative_c2cs_new_copy = relative_c2cs.copy()
        relative_c2cs_new_copy[:, :3, -1] *= optimized_scales
        w2cs_old = np.linalg.inv(c2ws_old)
        w2cs_new = [np.linalg.inv(c2ws_old[0])]
        for idx in range(relative_c2cs_new_copy.shape[0]):
            w2cs_new.append(relative_c2cs_new_copy[idx] @ w2cs_new[-1])
        w2cs_new_numpy = np.array(w2cs_new)
        c2ws_new_numpy = np.linalg.inv(w2cs_new_numpy)
        return c2ws_new_numpy
    
    
    # -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  
    def rectify_poses(self, poses, loop_start_id, loop_end_id, loop_start2end_c2w):
        # rectified_pose, optimized_scales = self.rectify_poses_v2(poses, loop_start_id, loop_end_id, loop_start2end_c2w)
        # rectified_poses, optimized_scales = self.rectify_poses_v1(poses, loop_start_id, loop_end_id, loop_start2end_c2w)
        
        # Debug Rectify Poses.
        debug_poses_dict = {'poses': poses, 'loop_start_id': loop_start_id, 'loop_end_id': loop_end_id, 'loop_start2end_c2w': loop_start2end_c2w}
        # torch.save(debug_poses_dict, '/data/wuke/workspace/VINGS-Mono/debug/debug_poses.pth')
        
        rectified_poses, optimized_scales = self.rectify_poses_v4(poses, loop_start_id, loop_end_id, loop_start2end_c2w)
        # rectified_poses, optimized_scales = self.rectify_poses_v1(poses, loop_start_id, loop_end_id, loop_start2end_c2w)
        
        if optimized_scales is None:
            optimized_scales = torch.ones((loop_end_id-loop_start_id+1), device=poses.device, dtype=torch.float32).reshape(-1,1,1)
        
        return rectified_poses, optimized_scales # (N, 4, 4), (N, 1, 1)
    
    
    def rectify_gaussians(self, loopdetect_dict, raw_globalkf_c2ws, new_globalkf_c2ws, new_scales, loop_model, gaussian_model):
        '''
        raw_globalkf_c2ws: (K, 4, 4)
        '''
        # Calculate score and attach a kf_id to each gaussian.
        intrinsic = {'fv': gaussian_model.tfer.fv, 'fu': gaussian_model.tfer.fu, 'cv': gaussian_model.tfer.cv, 'cu': gaussian_model.tfer.cu, 'H': gaussian_model.tfer.H, 'W': gaussian_model.tfer.W}
        kf_id_list = torch.arange(raw_globalkf_c2ws.shape[0], dtype=torch.int32).to(raw_globalkf_c2ws.device)
        
        gaussian_model.setup_optimizer()
        scores, globalkfids = loop_model.calc_score(gaussian_model, intrinsic, raw_globalkf_c2ws, kf_id_list)
        
        # scores = torch.ones_like(gaussian_model._xyz)[:, 0]
        # globalkfids = gaussian_model._globalkf_id
        
        
        globalkfids = globalkfids.to(torch.long)
        globalkfids_first = globalkfids[:,0]
        
        # np.save('/data/wuke/workspace/Droid2DAcc/notebooks/loop/kitti_scores.npy', scores.cpu().numpy())

        # Delete Gaussians whose score = 0.0.
        # update_mask = torch.bitwise_and(scores[:, 0] > 1e-1, torch.bitwise_and(globalkfids[:,0]>=loopdetect_dict['start_kf_idx'], globalkfids[:,0]<=loopdetect_dict['end_kf_idx']))
        update_mask = torch.bitwise_and(globalkfids_first>=loopdetect_dict['start_kf_idx'], globalkfids_first<=loopdetect_dict['end_kf_idx'])
        globalkfids_first_update = globalkfids_first[update_mask]
        
        delete_mask = scores[:, 0] < 0.1
        
        raw_xyz      = gaussian_model._xyz[update_mask] + 0.0
        # raw_rgb      = gaussian_model._rgb[update_mask] + 0.0
        # raw_scaling  = gaussian_model._scaling[update_mask] + 0.0
        raw_rotation = gaussian_model._rotation[update_mask] + 0.0
        # raw_opacity  = gaussian_model._opacity[update_mask] + 0.0
        
        # gaussian_model._global_scores = gaussian_model._global_scores[valid_mask]
        # gaussian_model._local_scores  = gaussian_model._local_scores[valid_mask]
        # gaussian_model._stable_mask   = gaussian_model._stable_mask[valid_mask]

        # TTD 2024/10/09
        transforms_perpose = torch.bmm(new_globalkf_c2ws, torch.linalg.inv(raw_globalkf_c2ws)) # (F, 4, 4)
        # # transforms_perpose[loopdetect_dict['start_kf_idx']:loopdetect_dict['end_kf_idx']+1] *= new_scales
        transforms_pergaussian = transforms_perpose[globalkfids_first_update] # (P, 4, 4)
        
        # new_xyz_world = torch.bmm(transforms_pergaussian[:, :3, :3], raw_xyz.unsqueeze(-1)).squeeze(-1) # (P, 3)
        # new_xyz_world += transforms_pergaussian[:, :3, 3] # (P, 3)
        
        # -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
        # TTD 2024/11/09
        # Transform gaussians to relative cam coordinates.
        new_scales_update   = new_scales[globalkfids_first_update-loopdetect_dict['start_kf_idx'], ...].reshape(-1, 1)
        
        raw_globalkf_w2cs   = torch.linalg.inv(raw_globalkf_c2ws) # (K, 4, 4)
        old_w2c_pergaussian = raw_globalkf_w2cs[globalkfids_first_update] # (p, 4, 4), p < P
        old_c2g_xyz         = torch.matmul(old_w2c_pergaussian[:, :3, :3], raw_xyz.unsqueeze(-1)).squeeze(-1) + old_w2c_pergaussian[:, :3, 3] # (p, 3)
        new_c2g_xyz         = old_c2g_xyz.detach() * new_scales_update # (P, 3)
        new_xyz_world       = torch.matmul(new_globalkf_c2ws[globalkfids_first_update, :3, :3], new_c2g_xyz.unsqueeze(-1)).squeeze(-1) + new_globalkf_c2ws[globalkfids_first_update, :3, 3]
        # -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
        
        
        # Get new_xyz and new_rotation.
        raw_R         = q2R(raw_rotation)
        new_rotation  = R2q(torch.bmm(transforms_pergaussian[:, :3, :3], raw_R))
        
        # Recify Gaussians' property.
        with torch.no_grad():
            gaussian_model._xyz[update_mask]      *= 0
            gaussian_model._xyz[update_mask]      += new_xyz_world
            gaussian_model._rotation[update_mask] *= 0
            gaussian_model._rotation[update_mask] += new_rotation
        
        # TTD 2024/11/09
        gaussian_model._xyz = gaussian_model._xyz[~delete_mask].detach().requires_grad_(True)
        gaussian_model._rgb = gaussian_model._rgb[~delete_mask].detach().requires_grad_(True)
        gaussian_model._opacity = gaussian_model._opacity[~delete_mask].detach().requires_grad_(True)
        gaussian_model._scaling = gaussian_model._scaling[~delete_mask].detach().requires_grad_(True)
        gaussian_model._rotation = gaussian_model._rotation[~delete_mask].detach().requires_grad_(True)
        
        gaussian_model._global_scores = gaussian_model._global_scores[~delete_mask]
        gaussian_model._local_scores = gaussian_model._local_scores[~delete_mask]
        gaussian_model._stable_mask = gaussian_model._stable_mask[~delete_mask]
        gaussian_model._globalkf_id = gaussian_model._globalkf_id[~delete_mask]
        gaussian_model._globalkf_max_scores = gaussian_model._globalkf_max_scores[~delete_mask]
        if hasattr(gaussian_model, "_birth_globalkf_id"):
            if gaussian_model._birth_globalkf_id.shape[0] == delete_mask.shape[0]:
                gaussian_model._birth_globalkf_id = gaussian_model._birth_globalkf_id[~delete_mask]
            else:
                gaussian_model._birth_globalkf_id = gaussian_model._globalkf_id.detach().clone()
        
        gaussian_model.setup_optimizer()
        
        torch.cuda.empty_cache()
        
        # TODO-wuke: Run several iters on (loopstart_idx, loopend_idx).
    
    def rectify_tracker_v1(self, mapper, tracker, loop_start_id, loop_end_id, recitified_history_kf_c2w):
        with torch.no_grad():
            intrinsic = {'fv': mapper.tfer.fv, 'fu': mapper.tfer.fu, 'cv': mapper.tfer.cv, 'cu': mapper.tfer.cu, 'H': mapper.tfer.H, 'W': mapper.tfer.W}
            for globalkf_idx in range(loop_start_id, loop_end_id+1):
                rectified_w2c  = torch.linalg.inv(recitified_history_kf_c2w[globalkf_idx])
                rendered_depth = mapper.render(rectified_w2c, intrinsic)['depth'].squeeze(0) # (H, W)            
                # Rectify tracker.frontend.video.poses_save, disps_save, disps_up_save.
                tracker.frontend.video.poses_save[globalkf_idx]    = matrix_to_tq(rectified_w2c.unsqueeze(0)).cpu().squeeze(0)
                update_mask = rendered_depth.cpu() > 0
                tracker.frontend.video.disps_up_save[globalkf_idx, update_mask] = rendered_depth.cpu()[update_mask]
                H, W = rendered_depth.shape
                tracker.frontend.video.disps_save[globalkf_idx] = torch.mean(((tracker.frontend.video.disps_up_save[globalkf_idx].reshape(H//8, 8, W//8, 8).permute(0, 2, 1, 3))[..., 3:5, 3:5]).reshape(H//8, W//8, -1), dim=-1) # (H//8, W//8)
            
            # Rectify tracker.frontend.video.poses, disps, disps_up. 
            local_start_id = max(0, loop_start_id - tracker.local_to_global_bias)
            local_end_id   = loop_end_id - tracker.local_to_global_bias
            DEVICE = tracker.frontend.video.poses.device
            for localkf_idx in range(local_start_id, local_end_id):
                globalkf_idx = localkf_idx + tracker.local_to_global_bias
                tracker.frontend.video.poses[localkf_idx]    = tracker.frontend.video.poses_save[globalkf_idx].to(DEVICE)
                tracker.frontend.video.disps_up[localkf_idx] = tracker.frontend.video.disps_up_save[globalkf_idx].to(DEVICE)
                tracker.frontend.video.disps[localkf_idx]    = tracker.frontend.video.disps_save[globalkf_idx].to(DEVICE)
            
            # Recitify upper parts poses.
            # 
    
    
    def rectify_tracker(self, mapper, tracker, loop_start_id, loop_end_id, recitified_history_kf_c2w):
        with torch.no_grad():
            intrinsic = {'fv': mapper.tfer.fv, 'fu': mapper.tfer.fu, 'cv': mapper.tfer.cv, 'cu': mapper.tfer.cu, 'H': mapper.tfer.H, 'W': mapper.tfer.W}
            for globalkf_idx in range(loop_start_id, loop_end_id+1):
                rectified_w2c  = torch.linalg.inv(recitified_history_kf_c2w[globalkf_idx])
                rendered_depth = mapper.render(rectified_w2c, intrinsic)['depth'].squeeze(0) # (H, W)            
                # Rectify tracker.frontend.video.poses_save, disps_save, disps_up_save.
                tracker.frontend.video.poses_save[globalkf_idx]    = matrix_to_tq(rectified_w2c.unsqueeze(0)).cpu().squeeze(0)
                update_mask = rendered_depth.cpu() > 0
                tracker.frontend.video.disps_up_save[globalkf_idx, update_mask] = rendered_depth.cpu()[update_mask]
                H, W = rendered_depth.shape
                tracker.frontend.video.disps_save[globalkf_idx] = torch.mean(((tracker.frontend.video.disps_up_save[globalkf_idx].reshape(H//8, 8, W//8, 8).permute(0, 2, 1, 3))[..., 3:5, 3:5]).reshape(H//8, W//8, -1), dim=-1) # (H//8, W//8)
            
            # Rectify tracker.frontend.video.poses, disps, disps_up. 
            if hasattr(tracker, 'local_to_global_bias') and tracker.local_to_global_bias > 10:
                local_start_id = max(0, loop_start_id - tracker.local_to_global_bias)
                local_end_id   = loop_end_id - tracker.local_to_global_bias
                DEVICE = tracker.frontend.video.poses.device
                for localkf_idx in range(local_start_id, local_end_id):
                    globalkf_idx = localkf_idx + tracker.local_to_global_bias
                    tracker.frontend.video.poses[localkf_idx]    = tracker.frontend.video.poses_save[globalkf_idx].to(DEVICE)
                    tracker.frontend.video.disps_up[localkf_idx] = tracker.frontend.video.disps_up_save[globalkf_idx].to(DEVICE)
                    tracker.frontend.video.disps[localkf_idx]    = tracker.frontend.video.disps_save[globalkf_idx].to(DEVICE)
            
    
    
    # TTD 2024/11/06
    def rectify_tracker_nerfslam(self, mapper, tracker, loop_start_id, loop_end_id, recitified_history_kf_c2w):
        with torch.no_grad():
            intrinsic = {'fv': mapper.tfer.fv, 'fu': mapper.tfer.fu, 'cv': mapper.tfer.cv, 'cu': mapper.tfer.cu, 'H': mapper.tfer.H, 'W': mapper.tfer.W}
            for globalkf_idx in range(loop_start_id, loop_end_id+1):
                rectified_w2c  = torch.linalg.inv(recitified_history_kf_c2w[globalkf_idx])
                rendered_depth = mapper.render(rectified_w2c, intrinsic)['depth'].squeeze(0) # (H, W)            
                # Rectify tracker.frontend.video.poses_save, disps_save, disps_up_save.
                tracker.visual_frontend.cam0_T_world[globalkf_idx]    = matrix_to_tq(rectified_w2c.unsqueeze(0)).cpu().squeeze(0)
                # Attention! c2i is eye matrix👀.
                tracker.visual_frontend.world_T_body[globalkf_idx]    = matrix_to_tq(torch.linalg.inv(rectified_w2c).unsqueeze(0)).cpu().squeeze(0)
                
                CORRECT_DEPTHS = False
                if CORRECT_DEPTHS:
                    update_mask = rendered_depth > 0
                    tracker.visual_frontend.cam0_idepths_up[globalkf_idx, update_mask] = rendered_depth[update_mask]
                    H, W = rendered_depth.shape
                    tracker.visual_frontend.cam0_idepths[globalkf_idx] = torch.mean(((tracker.visual_frontend.cam0_idepths_up[globalkf_idx].reshape(H//8, 8, W//8, 8).permute(0, 2, 1, 3))[..., 3:5, 3:5]).reshape(H//8, W//8, -1), dim=-1) # (H//8, W//8)
            
            
    
    # TTD 2024/10/18
    def retrain_gaussian(self, mapper, tracker, loop_start_id, loop_end_id):
        '''
        Iterative on tracker.frontend.video.poses_save, disps_up_save, images_up_save.
            - (1) Use invariant depth loss.
            - (2) Run this funtion after looper.rectify_tracker.
            - (3) Maybe we should add storage control here?
        '''
        loop_start_id = 0
        # STEP 1 Prepare processed_dict. (Check shape)
        processed_dict = {}
        if not self.cfg['mode'] == 'vo_nerfslam':
            processed_dict["poses"]      = SE3(tracker.frontend.video.poses_save[loop_start_id:loop_end_id+1]).inv().matrix().cuda()
            processed_dict["images"]     = tracker.frontend.video.images_up_save[loop_start_id:loop_end_id+1].cuda()[...,[2,1,0]]
            processed_dict["depths"]     = 1/(tracker.frontend.video.disps_up_save[loop_start_id:loop_end_id+1].cuda()+1e-4).cuda().unsqueeze(-1)
            processed_dict["depths_cov"] = tracker.frontend.video.depths_cov_up_save[loop_start_id:loop_end_id+1].cuda().unsqueeze(-1)
        else:
            processed_dict["poses"]      = SE3(tracker.visual_frontend.cam0_T_world[loop_start_id:loop_end_id+1]).inv().matrix().cuda()
            processed_dict["images"]     = tracker.visual_frontend.cam0_images[loop_start_id:loop_end_id+1].cuda().permute(0,2,3,1)[...,[2,1,0]] / 255.0
            processed_dict["depths"]     = 1/(tracker.visual_frontend.cam0_idepths_up[loop_start_id:loop_end_id+1].cuda()+1e-4).cuda().unsqueeze(-1)
            processed_dict["depths_cov"] = tracker.visual_frontend.cam0_depths_cov_up[loop_start_id:loop_end_id+1].cuda().unsqueeze(-1)
        #   cov_median = torch.tensor(np.median(depths_cov.cpu().numpy().reshape(N_frames, -1), axis=1)[:, None, None, None], device=depths.device) # (N, 1, 1)
        #   zero_mask = torch.bitwise_or(processed_dict["depths"] > mapper.cfg['middleware']['max_depth'], processed_dict["depths_cov"]>mapper.cfg['middleware']['cov_times']*(cov_median))
        processed_dict["depths"][processed_dict["depths"] > mapper.cfg['middleware']['max_depth']] = 0
        processed_dict["intrinsic"]  = {'fu': mapper.tfer.fu, 'fv': mapper.tfer.fv, 'cu': mapper.tfer.cu, 'cv': mapper.tfer.cv, 'H': mapper.tfer.H, 'W': mapper.tfer.W}
        
        # STEP 2 
        batch = processed_dict
        poses              = batch["poses"]                   # (N, 4, 4)
        images             = batch["images"]                  # (N, 344, 616, 3)
        depths             = batch["depths"]                  # (N, 344, 616, 1)
        depths_cov         = batch["depths_cov"]              # (N, 344, 616, 1) 
        intrinsic_dict     = batch["intrinsic"]               # {'fu', 'fv', 'cu', 'cv', 'H', 'W'}
        
        train_iters = (loop_end_id - loop_start_id) * 2
        for curr_iter in range(train_iters):
            curr_id = random.randint(0, poses.shape[0]-1)
            c2w = poses[curr_id]
            w2c = torch.linalg.inv(c2w)
            
            pred_dict = mapper.render(w2c, intrinsic_dict, None, w2c2=torch.linalg.inv(poses[min(curr_id+1, poses.shape[0]-1)]))
            gt_dict = {'rgb': images[curr_id].permute(2,0,1), 'depth': depths[curr_id].permute(2,0,1), 'uncert': depths_cov[curr_id].permute(2,0,1), 'c2w': c2w}
            gt_dict['depth_cov'] = depths_cov[curr_id].permute(2,0,1)
            
            # pred_dict['time_idx'] = self.time_idx
            new_cfg = copy.deepcopy(self.cfg)  
            new_cfg['training_args']['loss_weights']['depth_loss'] = 0.0  
            total_loss = get_loss(new_cfg, pred_dict, gt_dict)
            total_loss.backward()
        
        # Storage Control.
        # temp_importance_scores = torch.zeros_like(mapper._local_scores[:, 0]) # (P, )
        # intrinsic_dict = batch["intrinsic"]
        # for kf_idx in range(batch["poses"].shape[0]):
        #     c2w, gt_rgb = batch["poses"][kf_idx], batch["images"][kf_idx].permute(2, 0, 1) # (4, 4), (3, H, W)
        #     pred_rgb = mapper.render(torch.linalg.inv(c2w), intrinsic_dict)['rgb']
        #     (torch.abs(pred_rgb-gt_rgb)[:, gt_rgb.sum(axis=0)>0]).mean().backward()
        #     temp_importance_scores += mapper._zeros.grad.detach()[:, 0]
        #     mapper.optimizer.zero_grad()
        #     mapper._zeros.grad.zero_()
        # # prune_gaussianmask = (temp_importance_scores > 0.1) & (~self._stable_mask) & (temp_importance_scores < 0.8)    
        # prune_gaussianmask = (temp_importance_scores < 2.0)
        # new_dict = mapper.prune_tensors_from_optimizer(mapper.optimizer, prune_gaussianmask)
        # mapper.update_properties(new_dict)
        # mapper.update_records(mode="prune", prune_gaussianmask=prune_gaussianmask)
        
        
        # Retrain.
        train_iters = (loop_end_id - loop_start_id) * 2
        for curr_iter in range(train_iters):
            curr_id = random.randint(0, poses.shape[0]-1)
            c2w = poses[curr_id]
            w2c = torch.linalg.inv(c2w)
            
            pred_dict = mapper.render(w2c, intrinsic_dict, None, w2c2=torch.linalg.inv(poses[min(curr_id+1, poses.shape[0]-1)]))
            gt_dict = {'rgb': images[curr_id].permute(2,0,1), 'depth': depths[curr_id].permute(2,0,1), 'uncert': depths_cov[curr_id].permute(2,0,1), 'c2w': c2w}
            gt_dict['depth_cov'] = depths_cov[curr_id].permute(2,0,1)
            
            # pred_dict['time_idx'] = self.time_idx
            new_cfg = copy.deepcopy(self.cfg)  
            new_cfg['training_args']['loss_weights']['depth_loss'] = 0.0
            total_loss = get_loss(new_cfg, pred_dict, gt_dict)
            total_loss.backward()
        
        
        
        torch.cuda.empty_cache()
        
        
    # TTD 2024/10/18
    def rectify_gaussians_storage_control(self):
        pass
    
    
    def retrain_gaussian_storage_control(self):
        pass
