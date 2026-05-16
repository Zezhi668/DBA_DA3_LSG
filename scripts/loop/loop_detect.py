import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import cv2
import time
import kornia
# from LightGlue.lightglue import LightGlue, SuperPoint, match_pair
# from LightGlue.lightglue.utils import load_image
# import sys
# from onnx_runner import LightGlueRunner
from loop.lightglue import LightGlueRunner
from gaussian.loss_utils import ssim_loss
from vings_utils.sim3_utils import ransac_umeyama


class LoopDetector:
    def __init__(self, cfg):
        # -  -  -  -  -  -  -  -  -  -  -  -  -  -
        self.cfg = cfg
        ONNX_W = 512
        H, W = cfg['frontend']['image_size'][0], cfg['frontend']['image_size'][1]
        WEIGHT_DIR = cfg['looper']['lightglue_weight_dir'] # '/data/wuke/workspace/LightGlue-ONNX/weights/'
        # -  -  -  -  -  -  -  -  -  -  -  -  -  -

        # Light glue declarations:
        # torch.set_grad_enabled(False)
        self.device = torch.device(cfg['device']['mapper'])
        
        # providers = ["CPUExecutionProvider"]
        providers = ["CUDAExecutionProvider"]
        # providers = [("TensorrtExecutionProvider", {"trt_fp16_enable": True, "trt_engine_cache_enable": True, "trt_engine_cache_path": "weights/cache"})]
        self.matcher = LightGlueRunner(
            extractor_path=f"{WEIGHT_DIR}/superpoint.onnx",
            lightglue_path=f"{WEIGHT_DIR}/superpoint_lightglue.onnx",
            providers=providers,
            # TensorrtExecutionProvider, OpenVINOExecutionProvider
        )

        NEW_H, NEW_W = int(ONNX_W/W*H), ONNX_W
        # HW.
        self.img_transform_pipeline = transforms.Compose([
            transforms.Resize((NEW_H, NEW_W)),  # 调整图片大小
            transforms.Grayscale(num_output_channels=1),  # 转换为灰度图
        ])
        # vu.
        self.img_scales = np.array([ONNX_W/W, int(ONNX_W/W*H)/H])

    def _min_depth(self):
        return float(self.cfg.get('looper', {}).get('min_depth', 0.2))

    def _max_depth(self):
        return float(self.cfg.get('looper', {}).get('max_depth', 15.0))

    def _accum_threshold(self):
        return float(self.cfg.get('looper', {}).get('accum_threshold', 0.95))

    def _pose_jump_threshold(self):
        return float(
            self.cfg.get('looper', {}).get(
                'pose_jump_threshold',
                self.cfg.get('submap', {}).get('pose_jump_threshold', 15.0),
            )
        )

    @staticmethod
    def _form_transf(R, t):
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t.reshape(-1)
        return T

    def get_matches(self, img1, img2):
        '''
        img1.shape = (3, 375, 1242) or (3, 344, 616)
        '''
        image0, image1 = img1.unsqueeze(0), img2.unsqueeze(0)
        image0, image1 = self.img_transform_pipeline(image0), self.img_transform_pipeline(image1)
        
        image0, image1 = image0.cpu().numpy(), image1.cpu().numpy()
        q1, q2 = self.matcher.run(image0, image1, self.img_scales, self.img_scales) # 400ms
        return q1,q2
    
    def get_pose(self, c2w1, gt_depth1, vu_kp1, vu_kp2, tfer):
        '''
        Attention, we should pass vu here, there is no uv.
        '''
        # Get kp1's world points.
        min_depth = self._min_depth()
        max_depth = self._max_depth()
        valid_depth = gt_depth1.squeeze(0)[vu_kp1[:,1], vu_kp1[:,0]]
        valid_mask = torch.bitwise_and(valid_depth > min_depth, valid_depth < max_depth) # (N, )
        if valid_mask.sum() < 10: return None 
        uv_depth = torch.stack([vu_kp1[valid_mask,1], vu_kp1[valid_mask,0], gt_depth1.squeeze(0)[vu_kp1[valid_mask,1], vu_kp1[valid_mask,0]]], dim=-1)
        world_points1 = tfer.transform(uv_depth, 'pixel', 'world', pose=c2w1).unsqueeze(0) # (1, N, 3)
        img_points2   = vu_kp2[valid_mask].unsqueeze(0) # (1, N, 2)
        intrinsics    = torch.tensor([[tfer.fv, 0, tfer.cv], [0, tfer.fu, tfer.cu], [0, 0, 1]], dtype=torch.float32, device=c2w1.device).unsqueeze(0) # (1, 3, 3)
        
        # PnP.
        # pred_w2c2 = kornia.geometry.solve_pnp_dlt(world_points1, img_points2.to(torch.float32), intrinsics).squeeze(0)        
        # pred_w2c2 = torch.concat([pred_w2c2, torch.tensor([0, 0, 0, 1], device=pred_w2c2.device, dtype=torch.float32).unsqueeze(0)], dim=0) # (4, 4)
        
        # torch.save({'world_points1': world_points1, 'c2w1': c2w1, ''})
        
        # Test opencv.
        dist_coeffs = np.float32([0, 0, 0, 0]).reshape(4, 1) 
        
        # assert img_points2.shape[0] > 8, "Too few points to solve PnP."
        success, rotation_vector, translation_vector = cv2.solvePnP(np.float32(world_points1.cpu().squeeze(0)), np.float32(img_points2.squeeze(0).cpu()), np.float32(intrinsics.squeeze(0).cpu()), dist_coeffs)
            
        if success:
            R, _ = cv2.Rodrigues(rotation_vector)
            pred_w2c2 = torch.tensor(self._form_transf(R, translation_vector), dtype=torch.float32, device=c2w1.device)
        else:
            pred_w2c2 = None
        return pred_w2c2

    def get_sim3(self, depth_ref, depth_query, vu_ref, vu_query, tfer):
        '''
        Estimate query_from_ref as a 3D-3D Sim3 using matched depth on both sides.
        All inputs are in the repo's (v, u) convention.
        '''
        depth_ref = depth_ref.squeeze()
        depth_query = depth_query.squeeze()
        if depth_ref.ndim != 2 or depth_query.ndim != 2:
            return None

        z_ref = depth_ref[vu_ref[:, 1], vu_ref[:, 0]]
        z_query = depth_query[vu_query[:, 1], vu_query[:, 0]]
        min_depth = self._min_depth()
        max_depth = self._max_depth()
        valid_mask = (
            torch.isfinite(z_ref)
            & torch.isfinite(z_query)
            & (z_ref > min_depth)
            & (z_query > min_depth)
            & (z_ref < max_depth)
            & (z_query < max_depth)
        )

        min_matches = int(
            self.cfg.get('submap', {}).get(
                'validation_min_matches',
                self.cfg['looper']['is_loop_min_match_num'],
            )
        )
        if valid_mask.sum().item() < min_matches:
            return None

        ref_uvd = torch.stack(
            [vu_ref[valid_mask, 1].float(), vu_ref[valid_mask, 0].float(), z_ref[valid_mask].float()],
            dim=-1,
        )
        query_uvd = torch.stack(
            [vu_query[valid_mask, 1].float(), vu_query[valid_mask, 0].float(), z_query[valid_mask].float()],
            dim=-1,
        )

        ref_pts_cam = tfer.transform(ref_uvd, 'pixel', 'cam').detach().cpu().numpy()
        query_pts_cam = tfer.transform(query_uvd, 'pixel', 'cam').detach().cpu().numpy()
        sim3_query_from_ref, inliers = ransac_umeyama(
            ref_pts_cam,
            query_pts_cam,
            iters=int(self.cfg.get('sim3_ransac_iters', 256)),
            sample_size=int(self.cfg.get('sim3_ransac_sample_size', 4)),
            min_inliers=int(self.cfg.get('sim3_min_inliers', min_matches)),
            inlier_threshold=float(self.cfg.get('sim3_inlier_threshold', 0.25)),
        )
        if sim3_query_from_ref is None or inliers is None:
            return None

        return {
            'R': sim3_query_from_ref.R,
            't': sim3_query_from_ref.t,
            's': float(sim3_query_from_ref.s),
            'inlier_mask': inliers,
            'match_count': int(inliers.sum()),
        }

    @staticmethod
    def dilation(input_image, kernel_size=50):
        '''
        input_image: (H, W)
        '''
        H, W = input_image.shape[0], input_image.shape[1]
        kernel = torch.ones((1, 1, kernel_size, kernel_size), dtype=torch.float32, device=input_image.device)
        input_image = input_image.unsqueeze(0).unsqueeze(0)
        dilated_image = F.conv2d(input_image, kernel, padding=kernel_size//2)
        dilated_image = dilated_image.squeeze(0).squeeze(0)[:H, :W]
        return dilated_image
    
    def get_kp_mask(self, kp_vu, gt_img):
        '''
        Shall we test mask by accum to juedge where has been sufficently optimized.
        kp_vu:     (N, 2)
        gt_img:    (3, H, W)
        '''
        kp_mask = torch.zeros_like(gt_img[0])
        kp_mask[kp_vu[:,1], kp_vu[:,0]] = 1.0
        kp_mask = self.dilation(torch.tensor(kp_mask, device=gt_img.device, dtype=torch.float32)) > 0
        return kp_mask # (H, W), bool

    def detect_loop(self, gt_img1, gt_img2, c2w2, gt_depth2, gaussian_model, debug=False):
        '''
          gt_img1: (3, H, W)
             c2w1: (4, 4)
        gt_depth2: (1, H, W), but we suggest always None.
        '''
        # Get Relative Pose.
        vu_kp1, vu_kp2 = self.get_matches(gt_img1, gt_img2)
        vu_kp1, vu_kp2 = torch.tensor(vu_kp1, device=c2w2.device, dtype=torch.long), torch.tensor(vu_kp2, device=c2w2.device, dtype=torch.long)
        
        if vu_kp1.shape[0] < self.cfg['looper']['is_loop_min_match_num']:
            return None, 1.0
        
        # Render Image and cal L1Loss between img2.
        intrinsic_dict = {'fu': gaussian_model.tfer.fu, 'fv': gaussian_model.tfer.fv, 'cu': gaussian_model.tfer.cu, 'cv': gaussian_model.tfer.cv, 'H': gaussian_model.tfer.H, 'W': gaussian_model.tfer.W}
        if gt_depth2 is None:
            with torch.no_grad():
                gt_depth2 = gaussian_model.render(torch.linalg.inv(c2w2), intrinsic_dict)['depth']
                # gt_depth2[gt_depth2>torch.median(gt_depth2)*10] = 0.0
                gt_depth2[gt_depth2 > self._max_depth()] = 0.0
                
        if (gt_depth2.squeeze(0)[vu_kp2[:,1], vu_kp2[:,0]] > 0.0).sum() < 10:
            return None, 1.0 # torch.tensor(1.0, device=c2w2.device)
        
        # w2c2 = self.get_pose(c2w1, gt_depth1, vu_kp1, vu_kp2, gaussian_model.tfer).squeeze(0)
        w2c1 = self.get_pose(c2w2, gt_depth2, vu_kp2, vu_kp1, gaussian_model.tfer)
        if w2c1 is None: 
            return None, 1.0 
        else:
            w2c1 = w2c1.squeeze(0)
        
        # TTD 2024/12/12
        # Dangerous Option.
        if torch.linalg.norm(torch.linalg.inv(w2c1)[:3, 3] - c2w2[:3, 3]) > self._pose_jump_threshold():
            return None, 1.0
        
        with torch.no_grad():
            pred_dict   = gaussian_model.render(w2c1, intrinsic_dict)
            # pred_dict   = gaussian_model.render_indistance(w2c1, intrinsic_dict)
            pred_img1   = pred_dict['rgb'][[0,1,2], ...]
            pred_depth1 = pred_dict['depth']
            pred_accum1 = pred_dict['accum'].squeeze(0)
        
        # kp_mask    = self.get_kp_mask(vu_kp2, pred_img2)
        # Calculate relative L1Loss to judge wether detect a loop, or we can use neightbour frames to set the threshold.
        # mask_error = (pred_img2[:, kp_mask] - gt_img2[:, kp_mask]).abs().mean() # (3, H, W)
        valid_mask = torch.bitwise_and(
            pred_accum1 > self._accum_threshold(),
            pred_depth1.squeeze(0) < self._max_depth(),
        )
        # valid_mask = gt_img1.sum(axis=0) > 0.0
        
        
        l1_error = (pred_img1 - gt_img1)[:, valid_mask].abs().mean() # (3, H, W)
        
        # if l1_error < 0.25:
        #    torch.save({'pred_img': pred_img1, 'gt_img': gt_img1}, '/data/wuke/workspace/VINGS-Mono/debug/compare_LC/loop_img.pt')
        
        
        # 这个损失要不用带SSIM的损失，我现在觉得是L1 Loss太der啦;
        # ssim_error = 1.0 - ssim_loss(pred_img1, gt_img1, valid_mask)
        
        # mask_error = 0.8 * l1_error + 0.2 * ssim_error
        mask_error = l1_error 
        
        # torch.save({'gt_depth2': gt_depth2, 'gt_img2': gt_img2, 'pred_img1': pred_img1, 'gt_img1': gt_img1}, "/data/wuke/workspace/VINGS-Mono/debug/loop_img.pt")
        # Use for debug.
        if debug:
            byproduct_dict = {'vu_kp1': vu_kp1, 'vu_kp2': vu_kp2, 'gt_depth2': gt_depth2, 'kp_mask': None, 'mask_error': mask_error, 'c2w1': torch.linalg.inv(w2c1)}
            return pred_img1, byproduct_dict
        # torch.save({'vu_kp1': vu_kp1, 'vu_kp2': vu_kp2, 'c2w1': torch.linalg.inv(w2c1), 'c2w2': c2w2, 'gt_depth2': gt_depth2, 'gt_img2': gt_img2, 'pred_img1': pred_img1, 'gt_img1': gt_img1}, "/data/wuke/workspace/VINGS-Mono/debug/loop_img.pt")
        
        # print(f"Number of Match: {vu_kp1.shape[0]}, mask_error: {mask_error}")        
        return pred_img1, mask_error.item()
    

    def detect_loop_compare(self, gt_img1, gt_img2, c2w2, gt_depth2, gaussian_model, debug=False):
        '''
          gt_img1: (3, H, W)
             c2w1: (4, 4)
        gt_depth2: (1, H, W), but we suggest always None.
        '''
        # Get Relative Pose.
        vu_kp1, vu_kp2 = self.get_matches(gt_img1, gt_img2)
        vu_kp1, vu_kp2 = torch.tensor(vu_kp1, device=c2w2.device, dtype=torch.long), torch.tensor(vu_kp2, device=c2w2.device, dtype=torch.long)
        
        if vu_kp1.shape[0] < self.cfg['looper']['is_loop_min_match_num']:
            return None, 1.0, 1.0, None, 1e3
        
        # Render Image and cal L1Loss between img2.
        intrinsic_dict = {'fu': gaussian_model.tfer.fu, 'fv': gaussian_model.tfer.fv, 'cu': gaussian_model.tfer.cu, 'cv': gaussian_model.tfer.cv, 'H': gaussian_model.tfer.H, 'W': gaussian_model.tfer.W}
        if gt_depth2 is None:
            with torch.no_grad():
                gt_depth2 = gaussian_model.render(torch.linalg.inv(c2w2), intrinsic_dict)['depth']
                # gt_depth2[gt_depth2>torch.median(gt_depth2)*10] = 0.0
                gt_depth2[gt_depth2 > self._max_depth()] = 0.0
                
        if (gt_depth2.squeeze(0)[vu_kp2[:,1], vu_kp2[:,0]] > 0.0).sum() < 12:
            return None, 1.0, 1.0, None, 1e3 # torch.tensor(1.0, device=c2w2.device)
        
        # w2c2 = self.get_pose(c2w1, gt_depth1, vu_kp1, vu_kp2, gaussian_model.tfer).squeeze(0)
        w2c1 = self.get_pose(c2w2, gt_depth2, vu_kp2, vu_kp1, gaussian_model.tfer)
        if w2c1 is None: 
            return None, 1.0, 1.0, None, 1e3 
        else:
            w2c1 = w2c1.squeeze(0)
        
        # TTD 2024/12/12
        # Dangerous Option.
        distance = torch.linalg.norm(torch.linalg.inv(w2c1)[:3, 3] - c2w2[:3, 3])
        # if distance > 30.0:
        #    return None, 1.0, 1e3
        
        with torch.no_grad():
            pred_dict   = gaussian_model.render(w2c1, intrinsic_dict)
            # pred_dict   = gaussian_model.render_indistance(w2c1, intrinsic_dict)
            pred_img1   = pred_dict['rgb'][[0,1,2], ...]
            pred_depth1 = pred_dict['depth']
            pred_accum1 = pred_dict['accum'].squeeze(0)
        
        # kp_mask    = self.get_kp_mask(vu_kp2, pred_img2)
        # Calculate relative L1Loss to judge wether detect a loop, or we can use neightbour frames to set the threshold.
        # mask_error = (pred_img2[:, kp_mask] - gt_img2[:, kp_mask]).abs().mean() # (3, H, W)
        valid_mask = torch.bitwise_and(
            pred_accum1 > self._accum_threshold(),
            pred_depth1.squeeze(0) < self._max_depth(),
        )
        if valid_mask.sum() < 0.25 * valid_mask.numel(): 
            return None, 1.0, 1.0, None, 1e3
        # valid_mask = gt_img1.sum(axis=0) > 0.0
        
        
        l1_error = (pred_img1 - gt_img1)[:, valid_mask].abs().mean() # (3, H, W)
        
        # if l1_error < 0.25:
        #    torch.save({'pred_img': pred_img1, 'gt_img': gt_img1}, '/data/wuke/workspace/VINGS-Mono/debug/compare_LC/loop_img.pt')
        
        
        # 这个损失要不用带SSIM的损失，我现在觉得是L1 Loss太der啦;
        ssim_error = 1.0 - ssim_loss(pred_img1, gt_img1, valid_mask)
        
        # mask_error = 0.8 * l1_error + 0.2 * ssim_error
        # mask_error = l1_error 
        
        # torch.save({'gt_depth2': gt_depth2, 'gt_img2': gt_img2, 'pred_img1': pred_img1, 'gt_img1': gt_img1}, "/data/wuke/workspace/VINGS-Mono/debug/loop_img.pt")
        # Use for debug.
        # torch.save({'vu_kp1': vu_kp1, 'vu_kp2': vu_kp2, 'c2w1': torch.linalg.inv(w2c1), 'c2w2': c2w2, 'gt_depth2': gt_depth2, 'gt_img2': gt_img2, 'pred_img1': pred_img1, 'gt_img1': gt_img1}, "/data/wuke/workspace/VINGS-Mono/debug/loop_img.pt")
        
        # print(f"Number of Match: {vu_kp1.shape[0]}, mask_error: {mask_error}")        
        return pred_img1, l1_error.item(), ssim_error.item(), valid_mask, distance
    
