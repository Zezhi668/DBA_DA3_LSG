import numpy as np
import shutil
import os
import argparse
import sys


SCRIPT_ROOT = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_ROOT not in sys.path:
    sys.path.insert(0, SCRIPT_ROOT)

REPO_ROOT = os.path.dirname(SCRIPT_ROOT)
GTSAM_PYTHON_ROOT = os.path.join(REPO_ROOT, "submodules", "gtsam", "build", "python")
if os.path.isdir(GTSAM_PYTHON_ROOT) and GTSAM_PYTHON_ROOT not in sys.path:
    sys.path.insert(0, GTSAM_PYTHON_ROOT)


def _major_version(version_str):
    try:
        return int(version_str.split(".", 1)[0])
    except (AttributeError, ValueError):
        return None


if _major_version(np.__version__) is not None and _major_version(np.__version__) >= 2:
    raise RuntimeError(
        "Detected NumPy {}. DPT-LSG currently requires `numpy<2` because the "
        "PyTorch 2.0.1 / DBAF / LieTorch stack in this repo relies on extensions "
        "built against the NumPy 1.x ABI. Please downgrade NumPy and rebuild "
        "`submodules/dbaf` in the same environment.".format(np.__version__)
    )

try:
    import scipy  # noqa: F401
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "SciPy is required by the frontend and utility modules, but it is not "
        "installed in this environment. Install `scipy` before running DPT-LSG."
    ) from exc

import torch
from lietorch import SE3
from frontend.dbaf import DBAFusion
from gaussian.gaussian_model import GaussianModel
from gaussian.vis_utils import save_ply, vis_map, vis_bev

parser = argparse.ArgumentParser(description="Add config path.")
parser.add_argument("config")
parser.add_argument("--prefix", default='')
args = parser.parse_args()
config_path = args.config
from gaussian.general_utils import load_config, get_name
config = load_config(config_path)
import importlib
get_dataset = importlib.import_module(config["dataset"]["module"]).get_dataset
from vings_utils.middleware_utils import judge_and_package, retrieve_to_tracker, datapacket_to_nerfslam
from vings_utils.memory_monitor import MemoryMonitor
from storage.storage_manage import StorageManager
from submap.submap_manager import SubmapManager
from loop.loop_model import LoopModel
from metric.metric_model import Metric_Model
import time
from tqdm import tqdm
if config['mode'] == 'vo_nerfslam': from frontend_vo.vio_slam import VioSLAM
if config['mode'] == 'vo_mast3rslam': from frontend_mast3r.mast3r_slam import Mast3rSLAM


class Runner:
    def __init__(self, cfg):
        self.cfg = cfg
        self.memory_monitor = MemoryMonitor(cfg, label="run")
        self.dataset  = get_dataset(cfg)
        cfg['frontend']['c2i'] = self.dataset.c2i # (4, 4), ndarray
        
        if self.cfg['mode'] == 'vio' or self.cfg['mode'] == 'vo':
            self.tracker = DBAFusion(cfg)
        elif self.cfg['mode'] == 'vo_nerfslam':     
            self.tracker = VioSLAM(cfg)
        elif self.cfg['mode'] == 'vo_mast3rslam':
            self.tracker = Mast3rSLAM(cfg)
        else: assert False, "Error \"mode\" in config file."
        
        if 'phone' not in cfg['dataset']['module']: self.tracker.dataset_length = len(self.dataset)
        
        self.mapper = GaussianModel(cfg)
        self.looper = None
        
        if 'use_metric' in cfg.keys() and cfg['use_metric'] and self.cfg['mode'] != 'vo_mast3rslam':
            self.metric_predictor = Metric_Model(cfg) 
        
        if 'use_storage_manager' in cfg.keys() and cfg['use_storage_manager']:
            self.use_storage_manager = True
            self.storage_manager = StorageManager(cfg)
            if cfg['dataset']['module'] != 'phone':
                self.storage_manager.dataset_length = self.dataset.rgbinfo_dict['timestamp'][-1] - self.dataset.rgbinfo_dict['timestamp'][0] 
        else:
            self.use_storage_manager = False
            self.storage_manager = None

        self.submap_manager = None
        submap_cfg = self.cfg.get('submap', {})
        if submap_cfg.get('enabled', False):
            if self.cfg['mode'] != 'vo':
                print('Submap manager currently supports `mode: vo`; falling back to the legacy single-map path.')
            else:
                self.submap_manager = SubmapManager(
                    cfg,
                    self.mapper,
                    self.storage_manager if self.use_storage_manager else None,
                )

        self.use_submap_manager = self.submap_manager is not None
        if not self.use_submap_manager:
            self.looper = LoopModel(cfg)

        self.memory_monitor.record(-1, tag="post_init", force=True)

    def run(self):
        # Load imu data.
        self.tracker.frontend.all_imu   = self.dataset.preload_imu()
        self.tracker.frontend.all_stamp = self.dataset.preload_camtimestamp()
        
        mapper_run_times = 0
        try:
            # Run Tracking.
            for idx in tqdm(range(len(self.dataset))):
                
                data_packet = self.dataset[idx]
                
                if 'use_mobile' in self.cfg.keys() and self.cfg['use_mobile']:
                    self.tracker.frontend.all_imu   = self.dataset.preload_imu()
                    self.tracker.frontend.all_stamp = self.dataset.preload_camtimestamp()
                
                if 'use_metric' in self.cfg.keys() and self.cfg['use_metric'] and self.cfg['mode'] != 'vo_mast3rslam':
                    if 'depth' not in data_packet.keys() or data_packet['depth'] is None:
                        data_packet['depth'] = self.metric_predictor.predict(data_packet['rgb'][0])
                
                self.tracker.frontend.all_imu   = self.dataset.preload_imu()
                self.tracker.frontend.all_stamp = self.dataset.preload_camtimestamp()
                
                # torch.set_grad_enabled(False)
                if self.cfg['mode'] == 'vo_nerfslam':
                    tracker_input = datapacket_to_nerfslam(data_packet, idx)
                else:
                    tracker_input = data_packet
                self.tracker.track(tracker_input)
                # torch.set_grad_enabled(True)
                
                torch.cuda.empty_cache()
                # Judge whether new keyframe is added and package keyframe dict.
                viz_out = judge_and_package(self.tracker, data_packet['intrinsic'])
                
                if viz_out is not None and (self.cfg['mode'] in ['vo', 'vo_nerfslam', 'vo_mast3rslam'] or self.tracker.video.imu_enabled):
                    if self.use_submap_manager:
                        self.submap_manager.process(self.tracker, viz_out, idx)
                        self.mapper = self.submap_manager.active_mapper
                        self.storage_manager = self.submap_manager.active_storage_manager
                    else:
                        # Save and check.
                        self.mapper.run(viz_out, True)
                        
                        if 'use_loop' in list(self.cfg.keys()) and self.cfg['use_loop']:
                            if viz_out["global_kf_id"][-1] > 10 and viz_out["global_kf_id"][-1] % 3 == 0:
                                self.looper.run(self.mapper, self.tracker, viz_out, idx)

                        if self.use_storage_manager and (idx+1) % 10 == 0:
                            self.storage_manager.run(self.tracker, self.mapper, viz_out)
                            torch.cuda.empty_cache()
                    
                    if self.cfg['use_vis'] and (idx+1) % 1 == 0:
                        if (not self.use_storage_manager) or self.storage_manager is None or self.storage_manager._xyz.shape[0]==0:
                            vis_map(self.tracker, self.mapper)
                            vis_bev(self.tracker, self.mapper) 
                        else:
                            self.storage_manager.vis_map_storage(self.tracker, self.mapper)    
                            self.storage_manager.vis_bev_storage(self.tracker, self.mapper)    

                self.memory_monitor.record(idx, tag="frame_end")
                
                if (idx == len(self.dataset) - 1) and self.mapper._xyz.shape[0] > 0:
                # if ((idx+1) % 100 == 0 or (idx == len(self.dataset) - 1)) and self.mapper._xyz.shape[0] > 0:
                    export_storage = None
                    if self.use_submap_manager:
                        export_storage = self.submap_manager.build_export_storage_view()
                    elif self.use_storage_manager:
                        export_storage = self.storage_manager
                    save_ply(
                        self.mapper,
                        idx,
                        save_mode='2dgs',
                        storage_manager=export_storage,
                    )
                    if self.use_submap_manager:
                        self.submap_manager.export_submap_plys(idx, save_mode='2dgs')
                    # save_ply(self.mapper, idx, save_mode='pth')
        finally:
            self.memory_monitor.close()
            

if __name__ == '__main__':
    
    config['output']['save_dir'] = os.path.join(config['output']['save_dir'], get_name(config)+'-{}-'.format(config_path.split('/')[-1].strip('.yaml'))+args.prefix)
    os.makedirs(config['output']['save_dir']+'/droid_c2w', exist_ok=True)
    os.makedirs(config['output']['save_dir']+'/rgbdnua', exist_ok=True)
    os.makedirs(config['output']['save_dir']+'/ply', exist_ok=True)
    if 'debug_mode' in list(config.keys()) and config['debug_mode']:
        os.makedirs(config['output']['save_dir']+'/debug_dict', exist_ok=True)
    shutil.copy(config_path, config['output']['save_dir']+'/config.yaml')
    
    runner = Runner(config)
    torch.backends.cudnn.benchmark = True
    
    runner.run()
    
    
