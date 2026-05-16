import atexit
import csv
import json
import os
import time

_default_mpl_dir = os.path.expanduser("~/.config/matplotlib")
if "MPLCONFIGDIR" not in os.environ and (not os.path.isdir(_default_mpl_dir) or not os.access(_default_mpl_dir, os.W_OK)):
    os.environ["MPLCONFIGDIR"] = "/tmp/vings_matplotlib"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import psutil
import torch


BYTES_PER_MB = float(1024 ** 2)


def _to_mb(value_in_bytes):
    return float(value_in_bytes) / BYTES_PER_MB


def _normalize_cuda_device(device_name):
    device = torch.device(device_name)
    if device.type != "cuda":
        return None
    index = device.index if device.index is not None else torch.cuda.current_device()
    return f"cuda:{index}"


def _resolve_cuda_devices(cfg):
    if not torch.cuda.is_available():
        return []

    devices = []
    seen = set()
    for device_name in cfg.get("device", {}).values():
        if not isinstance(device_name, str):
            continue
        normalized = _normalize_cuda_device(device_name)
        if normalized is None or normalized in seen:
            continue
        seen.add(normalized)
        devices.append(normalized)

    if not devices:
        devices.append(f"cuda:{torch.cuda.current_device()}")
    return devices


def _prepare_cuda_devices(devices):
    if not devices:
        return []

    try:
        torch.cuda.init()
    except RuntimeError:
        return []

    prepared_devices = []
    for device_name in devices:
        try:
            torch.empty(0, device=device_name)
            torch.cuda.reset_peak_memory_stats(torch.device(device_name))
            prepared_devices.append(device_name)
        except RuntimeError:
            continue
    return prepared_devices


class MemoryMonitor:
    def __init__(self, cfg, label="run"):
        monitor_cfg = cfg.get("memory_monitor", {})
        self.enabled = monitor_cfg.get("enabled", True)
        self.interval = max(1, int(monitor_cfg.get("interval", 1)))
        self.label = label
        self.cfg = cfg
        self.samples = []
        self.closed = False
        self.process = psutil.Process(os.getpid())
        self.start_time = time.time()
        self.save_dir = os.path.join(cfg["output"]["save_dir"], monitor_cfg.get("subdir", "memory"))
        self.csv_path = os.path.join(self.save_dir, f"{self.label}_metrics.csv")
        self.plot_path = os.path.join(self.save_dir, f"{self.label}_plot.png")
        self.summary_path = os.path.join(self.save_dir, f"{self.label}_summary.json")
        self.cuda_devices = _resolve_cuda_devices(cfg) if self.enabled else []

        if self.enabled:
            os.makedirs(self.save_dir, exist_ok=True)
            self.cuda_devices = _prepare_cuda_devices(self.cuda_devices)
            atexit.register(self.close)
            self.record(-1, tag="start", force=True)

    def record(self, frame_idx, tag="frame_end", force=False):
        if not self.enabled or self.closed:
            return
        if not force and frame_idx is not None and frame_idx >= 0 and frame_idx % self.interval != 0:
            return
        self.samples.append(self._collect_sample(frame_idx, tag))

    def close(self):
        if not self.enabled or self.closed:
            return
        if not self.samples or self.samples[-1]["tag"] != "final":
            final_frame_idx = self.samples[-1]["frame_idx"] if self.samples else -1
            self.samples.append(self._collect_sample(final_frame_idx, "final"))
        self._write_csv()
        self._write_summary()
        self._write_plot()
        self.closed = True

    def _collect_sample(self, frame_idx, tag):
        process_info = self.process.memory_info()
        try:
            full_info = self.process.memory_full_info()
        except Exception:
            full_info = process_info
        system_memory = psutil.virtual_memory()
        system_swap = psutil.swap_memory()

        sample = {
            "frame_idx": int(frame_idx) if frame_idx is not None else len(self.samples),
            "sample_index": len(self.samples),
            "tag": tag,
            "time_sec": time.time() - self.start_time,
            "process_rss_mb": _to_mb(process_info.rss),
            "process_vms_mb": _to_mb(process_info.vms),
            "system_ram_used_mb": _to_mb(system_memory.used),
            "system_ram_available_mb": _to_mb(system_memory.available),
            "process_swap_mb": _to_mb(getattr(full_info, "swap", 0.0)),
            "system_swap_used_mb": _to_mb(system_swap.used),
            "system_swap_free_mb": _to_mb(system_swap.free),
        }

        for device_name in self.cuda_devices:
            alloc = _to_mb(torch.cuda.memory_allocated(torch.device(device_name)))
            reserved = _to_mb(torch.cuda.memory_reserved(torch.device(device_name)))
            max_alloc = _to_mb(torch.cuda.max_memory_allocated(torch.device(device_name)))
            max_reserved = _to_mb(torch.cuda.max_memory_reserved(torch.device(device_name)))
            free_bytes, total_bytes = torch.cuda.mem_get_info(torch.device(device_name))
            used = _to_mb(total_bytes - free_bytes)
            total = _to_mb(total_bytes)
            key = device_name.replace(":", "_")
            sample[f"{key}_allocated_mb"] = alloc
            sample[f"{key}_reserved_mb"] = reserved
            sample[f"{key}_max_allocated_mb"] = max_alloc
            sample[f"{key}_max_reserved_mb"] = max_reserved
            sample[f"{key}_device_used_mb"] = used
            sample[f"{key}_device_total_mb"] = total

        return sample

    def _write_csv(self):
        if not self.samples:
            return
        fieldnames = list(self.samples[0].keys())
        with open(self.csv_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.samples)

    def _write_summary(self):
        if not self.samples:
            return

        summary = {
            "label": self.label,
            "num_samples": len(self.samples),
            "duration_sec": self.samples[-1]["time_sec"],
            "peak_process_rss_mb": max(sample["process_rss_mb"] for sample in self.samples),
            "peak_process_swap_mb": max(sample["process_swap_mb"] for sample in self.samples),
            "peak_system_ram_used_mb": max(sample["system_ram_used_mb"] for sample in self.samples),
            "peak_system_swap_used_mb": max(sample["system_swap_used_mb"] for sample in self.samples),
            "devices": {},
        }

        for device_name in self.cuda_devices:
            key = device_name.replace(":", "_")
            summary["devices"][device_name] = {
                "peak_allocated_mb": max(sample[f"{key}_allocated_mb"] for sample in self.samples),
                "peak_reserved_mb": max(sample[f"{key}_reserved_mb"] for sample in self.samples),
                "peak_max_allocated_mb": max(sample[f"{key}_max_allocated_mb"] for sample in self.samples),
                "peak_max_reserved_mb": max(sample[f"{key}_max_reserved_mb"] for sample in self.samples),
                "peak_device_used_mb": max(sample[f"{key}_device_used_mb"] for sample in self.samples),
                "device_total_mb": self.samples[-1][f"{key}_device_total_mb"],
            }

        with open(self.summary_path, "w", encoding="utf-8") as summary_file:
            json.dump(summary, summary_file, indent=2)

    def _write_plot(self):
        if not self.samples:
            return

        frame_axis = [sample["frame_idx"] for sample in self.samples]
        fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)

        if self.cuda_devices:
            for device_name in self.cuda_devices:
                key = device_name.replace(":", "_")
                axes[0].plot(frame_axis, [sample[f"{key}_allocated_mb"] for sample in self.samples], label=f"{device_name} allocated")
                axes[0].plot(frame_axis, [sample[f"{key}_reserved_mb"] for sample in self.samples], label=f"{device_name} reserved")
                axes[0].plot(frame_axis, [sample[f"{key}_device_used_mb"] for sample in self.samples], linestyle="--", label=f"{device_name} device used")
            axes[0].set_ylabel("VRAM (MB)")
            axes[0].legend(loc="upper left", ncol=2)
        else:
            axes[0].text(0.5, 0.5, "CUDA not available", ha="center", va="center", transform=axes[0].transAxes)
            axes[0].set_ylabel("VRAM (MB)")

        axes[1].plot(frame_axis, [sample["process_rss_mb"] for sample in self.samples], label="Process RSS")
        axes[1].plot(frame_axis, [sample["system_ram_used_mb"] for sample in self.samples], linestyle="--", label="System RAM used")
        axes[1].set_ylabel("DRAM (MB)")
        axes[1].legend(loc="upper left")

        axes[2].plot(frame_axis, [sample["process_swap_mb"] for sample in self.samples], label="Process swap")
        axes[2].plot(frame_axis, [sample["system_swap_used_mb"] for sample in self.samples], linestyle="--", label="System swap used")
        axes[2].set_ylabel("Swap (MB)")
        axes[2].set_xlabel("Frame / Iteration")
        axes[2].legend(loc="upper left")

        fig.suptitle(f"Memory Usage Monitor: {self.label}")
        fig.tight_layout()
        fig.savefig(self.plot_path, dpi=180)
        plt.close(fig)
