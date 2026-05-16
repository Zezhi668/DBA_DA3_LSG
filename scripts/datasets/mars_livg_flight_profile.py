#!/usr/bin/env python3
"""Plot DJI flight-height signals for a MARS-LIVG rosbag.

The script reads DJI GPS / local position / velocity / IMU topics and writes:
  - a PNG plot for visual inspection
  - a JSON summary with a suggested trim window
  - an optional CSV with merged signals

The suggested trim window is based primarily on `/dji_osdk_ros/height_above_takeoff`
with a persistence threshold so take-off and landing are excluded.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import rosbag


DEFAULT_DATASET_ROOT = Path("/media/server/yzz_disk/Dataset_sx/MARS-LIVG")
DEFAULT_HEIGHT_TOPIC = "/dji_osdk_ros/height_above_takeoff"
DEFAULT_GPS_TOPIC = "/dji_osdk_ros/gps_position"
DEFAULT_LOCAL_POS_TOPIC = "/dji_osdk_ros/local_position"
DEFAULT_VELOCITY_TOPIC = "/dji_osdk_ros/velocity"
DEFAULT_IMU_TOPIC = "/dji_osdk_ros/imu"


@dataclass
class TimeSeries:
    timestamps: np.ndarray
    values: np.ndarray
    label: str

    @property
    def empty(self) -> bool:
        return self.timestamps.size == 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect MARS-LIVG flight height and suggest a trim window."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Root folder that contains the MARS-LIVG bags.",
    )
    parser.add_argument(
        "--sequence",
        type=str,
        default="HKairport03",
        help="Sequence stem, for example HKairport03.",
    )
    parser.add_argument(
        "--bag-path",
        type=Path,
        default=None,
        help="Optional explicit rosbag path. Overrides --sequence lookup.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output folder. Defaults to <dataset-root>/flight_profile/<sequence>.",
    )
    parser.add_argument(
        "--height-topic",
        type=str,
        default=DEFAULT_HEIGHT_TOPIC,
        help="Primary DJI height topic used for trim suggestions.",
    )
    parser.add_argument(
        "--gps-topic",
        type=str,
        default=DEFAULT_GPS_TOPIC,
        help="GPS topic used to plot altitude.",
    )
    parser.add_argument(
        "--local-position-topic",
        type=str,
        default=DEFAULT_LOCAL_POS_TOPIC,
        help="Local position topic used to plot altitude.",
    )
    parser.add_argument(
        "--velocity-topic",
        type=str,
        default=DEFAULT_VELOCITY_TOPIC,
        help="Velocity topic used to plot vertical speed.",
    )
    parser.add_argument(
        "--imu-topic",
        type=str,
        default=DEFAULT_IMU_TOPIC,
        help="IMU topic used to plot vertical acceleration.",
    )
    parser.add_argument(
        "--min-height",
        type=float,
        default=1.5,
        help="Minimum sustained height-above-takeoff in meters to consider the UAV airborne.",
    )
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=5.0,
        help="Required duration above the height threshold before suggesting a trim point.",
    )
    parser.add_argument(
        "--moving-average-seconds",
        type=float,
        default=1.5,
        help="Smoothing window for the plotted height and speed signals.",
    )
    parser.add_argument(
        "--trim-signal",
        choices=("auto", "height_above_takeoff", "gps_altitude_relative", "local_position_z_relative"),
        default="auto",
        help="Signal used to suggest the trim window.",
    )
    parser.add_argument(
        "--flat-speed-threshold",
        type=float,
        default=0.15,
        help="Absolute smoothed vertical speed threshold in m/s used to define a flat segment.",
    )
    parser.add_argument(
        "--motion-speed-threshold",
        type=float,
        default=0.5,
        help="Absolute smoothed vertical speed threshold in m/s used to define clear climb/descent motion.",
    )
    parser.add_argument(
        "--speed-hold-seconds",
        type=float,
        default=3.0,
        help="Required duration for flat or motion segments in the vertical-speed analysis.",
    )
    parser.add_argument(
        "--write-csv",
        action="store_true",
        help="Also write a merged CSV for the plotted signals.",
    )
    parser.add_argument(
        "--csv-rate-hz",
        type=float,
        default=20.0,
        help="Merged CSV sample rate. Use <=0 to keep every raw message timestamp.",
    )
    parser.add_argument(
        "--skip-imu",
        action="store_true",
        help="Skip the high-rate IMU topic. This is usually enough for trim-window selection.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50000,
        help="Print bag-reading progress every N selected messages. Use <=0 to disable.",
    )
    return parser.parse_args()


def infer_bag_path(dataset_root: Path, sequence: str, bag_path: Optional[Path]) -> Path:
    if bag_path is not None:
        return bag_path
    return dataset_root / f"{sequence}.bag"


def infer_output_dir(dataset_root: Path, sequence: str, output_dir: Optional[Path]) -> Path:
    if output_dir is not None:
        return output_dir
    return dataset_root / "flight_profile" / sequence


def moving_average(values: np.ndarray, window_size: int) -> np.ndarray:
    if values.size == 0 or window_size <= 1:
        return values.copy()
    window_size = min(window_size, values.size)
    left_pad = (window_size - 1) // 2
    right_pad = window_size // 2
    padded = np.pad(values.astype(np.float64), (left_pad, right_pad), mode="edge")
    cumulative = np.cumsum(np.insert(padded, 0, 0.0))
    return (cumulative[window_size:] - cumulative[:-window_size]) / float(window_size)


def estimate_frequency(timestamps: np.ndarray) -> float:
    if timestamps.size < 2:
        return 0.0
    deltas = np.diff(timestamps)
    positive = deltas[deltas > 0]
    if positive.size == 0:
        return 0.0
    return float(1.0 / np.median(positive))


def window_size_from_seconds(timestamps: np.ndarray, seconds: float) -> int:
    freq = estimate_frequency(timestamps)
    if freq <= 0.0:
        return 1
    return max(1, int(round(freq * seconds)))


def duration_true_segments(mask: np.ndarray, timestamps: np.ndarray, min_duration: float) -> List[Tuple[int, int]]:
    segments: List[Tuple[int, int]] = []
    if mask.size == 0:
        return segments

    start = None
    for idx, flag in enumerate(mask):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            end = idx - 1
            if timestamps[end] - timestamps[start] >= min_duration:
                segments.append((start, end))
            start = None

    if start is not None:
        end = mask.size - 1
        if timestamps[end] - timestamps[start] >= min_duration:
            segments.append((start, end))

    return segments


def suggest_trim_window(
    height_series: TimeSeries,
    min_height: float,
    hold_seconds: float,
    moving_average_seconds: float,
) -> Optional[Dict[str, float]]:
    if height_series.empty:
        return None

    smooth_window = window_size_from_seconds(height_series.timestamps, moving_average_seconds)
    smooth_height = moving_average(height_series.values, smooth_window)
    airborne_mask = smooth_height >= min_height
    segments = duration_true_segments(airborne_mask, height_series.timestamps, hold_seconds)
    if not segments:
        return None

    start_idx, end_idx = max(
        segments,
        key=lambda seg: height_series.timestamps[seg[1]] - height_series.timestamps[seg[0]],
    )
    return {
        "start_timestamp": float(height_series.timestamps[start_idx]),
        "end_timestamp": float(height_series.timestamps[end_idx]),
        "start_offset_sec": float(height_series.timestamps[start_idx] - height_series.timestamps[0]),
        "end_offset_sec": float(height_series.timestamps[end_idx] - height_series.timestamps[0]),
        "duration_sec": float(height_series.timestamps[end_idx] - height_series.timestamps[start_idx]),
        "candidate_segments": int(len(segments)),
        "min_height_m": float(min_height),
        "hold_seconds": float(hold_seconds),
        "smoothing_window_samples": int(smooth_window),
    }


def first_timestamp(series_map: Dict[str, TimeSeries]) -> Optional[float]:
    non_empty = [float(series.timestamps[0]) for series in series_map.values() if not series.empty]
    if not non_empty:
        return None
    return min(non_empty)


def make_trim_summary(
    start_timestamp: float,
    end_timestamp: float,
    reference_timestamp: Optional[float],
    extra_fields: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "start_timestamp": float(start_timestamp),
        "end_timestamp": float(end_timestamp),
        "duration_sec": float(end_timestamp - start_timestamp),
    }
    if reference_timestamp is None:
        summary["start_offset_sec"] = None
        summary["end_offset_sec"] = None
    else:
        summary["start_offset_sec"] = float(start_timestamp - reference_timestamp)
        summary["end_offset_sec"] = float(end_timestamp - reference_timestamp)
    if extra_fields:
        summary.update(extra_fields)
    return summary


def choose_trim_signal(
    requested_signal: str,
    series_map: Dict[str, TimeSeries],
    min_height: float,
    hold_seconds: float,
    moving_average_seconds: float,
) -> Tuple[str, Optional[Dict[str, float]], Dict[str, Optional[Dict[str, float]]]]:
    candidate_series = {
        "height_above_takeoff": series_map["height_above_takeoff"],
        "gps_altitude_relative": maybe_relativize(series_map["gps_altitude"], relative=True),
        "local_position_z_relative": maybe_relativize(series_map["local_position_z"], relative=True),
    }

    candidate_summaries: Dict[str, Optional[Dict[str, float]]] = {}
    for key, series in candidate_series.items():
        candidate_summaries[key] = suggest_trim_window(
            height_series=series,
            min_height=min_height,
            hold_seconds=hold_seconds,
            moving_average_seconds=moving_average_seconds,
        )

    if requested_signal != "auto":
        return requested_signal, candidate_summaries.get(requested_signal), candidate_summaries

    best_key = None
    best_summary = None
    best_rank = None
    for key, summary in candidate_summaries.items():
        if summary is None:
            continue
        rank = (
            float(summary["duration_sec"]),
            float(summary["end_timestamp"] - summary["start_timestamp"]),
            -float(summary["start_timestamp"]),
        )
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_key = key
            best_summary = summary

    if best_key is None:
        return "height_above_takeoff", None, candidate_summaries
    return best_key, best_summary, candidate_summaries


def tighten_trim_window_with_vertical_speed(
    suggested_trim: Optional[Dict[str, object]],
    vertical_speed_analysis: Optional[Dict[str, object]],
    reference_timestamp: Optional[float],
) -> Tuple[Optional[Dict[str, object]], Optional[Dict[str, float]], str]:
    speed_candidate_obj = None if vertical_speed_analysis is None else vertical_speed_analysis.get("flat_period_candidate")
    speed_candidate = speed_candidate_obj if isinstance(speed_candidate_obj, dict) else None

    if suggested_trim is None and speed_candidate is None:
        return None, None, "none"

    if suggested_trim is None and speed_candidate is not None:
        final_trim = make_trim_summary(
            float(speed_candidate["start_timestamp"]),
            float(speed_candidate["end_timestamp"]),
            reference_timestamp,
            {
                "trim_strategy": "vertical_speed_flat_period",
                "speed_candidate_start_timestamp": float(speed_candidate["start_timestamp"]),
                "speed_candidate_end_timestamp": float(speed_candidate["end_timestamp"]),
            },
        )
        return final_trim, speed_candidate, "vertical_speed_flat_period"

    if suggested_trim is not None and speed_candidate is None:
        final_trim = dict(suggested_trim)
        final_trim["trim_strategy"] = "altitude_threshold_only"
        return final_trim, None, "altitude_threshold_only"

    assert suggested_trim is not None
    assert speed_candidate is not None

    tightened_start = max(
        float(suggested_trim["start_timestamp"]),
        float(speed_candidate["start_timestamp"]),
    )
    tightened_end = min(
        float(suggested_trim["end_timestamp"]),
        float(speed_candidate["end_timestamp"]),
    )
    if tightened_end <= tightened_start:
        final_trim = dict(suggested_trim)
        final_trim["trim_strategy"] = "altitude_threshold_only"
        final_trim["speed_candidate_start_timestamp"] = float(speed_candidate["start_timestamp"])
        final_trim["speed_candidate_end_timestamp"] = float(speed_candidate["end_timestamp"])
        final_trim["speed_candidate_overlap"] = False
        return final_trim, speed_candidate, "altitude_threshold_only"

    final_trim = make_trim_summary(
        tightened_start,
        tightened_end,
        reference_timestamp,
        {
            "trim_strategy": "altitude_and_vertical_speed_intersection",
            "altitude_candidate_start_timestamp": float(suggested_trim["start_timestamp"]),
            "altitude_candidate_end_timestamp": float(suggested_trim["end_timestamp"]),
            "speed_candidate_start_timestamp": float(speed_candidate["start_timestamp"]),
            "speed_candidate_end_timestamp": float(speed_candidate["end_timestamp"]),
            "speed_candidate_overlap": True,
        },
    )
    return final_trim, speed_candidate, "altitude_and_vertical_speed_intersection"


def summarize_segment(
    timestamps: np.ndarray,
    values: np.ndarray,
    segment: Tuple[int, int],
) -> Dict[str, float]:
    start_idx, end_idx = segment
    segment_values = values[start_idx : end_idx + 1]
    return {
        "start_timestamp": float(timestamps[start_idx]),
        "end_timestamp": float(timestamps[end_idx]),
        "duration_sec": float(timestamps[end_idx] - timestamps[start_idx]),
        "mean_value": float(np.mean(segment_values)),
        "min_value": float(np.min(segment_values)),
        "max_value": float(np.max(segment_values)),
    }


def analyze_vertical_speed(
    speed_series: TimeSeries,
    moving_average_seconds: float,
    flat_speed_threshold: float,
    motion_speed_threshold: float,
    hold_seconds: float,
) -> Optional[Dict[str, object]]:
    if speed_series.empty:
        return None

    smooth_window = window_size_from_seconds(speed_series.timestamps, moving_average_seconds)
    smooth_speed = moving_average(speed_series.values, smooth_window)

    flat_segments = duration_true_segments(
        np.abs(smooth_speed) <= flat_speed_threshold,
        speed_series.timestamps,
        hold_seconds,
    )
    positive_segments = duration_true_segments(
        smooth_speed >= motion_speed_threshold,
        speed_series.timestamps,
        hold_seconds,
    )
    negative_segments = duration_true_segments(
        smooth_speed <= -motion_speed_threshold,
        speed_series.timestamps,
        hold_seconds,
    )

    flat_summaries = [
        summarize_segment(speed_series.timestamps, smooth_speed, segment) for segment in flat_segments
    ]
    positive_summaries = [
        summarize_segment(speed_series.timestamps, smooth_speed, segment) for segment in positive_segments
    ]
    negative_summaries = [
        summarize_segment(speed_series.timestamps, smooth_speed, segment) for segment in negative_segments
    ]

    post_positive_flat_events: List[Dict[str, float]] = []
    for segment in positive_segments:
        next_flat = next((flat for flat in flat_segments if flat[0] > segment[1]), None)
        if next_flat is None:
            continue
        post_positive_flat_events.append(
            {
                "motion_end_timestamp": float(speed_series.timestamps[segment[1]]),
                "flat_start_timestamp": float(speed_series.timestamps[next_flat[0]]),
                "gap_sec": float(speed_series.timestamps[next_flat[0]] - speed_series.timestamps[segment[1]]),
            }
        )

    pre_negative_motion_events: List[Dict[str, float]] = []
    for segment in negative_segments:
        prev_flat = next((flat for flat in reversed(flat_segments) if flat[1] < segment[0]), None)
        if prev_flat is None:
            continue
        pre_negative_motion_events.append(
            {
                "flat_end_timestamp": float(speed_series.timestamps[prev_flat[1]]),
                "motion_start_timestamp": float(speed_series.timestamps[segment[0]]),
                "gap_sec": float(speed_series.timestamps[segment[0]] - speed_series.timestamps[prev_flat[1]]),
            }
        )

    post_negative_flat_events: List[Dict[str, float]] = []
    for segment in negative_segments:
        next_flat = next((flat for flat in flat_segments if flat[0] > segment[1]), None)
        if next_flat is None:
            continue
        post_negative_flat_events.append(
            {
                "motion_end_timestamp": float(speed_series.timestamps[segment[1]]),
                "flat_start_timestamp": float(speed_series.timestamps[next_flat[0]]),
                "gap_sec": float(speed_series.timestamps[next_flat[0]] - speed_series.timestamps[segment[1]]),
            }
        )

    flat_period_candidate = None
    if post_positive_flat_events and negative_summaries:
        first_negative_start = float(negative_summaries[0]["start_timestamp"])
        valid_takeoff_returns = [
            event for event in post_positive_flat_events if event["flat_start_timestamp"] < first_negative_start
        ]
        if valid_takeoff_returns:
            takeoff_end = valid_takeoff_returns[-1]["flat_start_timestamp"]
            flat_period_candidate = {
                "start_timestamp": float(takeoff_end),
                "end_timestamp": float(first_negative_start),
                "duration_sec": float(first_negative_start - takeoff_end),
            }

    return {
        "flat_speed_threshold_mps": float(flat_speed_threshold),
        "motion_speed_threshold_mps": float(motion_speed_threshold),
        "hold_seconds": float(hold_seconds),
        "smoothing_window_samples": int(smooth_window),
        "flat_segments": flat_summaries,
        "positive_motion_segments": positive_summaries,
        "negative_motion_segments": negative_summaries,
        "post_positive_flat_events": post_positive_flat_events,
        "pre_negative_motion_events": pre_negative_motion_events,
        "post_negative_flat_events": post_negative_flat_events,
        "flat_period_candidate": flat_period_candidate,
    }


def to_relative_height(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.copy()
    return values - values[0]


def read_scalar_topic(bag: rosbag.Bag, topic: str, field_getter) -> TimeSeries:
    timestamps: List[float] = []
    values: List[float] = []
    for _, msg, t in bag.read_messages(topics=[topic]):
        timestamps.append(t.to_sec())
        values.append(float(field_getter(msg)))
    return TimeSeries(np.asarray(timestamps, dtype=np.float64), np.asarray(values, dtype=np.float64), topic)


def read_topics(
    bag_path: Path,
    height_topic: str,
    gps_topic: str,
    local_position_topic: str,
    velocity_topic: str,
    imu_topic: str,
    include_imu: bool = True,
    progress_every: int = 50000,
) -> Dict[str, TimeSeries]:
    topic_specs = {
        "height_above_takeoff": (height_topic, lambda msg: msg.data),
        "gps_altitude": (gps_topic, lambda msg: msg.altitude),
        "local_position_z": (local_position_topic, lambda msg: msg.point.z),
        "vertical_speed_z": (velocity_topic, lambda msg: msg.vector.z),
    }
    if include_imu:
        topic_specs["imu_accel_z"] = (imu_topic, lambda msg: msg.linear_acceleration.z)

    topic_to_keys: Dict[str, List[str]] = {}
    for key, (topic, _) in topic_specs.items():
        topic_to_keys.setdefault(topic, []).append(key)
    topics = list(topic_to_keys.keys())
    timestamps: Dict[str, List[float]] = {key: [] for key in topic_specs}
    values: Dict[str, List[float]] = {key: [] for key in topic_specs}

    with rosbag.Bag(str(bag_path), "r") as bag:
        try:
            total_selected = int(bag.get_message_count(topic_filters=topics))
        except TypeError:
            total_selected = int(bag.get_message_count(topics))
        except Exception:
            total_selected = 0
        print(
            f"Reading {bag_path} once for {len(topics)} topics"
            + (f" ({total_selected} selected messages)" if total_selected > 0 else ""),
            flush=True,
        )

        selected_count = 0
        for topic, msg, t in bag.read_messages(topics=topics):
            selected_count += 1
            stamp = t.to_sec()
            for key in topic_to_keys.get(topic, []):
                _, getter = topic_specs[key]
                try:
                    value = float(getter(msg))
                except Exception as exc:
                    print(f"Skipping malformed message on {topic}: {exc}", flush=True)
                    continue
                timestamps[key].append(stamp)
                values[key].append(value)
            if progress_every > 0 and selected_count % progress_every == 0:
                if total_selected > 0:
                    print(f"  read {selected_count}/{total_selected} selected messages", flush=True)
                else:
                    print(f"  read {selected_count} selected messages", flush=True)

    series_map = {
        key: TimeSeries(
            np.asarray(timestamps[key], dtype=np.float64),
            np.asarray(values[key], dtype=np.float64),
            topic_specs[key][0],
        )
        for key in topic_specs
    }
    if "imu_accel_z" not in series_map:
        series_map["imu_accel_z"] = TimeSeries(
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=np.float64),
            imu_topic,
        )
    for key, series in series_map.items():
        print(f"  {key}: {series.timestamps.size} messages from {series.label}", flush=True)
    return series_map


def maybe_relativize(series: TimeSeries, relative: bool) -> TimeSeries:
    if series.empty or not relative:
        return series
    return TimeSeries(series.timestamps.copy(), to_relative_height(series.values), series.label)


def csv_timestamps(series_map: Dict[str, TimeSeries], sample_rate_hz: float) -> List[float]:
    non_empty = [series for series in series_map.values() if not series.empty]
    if not non_empty:
        return []
    if sample_rate_hz <= 0:
        return sorted({float(timestamp) for series in non_empty for timestamp in series.timestamps.tolist()})
    start = min(float(series.timestamps[0]) for series in non_empty)
    end = max(float(series.timestamps[-1]) for series in non_empty)
    step = 1.0 / float(sample_rate_hz)
    return np.arange(start, end + 0.5 * step, step, dtype=np.float64).tolist()


def merge_series_for_csv(series_map: Dict[str, TimeSeries], sample_rate_hz: float) -> List[Dict[str, str]]:
    all_timestamps = csv_timestamps(series_map, sample_rate_hz)
    rows: List[Dict[str, str]] = []
    for timestamp in all_timestamps:
        row = {"timestamp": f"{timestamp:.9f}"}
        for key, series in series_map.items():
            if series.empty:
                row[key] = ""
                continue
            idx = int(np.searchsorted(series.timestamps, timestamp))
            best_idx = None
            best_delta = None
            for candidate in (idx - 1, idx):
                if 0 <= candidate < series.timestamps.size:
                    delta = abs(series.timestamps[candidate] - timestamp)
                    if best_delta is None or delta < best_delta:
                        best_delta = delta
                        best_idx = candidate
            if best_idx is None or best_delta is None or best_delta > 0.05:
                row[key] = ""
            else:
                row[key] = f"{series.values[best_idx]:.9f}"
        rows.append(row)
    return rows


def thin_for_plot(series: TimeSeries, max_points: int = 50000) -> TimeSeries:
    if series.empty or series.timestamps.size <= max_points:
        return series
    step = max(1, int(np.ceil(series.timestamps.size / float(max_points))))
    return TimeSeries(series.timestamps[::step], series.values[::step], series.label)


def write_csv(output_path: Path, rows: Sequence[Dict[str, str]], fieldnames: Sequence[str]) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_series(
    output_path: Path,
    bag_path: Path,
    series_map: Dict[str, TimeSeries],
    suggested_trim: Optional[Dict[str, object]],
    vertical_speed_analysis: Optional[Dict[str, object]],
    moving_average_seconds: float,
    min_height: float,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    height_series = series_map["height_above_takeoff"]
    gps_series = maybe_relativize(series_map["gps_altitude"], relative=True)
    local_series = maybe_relativize(series_map["local_position_z"], relative=True)
    speed_series = series_map["vertical_speed_z"]
    imu_series = series_map["imu_accel_z"]

    if not height_series.empty:
        smooth_window = window_size_from_seconds(height_series.timestamps, moving_average_seconds)
        smooth_height = moving_average(height_series.values, smooth_window)
        height_plot = thin_for_plot(height_series)
        smooth_height_plot = thin_for_plot(TimeSeries(height_series.timestamps, smooth_height, "smoothed_height"))
        axes[0].plot(height_plot.timestamps, height_plot.values, label="height_above_takeoff", linewidth=1.0)
        axes[0].plot(smooth_height_plot.timestamps, smooth_height_plot.values, label="smoothed_height", linewidth=2.0)
        axes[0].axhline(min_height, color="tab:red", linestyle="--", linewidth=1.0, label=f"min_height={min_height:.2f}m")

    if not gps_series.empty:
        gps_plot = thin_for_plot(gps_series)
        axes[0].plot(gps_plot.timestamps, gps_plot.values, label="gps_altitude_relative", linewidth=1.0, alpha=0.8)
    if not local_series.empty:
        local_plot = thin_for_plot(local_series)
        axes[0].plot(local_plot.timestamps, local_plot.values, label="local_position_z_relative", linewidth=1.0, alpha=0.8)
    axes[0].set_ylabel("Height / Altitude (m)")
    axes[0].set_title(f"Flight Height Profile: {bag_path.name}")
    axes[0].grid(True, alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, loc="best")

    if not speed_series.empty:
        smooth_window = window_size_from_seconds(speed_series.timestamps, moving_average_seconds)
        speed_plot = thin_for_plot(speed_series)
        smooth_speed_plot = thin_for_plot(
            TimeSeries(
                speed_series.timestamps,
                moving_average(speed_series.values, smooth_window),
                "smoothed_vertical_speed_z",
            )
        )
        axes[1].plot(speed_plot.timestamps, speed_plot.values, label="vertical_speed_z", linewidth=1.0)
        axes[1].plot(
            smooth_speed_plot.timestamps,
            smooth_speed_plot.values,
            label="smoothed_vertical_speed_z",
            linewidth=2.0,
        )
    axes[1].axhline(0.0, color="black", linewidth=1.0, alpha=0.4)
    axes[1].set_ylabel("Vertical Speed (m/s)")
    axes[1].grid(True, alpha=0.3)
    handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        axes[1].legend(handles, labels, loc="best")

    if not imu_series.empty:
        smooth_window = window_size_from_seconds(imu_series.timestamps, moving_average_seconds)
        imu_plot = thin_for_plot(imu_series)
        smooth_imu_plot = thin_for_plot(
            TimeSeries(
                imu_series.timestamps,
                moving_average(imu_series.values, smooth_window),
                "smoothed_imu_linear_acc_z",
            )
        )
        axes[2].plot(imu_plot.timestamps, imu_plot.values, label="imu_linear_acc_z", linewidth=1.0)
        axes[2].plot(
            smooth_imu_plot.timestamps,
            smooth_imu_plot.values,
            label="smoothed_imu_linear_acc_z",
            linewidth=2.0,
        )
    axes[2].set_ylabel("IMU Accel Z (m/s^2)")
    axes[2].set_xlabel("ROS Bag Timestamp (sec)")
    axes[2].grid(True, alpha=0.3)
    handles, labels = axes[2].get_legend_handles_labels()
    if handles:
        axes[2].legend(handles, labels, loc="best")

    marker_specs: List[Dict[str, object]] = []
    if suggested_trim is not None:
        marker_specs.extend(
            [
                {
                    "timestamp": float(suggested_trim["start_timestamp"]),
                    "color": "tab:green",
                    "linestyle": "--",
                    "linewidth": 1.5,
                    "label": "trim_start",
                },
                {
                    "timestamp": float(suggested_trim["end_timestamp"]),
                    "color": "tab:orange",
                    "linestyle": "--",
                    "linewidth": 1.5,
                    "label": "trim_end",
                },
            ]
        )

    if vertical_speed_analysis is not None:
        candidate = vertical_speed_analysis.get("flat_period_candidate")
        if isinstance(candidate, dict):
            candidate_markers = [
                {
                    "timestamp": float(candidate["start_timestamp"]),
                    "color": "tab:purple",
                    "linestyle": ":",
                    "linewidth": 1.2,
                    "label": "flat_start",
                },
                {
                    "timestamp": float(candidate["end_timestamp"]),
                    "color": "tab:brown",
                    "linestyle": ":",
                    "linewidth": 1.2,
                    "label": "descent_start",
                },
            ]
            for marker in candidate_markers:
                if any(abs(float(existing["timestamp"]) - float(marker["timestamp"])) < 1e-3 for existing in marker_specs):
                    continue
                marker_specs.append(marker)

    for marker in marker_specs:
        for axis in axes:
            axis.axvline(
                float(marker["timestamp"]),
                color=str(marker["color"]),
                linestyle=str(marker["linestyle"]),
                linewidth=float(marker["linewidth"]),
            )

    for idx, marker in enumerate(marker_specs):
        axes[0].annotate(
            f"{marker['label']}\n{float(marker['timestamp']):.6f}",
            xy=(float(marker["timestamp"]), 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(0, -4 - 52 * (idx % 2)),
            textcoords="offset points",
            rotation=90,
            ha="center",
            va="top",
            fontsize=8,
            color=str(marker["color"]),
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "white",
                "edgecolor": str(marker["color"]),
                "alpha": 0.8,
            },
            clip_on=False,
        )

    plt.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    bag_path = infer_bag_path(args.dataset_root, args.sequence, args.bag_path)
    if not bag_path.is_file():
        raise FileNotFoundError(f"Bag file not found: {bag_path}")

    output_dir = infer_output_dir(args.dataset_root, args.sequence, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    series_map = read_topics(
        bag_path=bag_path,
        height_topic=args.height_topic,
        gps_topic=args.gps_topic,
        local_position_topic=args.local_position_topic,
        velocity_topic=args.velocity_topic,
        imu_topic=args.imu_topic,
        include_imu=not args.skip_imu,
        progress_every=args.progress_every,
    )

    trim_signal_used, altitude_trim_candidate, candidate_trim_summaries = choose_trim_signal(
        requested_signal=args.trim_signal,
        series_map=series_map,
        min_height=args.min_height,
        hold_seconds=args.hold_seconds,
        moving_average_seconds=args.moving_average_seconds,
    )
    vertical_speed_analysis = analyze_vertical_speed(
        speed_series=series_map["vertical_speed_z"],
        moving_average_seconds=args.moving_average_seconds,
        flat_speed_threshold=args.flat_speed_threshold,
        motion_speed_threshold=args.motion_speed_threshold,
        hold_seconds=args.speed_hold_seconds,
    )
    suggested_trim, speed_trim_candidate, trim_strategy_used = tighten_trim_window_with_vertical_speed(
        suggested_trim=altitude_trim_candidate,
        vertical_speed_analysis=vertical_speed_analysis,
        reference_timestamp=first_timestamp(series_map),
    )

    plot_path = output_dir / f"{bag_path.stem}_flight_profile.png"
    summary_path = output_dir / f"{bag_path.stem}_flight_profile.json"
    csv_path = output_dir / f"{bag_path.stem}_flight_profile.csv"

    plot_series(
        output_path=plot_path,
        bag_path=bag_path,
        series_map=series_map,
        suggested_trim=suggested_trim,
        vertical_speed_analysis=vertical_speed_analysis,
        moving_average_seconds=args.moving_average_seconds,
        min_height=args.min_height,
    )

    summary = {
        "bag_path": str(bag_path),
        "plot_path": str(plot_path),
        "trim_signal_requested": args.trim_signal,
        "trim_signal_used": trim_signal_used,
        "trim_strategy_used": trim_strategy_used,
        "altitude_trim_candidate": altitude_trim_candidate,
        "speed_trim_candidate": speed_trim_candidate,
        "suggested_trim": suggested_trim,
        "candidate_trim_summaries": candidate_trim_summaries,
        "vertical_speed_analysis": vertical_speed_analysis,
        "series_stats": {
            key: {
                "count": int(series.timestamps.size),
                "start_timestamp": None if series.empty else float(series.timestamps[0]),
                "end_timestamp": None if series.empty else float(series.timestamps[-1]),
                "min_value": None if series.empty else float(np.min(series.values)),
                "max_value": None if series.empty else float(np.max(series.values)),
            }
            for key, series in series_map.items()
        },
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    if args.write_csv:
        rows = merge_series_for_csv(series_map, args.csv_rate_hz)
        fieldnames = ["timestamp"] + list(series_map.keys())
        write_csv(csv_path, rows, fieldnames)

    print(f"Saved plot to: {plot_path}")
    print(f"Saved summary to: {summary_path}")
    if args.write_csv:
        print(f"Saved CSV to: {csv_path}")

    if suggested_trim is None:
        print("No trim window was suggested. Try lowering --min-height or --hold-seconds.")
    else:
        print(
            "Suggested trim window: "
            f"{suggested_trim['start_timestamp']:.6f} -> {suggested_trim['end_timestamp']:.6f} "
            f"(signal: {trim_signal_used}, strategy: {trim_strategy_used})"
        )

    if vertical_speed_analysis is not None:
        for event in vertical_speed_analysis.get("post_positive_flat_events", [])[:10]:
            print(
                "Vertical speed returned near 0 after climb at: "
                f"{event['flat_start_timestamp']:.6f}"
            )
        for event in vertical_speed_analysis.get("pre_negative_motion_events", [])[:10]:
            print(
                "Vertical speed left flat and started descending at: "
                f"{event['motion_start_timestamp']:.6f}"
            )


if __name__ == "__main__":
    main()
