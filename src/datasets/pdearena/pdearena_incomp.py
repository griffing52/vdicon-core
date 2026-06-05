from __future__ import annotations

import math
import glob
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class _TrajectoryRef:
    file_path: str
    traj_idx: int


class PDEArenaIncompDataset(Dataset):
    """Incompressible PDEArena Navier-Stokes dataset for 2D image baselines."""

    def __init__(
        self,
        file_paths: str | list[str],
        ex_num: int = 5,
        split: str = "train",
        split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
        input_time_offset: int = 0,
        target_time_offset: int = 1,
        time_stride: int = 1,
        base_seed: int = 0,
    ) -> None:
        super().__init__()
        if isinstance(file_paths, str):
            file_paths = sorted(glob.glob(file_paths)) or [file_paths]

        self.file_paths = [str(Path(path)) for path in file_paths]
        self.ex_num = ex_num
        self.split = split
        self.split_ratios = split_ratios
        self.input_time_offset = input_time_offset
        self.target_time_offset = target_time_offset
        self.time_stride = time_stride
        self.base_seed = base_seed

        self._file_handles: dict[str, h5py.File] = {}
        self._trajectory_refs: list[_TrajectoryRef] = []
        self._time_steps: int | None = None

        for file_path in self.file_paths:
            with h5py.File(file_path, "r") as file_handle:
                if "velocity" not in file_handle:
                    raise KeyError(f"Missing 'velocity' dataset in {file_path}")
                traj_num, time_steps = file_handle["velocity"].shape[:2]
                if self._time_steps is None:
                    self._time_steps = time_steps
                elif self._time_steps != time_steps:
                    raise ValueError("All PDEArena files must share the same time dimension")
                for traj_idx in range(traj_num):
                    self._trajectory_refs.append(_TrajectoryRef(file_path=file_path, traj_idx=traj_idx))

        if not self._trajectory_refs:
            raise ValueError(f"No trajectories found in {self.file_paths}")

        self._split_refs = self._split_trajectory_refs()
        self._samples = self._build_samples()

    def _split_trajectory_refs(self) -> list[_TrajectoryRef]:
        total = len(self._trajectory_refs)
        if total < 3:
            return self._trajectory_refs

        train_ratio, valid_ratio, test_ratio = self.split_ratios
        train_count = max(1, int(math.floor(total * train_ratio)))
        valid_count = max(1, int(math.floor(total * valid_ratio)))
        test_count = total - train_count - valid_count

        if test_count < 1:
            test_count = 1
            if train_count > valid_count:
                train_count = max(1, total - valid_count - test_count)
            else:
                valid_count = max(1, total - train_count - test_count)

        split_bounds = {
            "train": (0, train_count),
            "valid": (train_count, train_count + valid_count),
            "test": (train_count + valid_count, total),
        }
        if self.split not in split_bounds:
            raise ValueError(f"Split must be 'train', 'valid', or 'test', got {self.split}")

        begin, end = split_bounds[self.split]
        split_refs = self._trajectory_refs[begin:end]
        if not split_refs:
            raise ValueError(f"Split '{self.split}' is empty for {total} trajectories")
        return split_refs

    def _build_samples(self) -> list[tuple[int, int]]:
        assert self._time_steps is not None
        max_input_time = self._time_steps - self.target_time_offset
        if max_input_time <= self.input_time_offset:
            raise ValueError("Not enough time steps for the requested offsets")

        samples: list[tuple[int, int]] = []
        for traj_idx in range(len(self._split_refs)):
            for time_idx in range(self.input_time_offset, max_input_time, self.time_stride):
                samples.append((traj_idx, time_idx))
        return samples

    def _get_file_handle(self, file_path: str) -> h5py.File:
        if file_path not in self._file_handles:
            self._file_handles[file_path] = h5py.File(file_path, "r")
        return self._file_handles[file_path]

    def _read_velocity_frame(self, ref: _TrajectoryRef, time_idx: int) -> torch.Tensor:
        file_handle = self._get_file_handle(ref.file_path)
        velocity = file_handle["velocity"][ref.traj_idx, time_idx]
        return torch.tensor(velocity, dtype=torch.float32).permute(2, 0, 1)

    def _sample_context_indices(self, qn_traj_idx: int, sample_idx: int) -> list[int]:
        candidate_indices = [idx for idx in range(len(self._split_refs)) if idx != qn_traj_idx]
        if not candidate_indices:
            return [qn_traj_idx] * self.ex_num

        generator = torch.Generator()
        generator.manual_seed(self.base_seed + sample_idx)
        perm = torch.randperm(len(candidate_indices), generator=generator).tolist()

        if len(candidate_indices) >= self.ex_num:
            return [candidate_indices[i] for i in perm[: self.ex_num]]

        chosen = [candidate_indices[i] for i in perm]
        while len(chosen) < self.ex_num:
            chosen.extend(candidate_indices)
        return chosen[: self.ex_num]

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int):
        qn_traj_idx, time_idx = self._samples[idx]
        target_time_idx = time_idx + self.target_time_offset
        context_traj_indices = self._sample_context_indices(qn_traj_idx, idx)

        ex_f = torch.stack([self._read_velocity_frame(self._split_refs[traj_idx], time_idx) for traj_idx in context_traj_indices])
        ex_g = torch.stack(
            [self._read_velocity_frame(self._split_refs[traj_idx], target_time_idx) for traj_idx in context_traj_indices]
        )
        qn_f = self._read_velocity_frame(self._split_refs[qn_traj_idx], time_idx).unsqueeze(0)
        qn_g = self._read_velocity_frame(self._split_refs[qn_traj_idx], target_time_idx).unsqueeze(0)

        data = {
            "ex_f": ex_f.unsqueeze(0),
            "ex_g": ex_g.unsqueeze(0),
            "qn_f": qn_f.unsqueeze(0),
        }

        description = (
            f"PDEArena incompressible NS, split={self.split}, sample={idx}, "
            f"qn_traj={qn_traj_idx}, time_idx={time_idx}, target_time_idx={target_time_idx}"
        )

        return {
            "description": np.array([description], dtype=np.dtypes.StringDType()),
            "data": data,
            "label": qn_g.unsqueeze(0),
        }

    def __del__(self):
        for file_handle in self._file_handles.values():
            if file_handle is not None:
                file_handle.close()