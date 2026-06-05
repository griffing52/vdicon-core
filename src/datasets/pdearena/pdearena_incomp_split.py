from __future__ import annotations

from .pdearena_incomp import PDEArenaIncompDataset


class PDEArenaIncompSplitDataset(PDEArenaIncompDataset):
    def __init__(self, file_paths: list[str], ex_num: int = 5, split: str = "train"):
        super().__init__(file_paths=file_paths, ex_num=ex_num, split=split)