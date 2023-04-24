from pathlib import Path
from typing import Union, Dict, Optional

import pandas as pd
from torch import load as load_pt
from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):
    """
    Simple wrapper for a dataset that contains embeddings that are saved as .pt files.
    """

    def __init__(self,
                 annot: Union[str, Path],
                 root: Union[str, Path],
                 expand_on: Optional[str] = None
                 ):
        self.root = Path(root)
        self.annot_p = Path(annot)
        self.annot = pd.read_parquet(self.annot_p)

        self.annot.filename = self.annot.filename.apply(lambda x: Path(x).with_suffix('.pt'))
        exists = self.annot.filename.apply(lambda x: (self.root / x).exists())
        print(f"Missing embeddings {self.annot_p.parent.name}:", len(self.annot) - exists.sum())
        self.annot = self.annot[exists]
        print(f"Remaining embeddings {self.annot_p.parent.name}:", len(self.annot))

        if expand_on is not None:
            self.annot = self.annot.explode(expand_on)

    def __len__(self):
        return len(self.annot)

    def __getitem__(self, i) -> Dict:
        annot = self.annot.iloc[i].to_dict()
        annot['embedding'] = load_pt(self.root / annot['filename'])
        return annot
