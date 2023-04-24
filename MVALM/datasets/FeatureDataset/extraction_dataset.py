from pathlib import Path
from typing import Optional, Union, List, NamedTuple, Callable, Dict, Type

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


def collect_image_positions(path_to_images, output_path) -> pd.DataFrame:
    """ Collect all images belonging to one video and saves the positions in a parquet file."""
    path = Path(path_to_images)
    print("Collecting image positions from", path)

    files = list(map(lambda p: (p.stem[:-5], int(p.stem[-3:])), path.iterdir()))
    df = pd.DataFrame(files, columns=['filename', 'position'])
    df['position'] = df['position'].astype('uint8')
    df = df.groupby('filename').agg(list)
    df.position = df.position.apply(lambda x: sorted(x))
    df.to_parquet(output_path)
    return df


def create_extracted_dataset(dataset_cls,
                             dataset_kwargs: Dict = {},
                             extraction_name: str = 'extraction-10s',
                             min_num_img: int = 5, ):
    labels_type = 'caption' if 'caption' in dataset_cls.LABELS else 'classification'
    for split in dataset_cls.LABELS[labels_type].keys():
        print("Split", split)
        dataset = dataset_cls(split=split, **dataset_kwargs)
        extraction_dir = dataset.root / extraction_name

        # get image positions
        image_positions_path = extraction_dir / f"{split}_image_ids.parquet"
        # if image_positions_path.exists():
        #     image_positions = pd.read_parquet(image_positions_path)
        # else:
        image_positions = collect_image_positions(extraction_dir / 'images' / split, image_positions_path)
        print(f"Found {len(image_positions)} image positions for {split}.")

        # get dataset annotations
        annot = dataset.annot
        annot.filename = annot.filename.apply(lambda s: Path(s).stem)
        annot = annot.set_index('filename')
        print(f"Found {len(annot)} annotations for {split}.")

        # join image positions and annotations
        annot = annot.join(image_positions, how='inner', validate='1:1')

        auditory_captions_path = dataset.root / "captions" / f"auditory_captions.csv"
        if auditory_captions_path.exists():
            print("Found auditory captions.")
            # the csv contains data for all splits, so we have to get the correct split
            auditory_cap = pd.read_csv(auditory_captions_path)

            auditory_cap[['split', 'filename']] = auditory_cap.apply(lambda x: x.filename.split('/'), axis=1,
                                                                     result_type='expand')
            auditory_cap = auditory_cap[auditory_cap['split'] == split].copy()
            auditory_cap.filename = auditory_cap.filename.apply(lambda s: Path(s).stem)
            auditory_cap = auditory_cap.set_index('filename').drop(columns='split').rename(
                columns={'caption': 'auditory_captions'})
            print(f"Found {len(auditory_cap)} auditory annotations for {split}.")

            annot = annot.rename(columns={'captions': 'visual_captions'})
            annot = annot.join(auditory_cap, how='inner', validate='1:1')

        visual_captions_path = dataset.root / "captions" / f"visual_captions_{split}.parquet"
        if visual_captions_path.exists():
            print("Found visual captions.")
            visual_cap = pd.read_parquet(visual_captions_path)
            visual_cap = visual_cap.rename(columns={'captions': 'visual_captions'})
            print(f"Found {len(visual_cap)} visual annotations for {split}.")

            annot = annot.rename(columns={'captions': 'auditory_captions'})
            annot = annot.join(visual_cap, how='inner', validate='1:1')

        if not {'position', 'visual_captions', 'auditory_captions'}.issubset(annot.columns):
            raise ValueError(f"Missing columns in annotations: {annot.columns}")

        if isinstance(min_num_img, int):
            annot = annot[annot.position.apply(len) >= min_num_img]

        print(f"Found {len(annot)} annotations for {split} after filtering.")
        annot.to_parquet(extraction_dir / f"full_annot_{split}.parquet")


class ExtractionOutput(NamedTuple):
    images: List[Image.Image]
    spectrogram: np.ndarray
    visual_captions: List[str]
    auditory_captions: List[str]


class ExtractionDataset(Dataset):
    def __init__(self,
                 root: Union[Path, str],
                 split: str,
                 number_images: int = 3,
                 image_pick_strategy: str = 'linspace',
                 transform: Optional[Callable] = None,
                 seed: int = 42, ):
        np.random.seed(seed)
        self.root = Path(root)
        self.split = split

        self.transform = transform
        self.number_images = number_images
        self.image_pick_strategy = image_pick_strategy

        self.annot = pd.read_parquet(self.root / f'full_annot_{split}.parquet')
        self.image_template = str(self.root / 'images' / f'{split}' / '{filename}_i{i:03d}.jpg')
        self.torch_template = str(self.root / '{file_type}' / f'{split}' / '{filename}.pt')

    def __repr__(self):
        return f"{self.root.parent.name}-{self.root.name} ({self.split})"

    def __str__(self):
        return self.__repr__()

    @staticmethod
    def has_split(root, split) -> bool:
        return (root / f'full_annot_{split}.parquet').exists()

    def __len__(self) -> int:
        return len(self.annot)

    def __getitem__(self, idx: int) -> ExtractionOutput:
        datapoint = self.annot.iloc[idx]

        spectrogram = torch.load(self.torch_template.format(file_type='spectrogram', filename=datapoint.name))
        spectrogram = spectrogram.numpy()

        visual_captions = [datapoint.visual_captions] if isinstance(datapoint.visual_captions,
                                                                    str) else datapoint.visual_captions
        auditory_captions = [datapoint.auditory_captions] if isinstance(datapoint.auditory_captions,
                                                                        str) else datapoint.auditory_captions

        if self.image_pick_strategy == 'linspace':
            image_indices = np.linspace(1, max(datapoint.position), self.number_images, dtype=int)
        elif self.image_pick_strategy == 'random':
            image_indices = np.random.choice(max(datapoint.position), self.number_images, replace=False)
        else:
            raise ValueError(f'Unknown image pick strategy {self.image_pick_strategy}')

        images = []
        for i in image_indices:
            images.append(Image.open(self.image_template.format(filename=datapoint.name, i=i)))

        out = ExtractionOutput(images=images,
                               spectrogram=spectrogram,
                               visual_captions=visual_captions,
                               auditory_captions=auditory_captions)
        if self.transform is not None:
            out = self.transform(out)
        return out
