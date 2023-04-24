import pandas as pd
from pathlib import Path


def prepare_annot():
    root = Path("/data/mmssl/VGGSound/")
    df = pd.read_csv(root / 'vggsound.csv')

    df['filename'] = df.apply(
        lambda row: str(Path(row['split']) / f"{row['youtube_id']}_{row['time_start']:06d}_{row['time_end']:06d}.mp4"),
        axis=1)
    df = df.drop(columns=['time_start', 'time_end', 'youtube_id'])
    df = df.rename({'caption': 'target'}, axis=1)
    df.target = df.target.astype('category')
    df.split = df.split.astype('category')
    df = df[df.filename.apply(lambda x: (root / 'data' / x).exists())]

    for split, group in df.groupby('split', as_index=False):
        print(f"Split {split}: {group.shape[0]} datapoints.")
        assert group.shape[0] == len(list((root / 'data' / split).iterdir()))
        group = group.drop(columns='split')
        group.to_parquet(root / f'annot_{split}.parquet', index=False)


def clean():
    root = Path("/data/mmssl/VGGSound/")
    train_df = pd.read_parquet(root / 'annot_train.parquet')
    test_df = pd.read_parquet(root / 'annot_test.parquet')
    train_df.filename = train_df.filename.apply(lambda x: Path(x).stem)
    test_df.filename = test_df.filename.apply(lambda x: Path(x).stem)
    meta = pd.read_csv(root / 'extracted_metadata.csv')

    corrupted_train = set(train_df.filename).difference(set(meta.filename))
    corrupted_train = list(map(lambda x: str('train' / Path(x).with_suffix('.mp4')), corrupted_train))
    corrupted_test = set(test_df.filename).difference(set(meta.filename))
    corrupted_test = list(map(lambda x: str('test' / Path(x).with_suffix('.mp4')), corrupted_test))
    corrupted = corrupted_train + corrupted_test
    print(f"Corrupted: {len(corrupted)}")

    for corrupted_vid_path in corrupted:
        (root / 'data' / corrupted_vid_path).unlink()


if __name__ == '__main__':
    root = Path("/data/mmssl/VGGSound/")
    train_df = pd.read_parquet(root / 'annot_train.parquet')
    test_df = pd.read_parquet(root / 'annot_test.parquet')

    print(train_df.shape, test_df.shape)
