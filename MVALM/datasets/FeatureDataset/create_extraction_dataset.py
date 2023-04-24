from collections import defaultdict
from MVALM.datasets import MSVD, AudioCaps, MSRVTT, VaTeX, VGGSound, ActivityNetDense
from MVALM.datasets.FeatureDataset.write_extraction_dataset import write_extraction_dataset


def create_extracted_beton():
    number_images = 1
    image_pick_strategy = 'linspace'
    token_length = 77
    spectrogram_width = 1000
    output_file = "/data/mmssl/beton/VAT_{split}.beton"

    datasets = defaultdict(list)
    available_datasets = [MSVD, AudioCaps, MSRVTT, VaTeX, VGGSound, ActivityNetDense]

    for dataset_cls in available_datasets:
        labels_type = 'caption' if 'caption' in dataset_cls.LABELS else 'classification'
        for split in dataset_cls.LABELS[labels_type].keys():
            datasets[split].append(dataset_cls.as_extraction_dataset(feature_dir='extraction-10s',
                                                                     split=split,
                                                                     number_images=number_images,
                                                                     image_pick_strategy=image_pick_strategy))
    print(datasets)

    for split, split_datasets in datasets.items():
        write_extraction_dataset(split_datasets,
                                 output_path=output_file.format(split=split),
                                 split=split,
                                 nr_images=number_images,
                                 token_length=token_length,
                                 spectrogram_width=spectrogram_width)
