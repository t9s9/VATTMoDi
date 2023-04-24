import sys

# sys.path.append('/home/schaumloeffel/Multimodal-VAL-Models')
sys.path.append('/home/t9s9/PycharmProjects/Multimodal-VAL-Models/')
from tqdm import tqdm
import torch
import pandas as pd
from torch.utils.data import DataLoader
from MVALM.datasets import VaTeX, ActivityNetDense, MSVD, MSRVTT, VGGSound, DiDeMo, AudioSet
from module import CaptioningModule
from dataset import AudioCaptionOutput

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    # ckpt_path = '/data/mmssl/ckpt/25zidru8_transformer_embedding-tag_train_gpt-epoch=4-val_loss=2.508.ckpt'
    ckpt_path = '/home/t9s9/Datasets/ckpt/Captioning/Tag/25zidru8_transformer_embedding-tag_train_gpt-epoch=4-val_loss=2.508.ckpt'
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    sd = {}
    for key, value in checkpoint['state_dict'].items():
        if 'gpt' in key:
            sd[key.replace('gpt', 'caption_model')] = checkpoint['state_dict'][key]
        else:
            sd[key] = checkpoint['state_dict'][key]

    sd['head.wte.weight'] = sd['head.caption_model.transformer.wte.weight']
    module = CaptioningModule(**checkpoint['hyper_parameters'])
    module.load_state_dict(sd)
    module = module.to(device).eval()
    for block in module.head.prefix_proj.transformer.blocks:
        block.attention.set_use_memory_efficient_attention_xformers(False)
        block.attention.store_attn = False


    def collate_fn(batch):
        # assert isinstance(batch[0]['labels'], (list, ))
        return AudioCaptionOutput(audio_embedding=batch[0]['embeddings'], filename=batch[0]['filename'],
                                  spectrogram=None, caption=None, tags=batch[0]['labels'])


    datasets = [AudioSet]

    for dataset_cls in datasets:
        for split, ds in dataset_cls.by_splits(label_type='classification',
                                               datasets_root='/home/t9s9/Datasets/',
                                               # datasets_root='/data/mmssl'
                                               ).items():
            if split != 'val':
                continue

            ds = ds.add_source('embeddings')
            dl = DataLoader(ds, batch_size=1, num_workers=2, collate_fn=collate_fn)
            print(ds.name, split, len(ds), len(dl))

            result = []
            for dp in tqdm(dl, desc=f'{ds.name} {split}'):
                caption = module.infer(dp, strategy='beam', beam_size=5, entry_length=60, temperature=1.2, top_p=0.8)
                result.append({'filename': dp.filename, 'caption': caption})

            pd.DataFrame(result).to_parquet(ds.root / f'annot_auditory_{split}.parquet', index=False)
