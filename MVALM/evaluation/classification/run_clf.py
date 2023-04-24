import sys

sys.path.append('/home/t9s9/PycharmProjects/Multimodal-VAL-Models/')

from MVALM.evaluation.classification.config import ex
from MVALM.evaluation.classification.trainer import train


# @ex.automain
# def main(_config, _log):
#     train(_config, _log)


@ex.main
def main(_config, _log):
    train(_config, _log)


if __name__ == '__main__':
    from ray.tune.search.variant_generator import generate_variants
    import ray.tune as tune

    search_conf = {
        'freeze_backbone': tune.grid_search([True]),
        'model': {
            'momentum': tune.grid_search([True]),
            'path': tune.grid_search([
                # "/home/t9s9/Datasets/ckpt/VAT-Distill/65mghu8x/65mghu8x-epoch=3-step=3860-val_loss=10.900.ckpt",
                "/home/t9s9/Datasets/ckpt/AT-Distill/zz9jwtyo_Distill/zz9jwtyo-epoch=3-step=3860-val_loss=2.561.ckpt"
            ]),
        },
    }

    for i, config in enumerate(generate_variants(search_conf)):
        config_updates = config[1]
        ex.run('main', config_updates=config_updates, named_configs=['fsd'])
