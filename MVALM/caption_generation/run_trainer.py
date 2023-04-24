import sys

sys.path.append('/home/t9s9/PycharmProjects/Multimodal-VAL-Models')
sys.path.append('/home/schaumloeffel/Multimodal-VAL-Models')

from config import ex
from trainer import train
from model import PrexfixMappingType
from ray.tune.search.variant_generator import generate_variants
import ray.tune as tune


@ex.automain
def main(_config, _log):
    train(_config, _log)


# @ex.main
# def main(_config, _log):
#     train(_config, _log)


# if __name__ == '__main__':
#     search_conf = {
#         'train_gpt': False,
#         'model': {
#             'prefix_mapping_type': PrexfixMappingType.MLP,
#             'const_prefix_length': 10,
#             'encoder_kwargs': {
#                 'freeze': tune.grid_search([True, False]),
#             }
#         },
#     }
#
#     for i, config in enumerate(generate_variants(search_conf)):
#         config_updates = config[1]
#         ex.run('main', config_updates=config_updates, named_configs=['server', 'tag_transformer'])
#
#     search_conf = {
#         'train_gpt': True,
#         'model': {
#             'prefix_mapping_type': PrexfixMappingType.MLP,
#             'const_prefix_length': tune.grid_search([1, 5, 20]),
#             'encoder_kwargs': {
#                 'freeze': False,
#             }
#         },
#     }
#
#     for i, config in enumerate(generate_variants(search_conf)):
#         config_updates = config[1]
#         ex.run('main', config_updates=config_updates, named_configs=['server', 'tag_transformer'])
