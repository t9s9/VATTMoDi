from MVALM.config.at_distill import ex
# from MVALM.config.vat_distill import ex
from MVALM.training.trainer_distill import train
from ray.tune.search.variant_generator import generate_variants
import ray.tune as tune


@ex.automain
def main(_config, _log):
    train(_config, _log)


# @ex.main
# def main(_config, _log):
#     train(_config, _log)
#
#
# if __name__ == '__main__':
#     search_conf = {
#         # 'queue_size': tune.grid_search([4096, 16384, 32768]),
#         'queue_size': 4096,
#         'distill': tune.grid_search([False]),
#         'momentum': tune.grid_search([0.995]),
#         'alpha': tune.grid_search([0.4]),
#     }
#
#     for i, config in enumerate(generate_variants(search_conf)):
#         config_updates = config[1]
#         ex.run('main', config_updates=config_updates, named_configs=['server', 'audiocaps'])
