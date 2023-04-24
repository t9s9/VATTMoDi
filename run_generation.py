from MVALM.generation.config import ex
from MVALM.generation.trainer import train


@ex.automain
def main(_config, _log):
    train(_config, _log)
