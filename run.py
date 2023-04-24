from MVALM.config.vat import ex
from MVALM.training import train
from MVALM.config.at import ex as ex_at


@ex.automain
def main(_config, _log):
    train(_config, _log)
