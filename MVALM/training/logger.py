import os
from pathlib import Path
from typing import Dict

import wandb
from dotenv import load_dotenv
from pytorch_lightning.loggers import WandbLogger, Logger, CSVLogger, TensorBoardLogger
from pytorch_lightning.loggers.logger import DummyLogger


def login_wandb():
    load_dotenv(verbose=True)
    secret_key = os.environ.get("WANDB_KEY")
    assert secret_key is not None

    os.environ["WANDB_START_METHOD"] = "thread"
    return wandb.login(key=secret_key, anonymous='allow', relogin=None)


def get_wandb_logger(config: Dict,
                     project: str,
                     id: str = None,
                     save_dir: str = None,
                     name: str = None,
                     store_checkpoints: bool = False,
                     group: str = None,
                     resume: bool = None,
                     notes: str = ""):
    login_wandb()
    entity = os.environ.get("ENTITY")
    save_dir = save_dir if save_dir is not None else '.'
    id = id if id is not None else wandb.util.generate_id()
    if isinstance(resume, str):
        resume = 'must'
    return WandbLogger(save_dir=save_dir,
                       name=name,
                       id=id,
                       project=project,
                       entity=entity,
                       config=config,
                       log_model=store_checkpoints,
                       group=group,
                       notes=notes,
                       resume=resume)


def get_logger(config: Dict) -> Logger:
    if config['debug'] or not config['logger']:
        return DummyLogger()
    else:
        if config['resume_from_checkpoint'] is not None:
            id = Path(config['resume_from_checkpoint']).name[:8]
        else:
            id = config['id']

        if config['logger'] == 'wandb':
            return get_wandb_logger(config,
                                    id=id,
                                    name=config['exp_name'],
                                    project=config['project'],
                                    group=config['group'],
                                    resume=config['resume_from_checkpoint'])
        elif config['logger'] == 'csv':
            return CSVLogger(save_dir=str(Path(config['checkpoint_path'])),
                             name=f"logs_{config['exp_name']}")
        elif config['logger'] == 'tensorboard':
            logger = TensorBoardLogger(save_dir=str(Path(config['checkpoint_path'])),
                                       name=f"logs_{config['exp_name']}")
            logger.log_hyperparams(config)

            return logger
        else:
            raise ValueError(f"Unknown logger: {config['logger']}. Use wandb or csv.")
