# logger.py
import logging
from options import args_parser
args = args_parser()

import os

def get_logger(name=None, log_file=None, level=logging.INFO):
    """
    Returns a logger that logs to console and optionally to a file.
    If log_file is None, only console logging is enabled.
    """
    # Create logs folder if it doesn't exist
    if log_file is not None:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger_name = name if name is not None else 'default_logger'
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    if not logger.handlers:
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # File handler
        if log_file is not None:
            fh = logging.FileHandler(log_file, mode='w')
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    return logger
