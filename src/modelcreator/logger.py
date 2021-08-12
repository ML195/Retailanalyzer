import logging
from pathlib import Path

# Setup a simple info logger
def setup_info_logger():
    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

    log_file_path = Path(__file__).parents[1].resolve() / 'logs' / 'info.log'

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(formatter)

    logger = logging.getLogger('info')
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    return logger