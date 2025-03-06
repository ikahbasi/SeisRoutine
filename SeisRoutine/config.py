import yaml
import logging


class Config:
    def __init__(self, **entries):
        for key, value in entries.items():
            if isinstance(value, dict):
                value = Config(**value)
            setattr(self, key, value)

    def __str__(self):
        text = ''
        for key, val in self.__dict__.items():
            text += f'{key}: {val}\n'
        return text

def load_config(file_path):
    with open(file_path, 'r') as file:
        config_dict = yaml.safe_load(file)
        return Config(**config_dict)

def configure_logging(level,
                      log_format='%(asctime)s - %(levelname)s - %(message)s',
                      mode='console',
                      filename='app.log'):
    """
    Configure logging settings based on mode.

    Parameters:
    level (str): Logging level.
    mode (str): Mode of logging - 'console', 'file', or 'both'.
    filename (str): The filename for logging to a file (if needed).
    """
    if not isinstance(level, int):
        numeric_level = getattr(logging, level.upper(), None)
    else:
        numeric_level = level
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
        
    logger = logging.getLogger()
    logger.setLevel(numeric_level)
    if logger.hasHandlers():
        logger.handlers.clear()
    
    if mode == 'console' or mode == 'both':
        console_handler = logging.StreamHandler()
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(console_handler)
    
    if mode == 'file' or mode == 'both':
        file_handler = logging.FileHandler(filename)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_handler)
    
    logger.propagate = False
    for name in logging.root.manager.loggerDict.keys():
        if name not in ('my_module', '__main__'):
            logging.getLogger(name).setLevel(logging.WARNING)
