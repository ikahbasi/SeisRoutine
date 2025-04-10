import yaml
import logging


class Config:
    """
    A helper class that converts a dictionary (e.g., parsed from a YAML file)
    into an object with attribute access. Supports nested dictionaries and 
    includes utilities for converting back to dictionary form and string representation.
    """

    def __init__(self, **entries):
        """
        Initializes the Config object with dictionary entries.

        Args:
            **entries: Arbitrary keyword arguments representing dictionary keys and values.
        """
        self.entries = entries
        self.dict2object()

    def dict2object(self):
        """
        Recursively sets dictionary keys as attributes on the object.
        Nested dictionaries are converted into nested Config objects.
        """
        for key, value in self.entries.items():
            if isinstance(value, dict):
                value = Config(**value)
            setattr(self, key, value)
    
    def to_dict(self):
        """
        Recursively converts the Config object back into a dictionary.

        Returns:
            dict: A dictionary representation of the Config object.
        """
        result = {}
        for key, value in self.__dict__.items():
            if key == 'entries':
                continue
            if isinstance(value, Config):
                value = value.to_dict()
            elif isinstance(value, list):
                value = [v.to_dict() if isinstance(v, Config) else v for v in value]
            result[key] = value
        return result
    
    def to_yaml(self, **yaml_kwargs):
        """
        Converts the Config object to a YAML-formatted string.

        Args:
            **yaml_kwargs: Additional keyword arguments to pass to yaml.dump.

        Returns:
            str: A YAML-formatted string.
        """
        return yaml.dump(self.to_dict(), **yaml_kwargs)

    def __str__(self):
        """
        Returns the Config object as a YAML-formatted string.
        """
        return self.to_yaml(default_flow_style=False)

    def __repr__(self):
        """
        Returns a developer-friendly representation of the Config object.

        Returns:
            str: A string representation suitable for debugging.
        """
        return f'Config({self.__dict__})'

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
