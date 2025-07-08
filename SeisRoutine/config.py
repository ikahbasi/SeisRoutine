from datetime import datetime
import logging
import os
import sys
import yaml
import ipynbname
import coloredlogs


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
    """
    Docstring
    """
    with open(file_path, 'r') as file:
        config_dict = yaml.safe_load(file)
        return Config(**config_dict)

def configure_logging(level,
                      log_format='%(asctime)s - [%(levelname)s] - %(message)s',
                      mode='console', colored_console=True,
                      filename_prefix='', filename='app.log', filepath='.'):
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

    if mode in ('console', 'both'):
        if colored_console:
            coloredlogs.install(level=numeric_level, fmt=log_format, logger=logger,
                                level_styles={
                                    'debug': {'color': 'blue'},
                                    'info': {'color': 'green'},
                                    'warning': {'color': 'yellow'},
                                    'error': {'color': 'red'},
                                    'critical': {'color': 'magenta', 'bold': True},
                                })
        else:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(numeric_level)
            console_handler.setFormatter(logging.Formatter(log_format))
            logger.addHandler(console_handler)

    if mode in ('file', 'both'):
        if filename == 'now':
            today_str = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
            filename = f'{today_str}.log'
        os.makedirs(filepath, exist_ok=True)
        filename = f'{filename_prefix}_{filename}'
        filename = os.path.join(filepath, filename)
        file_handler = logging.FileHandler(filename)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_handler)

    logger.propagate = False
    for name in logging.root.manager.loggerDict.keys():
        if name not in ('my_module', '__main__'):
            logging.getLogger(name).setLevel(logging.WARNING)


def getting_filename_and_path_of_the_running_code():
    """
    Get the filename and directory path of the currently executing code.
    
    This function works for both regular Python scripts (.py files) and Jupyter Notebooks
    (.ipynb files). For notebooks, it handles both VS Code's environment and standard
    Jupyter environments.

    Returns:
        tuple: A tuple containing (directory_path, filename) of the running code.
        
    Note:
        In Jupyter Notebook environments, returns the notebook name and path.
        In regular Python scripts, returns the script name and path.
    """
    _file = sys.argv[0]
    name = os.path.basename(_file)
    path = os.path.dirname(_file)
    if name == "ipykernel_launcher.py":
        try:
            _file = globals()['__vsc_ipynb_file__']
            name = os.path.basename(_file)
            path = os.path.dirname(_file)
        except Exception as error:
            print(error)
            name = ipynbname.name()
            path = ipynbname.path()
    return path, name
