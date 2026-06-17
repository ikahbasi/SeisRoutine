from datetime import datetime
from pathlib import Path
import logging
import os
import sys
import yaml
import coloredlogs
import platform
import subprocess
from importlib.metadata import distributions


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
    print(f"Logging Starts in:\n{filename}")


def get_running_file_info():
    """
    Return the path of the main script or notebook being executed.

    Examples
    --------
    >>> file_path = get_running_file()
    >>> file_path.name      # Filename with extension
    >>> file_path.stem      # Filename without extension
    >>> file_path.parent    # Directory path
    >>> str(file_path)      # Full path
    """

    try:
        return Path(__vsc_ipynb_file__).resolve()

    except NameError:

        file_path = Path(sys.argv[0]).resolve()

        if file_path.name == "ipykernel_launcher.py":
            import ipynbname
            file_path = Path(ipynbname.path()).resolve()

        return file_path


class EnvironmentInfo:
    """
    Collect information about the current execution environment.
    """

    def __init__(self):

        self._installed_packages = None
        self._imported_packages = None

    @staticmethod
    def system():
        """
        Return system and Python information.
        """

        return {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "executable": sys.executable,
            "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
            "virtual_env": os.environ.get("VIRTUAL_ENV"),
        }

    @staticmethod
    def git_commit():
        """
        Return current Git commit hash.
        """

        try:
            return subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()

        except Exception:
            return None

    def installed_packages(self):
        """
        Return all installed packages.
        """

        if self._installed_packages is None:

            self._installed_packages = {
                dist.metadata["Name"]: dist.version
                for dist in distributions()
                if dist.metadata.get("Name")
            }

        return dict(sorted(self._installed_packages.items()))

    def imported_packages(self):
        """
        Return only packages corresponding to imported modules.
        """

        if self._imported_packages is not None:
            return self._imported_packages

        imported_modules = {
            name.split(".")[0]
            for name in sys.modules
        }

        packages = {}

        for dist in distributions():

            package_name = dist.metadata.get("Name")

            if not package_name:
                continue

            try:

                top_level = dist.read_text("top_level.txt")

                if not top_level:
                    continue

                module_names = {
                    line.strip()
                    for line in top_level.splitlines()
                    if line.strip()
                }

                if imported_modules.intersection(module_names):
                    packages[package_name] = dist.version

            except Exception:
                continue

        self._imported_packages = dict(sorted(packages.items()))

        return self._imported_packages

    def pip_freeze(self):
        """
        Return pip freeze output.
        """

        try:

            return subprocess.check_output(
                [sys.executable, "-m", "pip", "freeze"],
                text=True,
            )

        except Exception:

            return None

    def report(
        self,
        packages="imported",
        include_git=True,
        include_freeze=False,
    ):
        """
        Generate a formatted environment report.

        Parameters
        ----------
        packages : {"imported", "all", None}
            Package listing mode.

        include_git : bool
            Include Git commit hash.

        include_freeze : bool
            Include pip freeze output.
        """

        lines = []

        lines.append("=" * 80)
        lines.append("Execution Environment")
        lines.append("=" * 80)

        for key, value in self.system().items():
            lines.append(f"{key:24s}: {value}")

        if include_git:

            git_hash = self.git_commit()

            if git_hash:
                lines.append(f"{'git_commit':24s}: {git_hash}")

        if packages is not None:

            if packages == "imported":

                title = "Imported Packages"
                pkg_dict = self.imported_packages()

            elif packages == "all":

                title = "Installed Packages"
                pkg_dict = self.installed_packages()

            else:

                raise ValueError(
                    "packages must be 'imported', 'all', or None"
                )

            lines.append("")
            lines.append("=" * 80)
            lines.append(title)
            lines.append("=" * 80)

            for name, version in pkg_dict.items():
                lines.append(f"{name}=={version}")

        if include_freeze:

            freeze = self.pip_freeze()

            if freeze:

                lines.append("")
                lines.append("=" * 80)
                lines.append("pip freeze")
                lines.append("=" * 80)

                lines.append(freeze)

        return "\n".join(lines)

    def log(
        self,
        logger,
        packages="imported",
        include_git=True,
        include_freeze=False,
    ):
        """
        Write environment report to logger.
        """

        logger.info(
            self.report(
                packages=packages,
                include_git=include_git,
                include_freeze=include_freeze,
            )
        )
