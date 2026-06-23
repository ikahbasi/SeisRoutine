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
import inspect
import json
from time import perf_counter
from functools import wraps


class ProgressMsg:
    """
    Utility class for building formatted progress messages.

    This class provides methods to compute completion percentages
    and generate a human-readable summary of multiple progress items.
    """

    @staticmethod
    def pct(part, total):
        """
        Calculate completion percentage.

        Args:
            part (int | float): Completed amount.
            total (int | float): Total amount.

        Returns:
            float: Percentage value between 0 and 100.
                   Returns 0.0 if total is zero.
        """
        return 100 * part / total if total else 0.0

    @classmethod
    def build(cls, subject="Passed", **kwargs):
        """
        Build a formatted progress message string.

        The first element of the message is a subject label (default: "Passed"),
        followed by progress entries in the form:
            part/total (percentage%) key

        Each keyword argument must be provided as:
            key = (part, total)

        Args:
            subject (str): Leading label for the message. Default is "Passed".
            **kwargs: Named progress items where each value is a tuple
                      of (part, total).

        Returns:
            str: A pipe-separated progress summary string.

        Example:
            ProgressMsg.build(
                subject="Completed",
                downloads=(3, 10),
                uploads=(2, 5)
            )

            Output:
            "Completed | 3/10 (30.00%) downloads | 2/5 (40.00%) uploads"
        """

        parts = [subject]

        for key, (part, total) in kwargs.items():
            parts.append(
                f"{part}/{total} ({cls.pct(part, total):.2f}%) {key}"
            )

        return " | ".join(parts)


def timestamp(format='%Y-%m-%dT%H-%M-%S'):
    return datetime.now().strftime(format)


def timer(func):
    """
    Measure and print the execution time of a function.

    This decorator wraps a function, records its start and end time
    using `time.perf_counter()`, and prints the elapsed execution
    time after the function completes.

    Args:
        func: The function to be wrapped.

    Returns:
        A wrapped function that behaves exactly like the original
        function while logging its execution time.

    Example:
        >>> @timer
        ... def process_data():
        ...     time.sleep(1)
        ...
        >>> process_data()
        process_data: 1.000123s

    Notes:
        - Uses `time.perf_counter()` for high-resolution timing.
        - Preserves the original function's metadata via
          `functools.wraps`.
        - Suitable for quick profiling and debugging.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = perf_counter()
        result = func(*args, **kwargs)
        elapsed = perf_counter() - start
        print(f"{func.__name__}: {elapsed:.6f}s")
        return result
    return wrapper


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

    @classmethod
    def load(cls, file_path):
        with open(file_path, "r") as file:
            config_dict = yaml.safe_load(file)
        return cls(**config_dict)

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


class RuntimeLocation:
    """
    Detects the calling context of a function (script file, notebook file,
    or interactive terminal) across Windows/Linux and across VSCode,
    Jupyter Notebook/Lab, and Spyder.

    All methods are static, so the class can be used without instantiation:

        from runtime_location import RuntimeLocation
        print(RuntimeLocation.log_location())
    
        
    A small utility to automatically detect "where" a function was called from,
    so the information can be written to a log file. It distinguishes between:

        1) Running a .py file directly
        (terminal `python script.py`, VSCode "Run", or Spyder's runfile())
        -> returns the name and directory of that file.

        2) Running inside a Jupyter notebook (.ipynb)
        - a notebook opened inside VSCode
        - a classic Jupyter Notebook / JupyterLab notebook
        -> tries to resolve the .ipynb file's name and directory in both cases.

        3) Interactive execution with no backing file (a "terminal")
        - Spyder's interactive console
        - an IPython terminal
        - a plain Python REPL
        -> returns the current working directory (cwd) in these cases.

    Cross-platform: works on both Windows and Linux (all paths are built with
    os.path / pathlib, which are platform-aware).

    Known limitation:
        Resolving the exact .ipynb path in classic Jupyter Notebook/Lab requires
        querying the local Jupyter server's REST API (the same technique used by
        well-known packages such as ipynbname / nci_ipynb). If that query fails
        for any reason (jupyter_core missing, server unreachable, etc.), the
        method falls back to the "Jupyter Notebook" label with just the current
        working directory instead of raising an error.
        If you install the `ipynbname` package (`pip install ipynbname`),
        detection is usually more accurate and robust; this class will use it
        automatically when available.
    """

    # ----------------------------------------------------------------- #
    # Internal helper methods
    # ----------------------------------------------------------------- #

    @staticmethod
    def _get_ipython_instance():
        """Return the IPython shell instance if running inside IPython/Jupyter, else None."""
        try:
            return get_ipython()  # noqa: F821  -> only defined inside an IPython/Jupyter runtime
        except NameError:
            return None

    @staticmethod
    def _is_spyder_console():
        """
        Detect whether the current process is a Spyder interactive console.
        The most reliable signal: Spyder consoles are always launched through
        the spyder_kernels package.
        """
        if "spyder_kernels" in sys.modules:
            return True
        if any(var in os.environ for var in ("SPY_PYTHONPATH", "SPYDER_ARGS")):
            return True
        return False

    @staticmethod
    def _vscode_notebook_file(ip):
        """
        VSCode's Jupyter extension injects a __vsc_ipynb_file__ variable into
        the user namespace (user_ns), pointing to the absolute path of the
        currently open .ipynb file.
        """
        try:
            path = ip.user_ns.get("__vsc_ipynb_file__")
        except Exception:
            return None
        if path and os.path.exists(path):
            return path
        return None

    @staticmethod
    def _classic_jupyter_notebook_path():
        """
        Try to resolve the .ipynb file path for classic Jupyter Notebook/Lab.
        First attempts the `ipynbname` package (if installed, since it is the
        most accurate and actively maintained implementation of this trick).
        Falls back to a manual implementation of the same approach: derive
        the kernel_id from the connection file, then query the running
        Jupyter server(s)' REST API for the matching session.
        """
        try:
            import ipynbname
            return str(ipynbname.path())
        except Exception:
            pass

        try:
            import ipykernel
            from itertools import chain
            from urllib.request import urlopen

            try:
                from jupyter_core.paths import jupyter_runtime_dir
                runtime_dir = Path(jupyter_runtime_dir())
            except Exception:
                return None

            connection_file = Path(ipykernel.get_connection_file()).stem
            kernel_id = connection_file.split("-", 1)[-1]

            server_files = sorted(
                chain(
                    runtime_dir.glob("nbserver-*.json"),  # classic Jupyter Notebook / Lab 2
                    runtime_dir.glob("jpserver-*.json"),  # JupyterLab 3+ / Notebook 7+
                ),
                key=os.path.getmtime,
                reverse=True,
            )

            for server_file in server_files:
                try:
                    info = json.loads(server_file.read_text())
                    token = info.get("token", "") or os.environ.get("JUPYTERHUB_API_TOKEN", "")
                    qs = f"?token={token}" if token else ""
                    url = f"{info['url']}api/sessions{qs}"
                    sessions = json.loads(urlopen(url, timeout=1.0).read())
                    for sess in sessions:
                        if sess.get("kernel", {}).get("id") == kernel_id:
                            # Response shape differs slightly across server versions
                            rel_path = sess.get("path") or sess.get("notebook", {}).get("path")
                            if not rel_path:
                                continue
                            root_dir = info.get("root_dir") or info.get("notebook_dir")
                            if root_dir:
                                return str(Path(root_dir) / rel_path)
                            return rel_path
                except Exception:
                    continue
        except Exception:
            pass
        return None

    # ----------------------------------------------------------------- #
    # Public API (import and use these)
    # ----------------------------------------------------------------- #

    @staticmethod
    def get_caller_info(_offset=0):
        """
        Return information about the environment this method was called from.

        You normally don't need to touch the _offset parameter; it only
        exists to let other helper methods (like log_location) build on top
        of this one while still pointing at the real caller.

        Returns -> dict with keys:
            kind       : machine-readable environment id:
                         'script' | 'jupyter' | 'vscode_notebook' |
                         'spyder_console' | 'terminal_ipython' | 'terminal_python'
            label      : human-readable label
            name       : file name (or environment label in interactive mode)
            directory  : directory path (file location, or current working
                         directory in interactive mode)
            full_path  : full file path if available, otherwise None
        """
        stack = inspect.stack()
        frame_info = stack[1 + _offset]
        caller_globals = frame_info.frame.f_globals
        file_in_globals = caller_globals.get("__file__")

        # 1) Direct execution of a .py file
        #    (terminal, VSCode "Run", or Spyder's runfile() - in all three
        #    cases __file__ is set to the script's path)
        if file_in_globals and os.path.isfile(file_in_globals):
            full_path = os.path.abspath(file_in_globals)
            return {
                "kind": "script",
                "label": "Python script (.py)",
                "name": os.path.basename(full_path),
                "directory": os.path.dirname(full_path),
                "full_path": full_path,
            }

        ip = RuntimeLocation._get_ipython_instance()

        if ip is not None:
            shell_name = ip.__class__.__name__

            if shell_name == "ZMQInteractiveShell":
                # 2) Notebook opened inside VSCode
                vsc_path = RuntimeLocation._vscode_notebook_file(ip)
                if vsc_path:
                    full_path = os.path.abspath(vsc_path)
                    return {
                        "kind": "vscode_notebook",
                        "label": "Jupyter notebook in VSCode",
                        "name": os.path.basename(full_path),
                        "directory": os.path.dirname(full_path),
                        "full_path": full_path,
                    }

                # Both Spyder's console and Jupyter notebooks use this same kernel shell class
                if RuntimeLocation._is_spyder_console():
                    return {
                        "kind": "spyder_console",
                        "label": "Spyder interactive console",
                        "name": "Spyder Console",
                        "directory": os.getcwd(),
                        "full_path": None,
                    }

                nb_path = RuntimeLocation._classic_jupyter_notebook_path()
                if nb_path:
                    full_path = os.path.abspath(nb_path)
                    return {
                        "kind": "jupyter",
                        "label": "Jupyter notebook (Notebook/Lab)",
                        "name": os.path.basename(full_path),
                        "directory": os.path.dirname(full_path),
                        "full_path": full_path,
                    }
                return {
                    "kind": "jupyter",
                    "label": "Jupyter notebook (exact path not found)",
                    "name": "Unknown Notebook",
                    "directory": os.getcwd(),
                    "full_path": None,
                }

            if shell_name == "TerminalInteractiveShell":
                return {
                    "kind": "terminal_ipython",
                    "label": "Terminal (IPython)",
                    "name": "Terminal (IPython)",
                    "directory": os.getcwd(),
                    "full_path": None,
                }

        # 3) Plain Python terminal (simple REPL, no IPython)
        return {
            "kind": "terminal_python",
            "label": "Terminal (Python REPL)",
            "name": "Terminal (Python)",
            "directory": os.getcwd(),
            "full_path": None,
        }

    @staticmethod
    def log_location():
        """
        Return a ready-to-use string for inserting into a log line.
        Call this directly anywhere in your code:

            from runtime_location import RuntimeLocation
            print(f"{RuntimeLocation.log_location()} your log message")

            # or with the logging module:
            logging.info(f"{RuntimeLocation.log_location()} your log message")
        """
        info = RuntimeLocation.get_caller_info(_offset=1)  # look one frame up: the real caller of log_location
        if info["full_path"]:
            return f"[{info['label']} | {info['name']} | {info['directory']}]"
        return f"[{info['label']} | {info['directory']}]"


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
