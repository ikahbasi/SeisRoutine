import inspect
import logging
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)
from dataclasses import dataclass, field
import typing
import seisbench.models as sbm
import torch

logger = logging.getLogger(__name__)

ModelSpec = typing.Union[str, type]


class ModelCache:
    """
    Discover SeisBench models and cache their available pretrained weights.
    from: https://hifis-storage.desy.de:2880/Helmholtz/HelmholtzAI/SeisBench/
    """

    @dataclass
    class CacheResult:
        """Result of caching pretrained models."""

        cached: list[tuple[str, str]] = field(default_factory=list)
        failed: list[tuple[str, str, str]] = field(default_factory=list)
        skipped: list[str] = field(default_factory=list)

        @property
        def total(self) -> int:
            """Return the total number of attempted models."""
            return len(self.cached) + len(self.failed)

        def summary(self) -> str:
            """Return a human-readable summary of the caching result."""
            return (
                f"Cached {len(self.cached)}/{self.total} pretrained models "
                f"({len(self.failed)} failed, {len(self.skipped)} skipped)"
            )
        
        def __str__(self) -> str:
            """Return a human-readable representation of the caching result."""
            lines = [
                "",
                "=" * 60,
                "SeisBench Model Cache Result",
                "=" * 60,
                f"Cached : {len(self.cached)}",
                f"Failed : {len(self.failed)}",
                f"Skipped: {len(self.skipped)}",
                f"Total  : {self.total}",
                "-" * 60,
            ]

            if self.cached:
                lines.append("Cached models:")
                for model_name, pretrained_name in sorted(self.cached):
                    lines.append(f"  [OK] {model_name:<20} {pretrained_name}")
                lines.append("-" * 60)

            if self.failed:
                lines.append("")
                lines.append("Failed models:")
                for model_name, pretrained_name, error in sorted(self.failed):
                    lines.append(f"  [FAIL] {model_name:<20} {pretrained_name}  {error}")
                lines.append("-" * 60)

            if self.skipped:
                lines.append("")
                lines.append("Skipped models:")
                for model_name in sorted(self.skipped):
                    lines.append(f"  - {model_name}")

            lines.append("=" * 60)

            return "\n".join(lines)

    def __init__(
        self,
        *,
        max_workers: int = 4,
        raise_on_error: bool = False,
    ) -> None:
        """
        Initialize the SeisBench model cache manager.

        Parameters
        ----------
        max_workers : int, default=4
            Maximum number of concurrent model downloads.
        raise_on_error : bool, default=False
            If True, raise RuntimeError when one or more downloads fail.
        """
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")

        self.max_workers = max_workers
        self.raise_on_error = raise_on_error

        self._registry = self._discover_models()

    def _discover_models(self) -> dict[str, type]:
        """
        Discover available SeisBench model classes.

        Returns
        -------
        dict[str, type]
            Mapping from lowercase model names to model classes.
        """
        

        models: dict[str, type] = {}

        for name, obj in inspect.getmembers(sbm, inspect.isclass):
            # Only include classes defined inside the SeisBench models package.
            if not obj.__module__.startswith("seisbench.models"):
                continue

            # Only include classes implementing the pretrained model API.
            if not hasattr(obj, "list_pretrained"):
                continue

            if not hasattr(obj, "from_pretrained"):
                continue

            models[name.lower()] = obj

        return models

    def _resolve_model(self, spec: ModelSpec) -> type:
        """
        Resolve a model specification to a model class.

        Parameters
        ----------
        spec : str or type
            Model class or case-insensitive model class name.

        Returns
        -------
        type
            Resolved SeisBench model class.
        """
        if inspect.isclass(spec):
            return spec

        if isinstance(spec, str):
            key = spec.lower()

            if key not in self._registry:
                available = ", ".join(sorted(self._registry))
                raise KeyError(
                    f"Model '{spec}' not found in seisbench.models. "
                    f"Available: {available}"
                )

            return self._registry[key]

        raise TypeError(
            f"Expected a model name (str) or model class, "
            f"got {type(spec).__name__}"
        )

    def _get_model_classes(
        self,
        models: typing.Iterable[ModelSpec] | None,
    ) -> list[type]:
        """
        Resolve and deduplicate the requested model classes.
        """
        if models is None:
            model_classes = list(self._registry.values())

            logger.info(
                "No models given; discovered %d SeisBench models.",
                len(model_classes),
            )

            return model_classes

        model_classes: list[type] = []
        seen: set[type] = set()

        for spec in models:
            try:
                model_cls = self._resolve_model(spec)

            except (KeyError, TypeError) as exc:
                logger.warning("Skipping '%s': %s", spec, exc)
                continue

            if model_cls not in seen:
                seen.add(model_cls)
                model_classes.append(model_cls)

        return model_classes

    def _collect_tasks(
        self,
        model_classes: typing.Iterable[type],
    ) -> list[tuple[type, str, str]]:
        """
        Collect all available pretrained models for the given model classes.
        """
        tasks: list[tuple[type, str, str]] = []

        for model_cls in model_classes:
            model_name = getattr(model_cls, "__name__", str(model_cls))

            try:
                pretrained_names = list(model_cls.list_pretrained())

            except Exception as exc:
                logger.warning(
                    "Could not list pretrained models for %s: %s",
                    model_name,
                    exc,
                )
                continue

            for pretrained_name in pretrained_names:
                tasks.append(
                    (
                        model_cls,
                        model_name,
                        pretrained_name,
                    )
                )

        return tasks

    @staticmethod
    def _download(
        model_cls: type,
        model_name: str,
        pretrained_name: str,
    ) -> tuple[str, str]:
        """
        Download and cache a pretrained model.
        """
        model_cls.from_pretrained(pretrained_name)

        return model_name, pretrained_name

    def cache(
        self,
        models: typing.Iterable[ModelSpec] | None = None,
    ) -> CacheResult:
        """
        Download and locally cache available pretrained SeisBench models.

        Parameters
        ----------
        models : typing.Iterable[str | type] | None, default=None
            Model classes or model names. If None, all discoverable
            SeisBench models are used.

        Returns
        -------
        CacheResult
            Result containing successful, failed, and skipped models.

        Examples
        --------
        Cache all available SeisBench models:

        >>> cache = ModelCache()
        >>> result = cache.cache()

        Cache selected models:

        >>> cache = ModelCache(max_workers=4)
        >>> result = cache.cache(
        ...     ["PhaseNet", "EQTransformer", "GPD"]
        ... )

        Or using model classes:

        >>> import seisbench.models as sbm
        >>> result = cache.cache(
        ...     [sbm.PhaseNet, sbm.EQTransformer]
        ... )
        """
        model_classes = self._get_model_classes(models)

        tasks = self._collect_tasks(model_classes)

        logger.info(
            "Found %d pretrained model(s) to cache.",
            len(tasks),
        )

        result = self.CacheResult()

        with ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as pool:

            futures = {
                pool.submit(
                    self._download,
                    model_cls,
                    model_name,
                    pretrained_name,
                ): (model_name, pretrained_name)
                for model_cls, model_name, pretrained_name in tasks
            }

            for future in as_completed(futures):
                model_name, pretrained_name = futures[future]

                try:
                    future.result()

                    result.cached.append(
                        (model_name, pretrained_name)
                    )

                    logger.info(
                        "Cached %s/%s",
                        model_name,
                        pretrained_name,
                    )

                except Exception as exc:
                    result.failed.append(
                        (
                            model_name,
                            pretrained_name,
                            str(exc),
                        )
                    )

                    logger.error(
                        "Failed %s/%s: %s",
                        model_name,
                        pretrained_name,
                        exc,
                    )

        logger.info(result.summary())

        if self.raise_on_error and result.failed:
            raise RuntimeError(
                f"{len(result.failed)} model(s) failed to cache: "
                f"{result.failed}"
            )

        return result


def move_models_to_gpu(dl_pickers):
    """
    Move all deep-learning models to the available CUDA device.

    If CUDA is unavailable, models remain on CPU.
    If transferring an individual model fails, that model remains on CPU
    and the function continues with the remaining models.
    """
    if not torch.cuda.is_available():
        logging.info("Running on CPU. CUDA is not available!")
        return

    for key, dl_picker in dl_pickers.items():
        msg_model = (
            f"{key=}\n"
            f"{dl_picker.name=}\n"
            f"{dl_picker.weights_version=}\n"
            f"{dl_picker.weights_docstring=}\n"
        )

        try:
            dl_picker.cuda()
        except Exception as error:
            logging.warning(
                "key=%s Running on CPU due to %s.\n%s",
                key, error, msg_model,
            )
            continue

        logging.info(
            "Running on GPU.\n%s",
            msg_model
        )
