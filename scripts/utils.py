"Extras"

import threading
from typing import Any, Dict, Union
from pathlib import Path
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from kedro.framework.startup import _get_project_metadata
from kedro.framework.hooks.manager import _create_hook_manager

from raftel.context import RaftelContext
from raftel import settings


def _create_kedro_context():
    """
    Initializes the Kedro context.
    """
    project_path = Path(Path.cwd()).resolve()
    metadata = _get_project_metadata(project_path)
    context_class = settings.CONTEXT_CLASS
    env = os.getenv("KEDRO_ENV")
    hook_manager = _create_hook_manager()

    config_loader_class = settings.CONFIG_LOADER_CLASS
    config_loader = config_loader_class(
        conf_source=str(metadata.project_path / settings.CONF_SOURCE),
        env=env,
        **settings.CONFIG_LOADER_ARGS,
    )
    context = context_class(
        package_name=metadata.package_name,
        project_path=metadata.project_path,
        config_loader=config_loader,
        env=env,
        extra_params={},
        hook_manager=hook_manager,
    )
    return context


def get_kedro_context() -> Union[RaftelContext, None]:
    """
    Returns the current Kedro context and tries to create one if not possible.
    """
    context = CurrentKedroContext().context
    if context is None:
        try:
            return _create_kedro_context()
        except Exception:  # pylint: disable=broad-except
            return None

    return context


class SingletonMeta(type):
    """
    Implementation of a Singleton class using the metaclass method
    """

    _instances = {}
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        """
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    instance = super(SingletonMeta, cls).__call__(*args, **kwargs)
                    cls._instances[cls] = instance
        return cls._instances[cls]


class CurrentKedroContext(metaclass=SingletonMeta):
    """
    A singleton class that stores the current Kedro context.
    """

    # pylint: disable=too-few-public-methods
    def __init__(self, context: RaftelContext = None):
        """
        Virtually private constructor.
        Args:
            context: The current Kedro context.
        """
        self._context: RaftelContext = context

    @property
    def context(self) -> RaftelContext:
        """Returns the current Kedro context."""
        return self._context

    @context.setter
    def context(self, context: RaftelContext):
        """Sets the current Kedro context."""
        self._context = context
