"""Project settings. There is no need to edit this file unless you want to change values
from the Kedro defaults. For further information, including these default values, see
https://kedro.readthedocs.io/en/stable/kedro_project_setup/settings.html."""

import os

from kedro.config import TemplatedConfigLoader
from kedro.io import DataCatalog
from raftel.hooks import ContextHooks
from raftel.context import RaftelContext


# Instantiated project hooks.
# from raftel.hooks import ProjectHooks
HOOKS = (ContextHooks(),)

# Installed plugins for which to disable hook auto-registration.
# DISABLE_HOOKS_FOR_PLUGINS = ("kedro-viz",)

# Class that manages storing KedroSession data.
# from kedro.framework.session.shelvestore import ShelveStore
# SESSION_STORE_CLASS = ShelveStore
# Keyword arguments to pass to the `SESSION_STORE_CLASS` constructor.
# SESSION_STORE_ARGS = {
#     "path": "./sessions"
# }

# Class that manages Kedro's library components.
# from kedro.framework.context import KedroContext
CONTEXT_CLASS = RaftelContext

# Directory that holds configuration.
CONF_SOURCE = os.getenv("KEDRO_CONF_ROOT", "conf")

# Class that manages how configuration is loaded.
CONFIG_LOADER_CLASS = TemplatedConfigLoader
# Keyword arguments to pass to the `CONFIG_LOADER_CLASS` constructor.
CONFIG_LOADER_ARGS = {
      "config_patterns": {
          "spark" : ["spark*", "spark*/**", "**/spark*"],
      },
      "globals_pattern": "*globals.yml",
}

# Class that manages the Data Catalog.
# from kedro.io import DataCatalog
DATA_CATALOG_CLASS = DataCatalog
