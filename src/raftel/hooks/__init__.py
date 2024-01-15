"""Project hooks."""
from kedro.framework.hooks import hook_impl

from raftel.context import RaftelContext
from raftel.extras.utils import CurrentKedroContext


class ContextHooks:
    """Project Hooks"""

    # pylint: disable=too-few-public-methods
    @hook_impl
    def after_context_created(self, context: "RaftelContext") -> None:
        """Stores the current kedro context inside a singleton class.
        Args:
            context: The current Kedro context.
        """
        _ = CurrentKedroContext(context)
