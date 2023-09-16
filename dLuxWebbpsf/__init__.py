name = "dLuxWebbpsf"
__version__ = "0.0.1"

# Import as modules
from . import optical_layers
from . import instruments
from . import propagators
from . import detector_layers

# Import core functions from modules
from .optical_layers import *
from .instruments import *
from .propagators import *
from .detector_layers import *


# Add to __all__
__all__ = (
      optical_layers.__all__
    + instruments.__all__
    + propagators.__all__
    + detector_layers.__all__
)


# Check for 64-bit
from jax import config

if not config.x64_enabled:
    print(
        "dLux: Jax is running in 32-bit, to enable 64-bit visit: "
        "https://jax.readthedocs.io/en/latest/notebooks/"
        "Common_Gotchas_in_JAX.html#double-64bit-precision"
    )
