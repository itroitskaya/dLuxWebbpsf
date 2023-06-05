name = "dLuxWebbpsf"
__version__ = "0.0.1"

# Import as modules
from . import core
from . import optics

# Import core functions from modules
from .core         import *
from .optics       import *

# Add to __all__
__all__ = core.__all__ + optics.__all__ 

# Check for 64-bit
from jax import config
if not config.x64_enabled:
    print("dLux: Jax is running in 32-bit, to enable 64-bit visit: "
          "https://jax.readthedocs.io/en/latest/notebooks/"
          "Common_Gotchas_in_JAX.html#double-64bit-precision")