import logging

# ============================== Optimizers ==============================

from .adam_mini import Adam_mini
from .ada_ema_mix import AdEMAMix
from .muon import Muon
try:
    from .shampoo_optimizers import distributed_shampoo
except ImportError:
    distributed_shampoo = None
    logging.warning('shampoo optimizer is not installed')

from .SOAP import SOAP  # original implementation
# other implementations, including scheduler-free, precondition, palm, etc.
# from .SOAP_for_each import PaLMForeachSOAP as SOAP_for_each
# from .SOAP_SF_for_each import PaLMForeachSOAP as SOAP_SF_for_each
from heavyball import (
    # SOAP family ===================
    ForeachSOAP,
    PaLMForeachSOAP,
    SFPaLMForeachSOAP,
    PrecondScheduleForeachSOAP,
    PrecondSchedulePaLMForeachSOAP,
    PrecondScheduleSFPaLMSOAP,    
    # PSGD family ===================
    ForeachPSGDKron,
    ForeachPurePSGD,
    ForeachDelayedPSGD,
    utils as heavyball_utils
)

# add to make torch serialization safe
import torch
from accelerate.utils.other import TORCH_SAFE_GLOBALS
from collections import defaultdict

_SAFE_OPTIMIZERS_CLS = [SOAP,
                        ForeachSOAP,
                        PaLMForeachSOAP,
                        SFPaLMForeachSOAP,
                        PrecondScheduleForeachSOAP,
                        PrecondSchedulePaLMForeachSOAP,
                        PrecondScheduleSFPaLMSOAP,
                        ForeachPSGDKron,
                        ForeachPurePSGD,
                        ForeachDelayedPSGD,
                        Muon, 
                        Adam_mini,
                        AdEMAMix]
torch.serialization.add_safe_globals(_SAFE_OPTIMIZERS_CLS)

TORCH_SAFE_GLOBALS.extend(_SAFE_OPTIMIZERS_CLS)
TORCH_SAFE_GLOBALS.extend([defaultdict, dict])

# ============================== Models ==============================

# Unet used in EMMA loss for image fusion
from .unet5 import UNet5 as TranslationUnet