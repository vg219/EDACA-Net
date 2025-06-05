import os
from torch import Tensor

import jaxtyping
from jaxtyping import jaxtyped
from beartype import beartype
from beartype.door import is_bearable

class TorchTyping:
    def __init__(self, abstract_dtype):
        self.abstract_dtype = abstract_dtype

    def __getitem__(self, shapes: str):
        return self.abstract_dtype[Tensor, shapes]


if bool(os.getenv('type_check')):
    typing_checker = jaxtyped(typechecker=beartype)
else:
    typing_checker = jaxtyped(typechecker=None)

Float = TorchTyping(jaxtyping.Float)
Int   = TorchTyping(jaxtyping.Int)
Bool  = TorchTyping(jaxtyping.Bool)

FloatBatchedImage = Float["B C H W"]
FloatBatchedSeqCLast = Float["B L C"]
FloatBatchedSeqCFirst = Float["B C L"]

