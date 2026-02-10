from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

from utils.masks import EdgewiseMask


class CircuitDiscoveryAlgorithm(ABC):
    @abstractmethod
    def learn_mask(
        self,
        prompt: str,
        prompt_token_ids: List[int],
        branch_token_ids: List[List[int]],
        **kwargs: Any,
    ) -> Tuple[EdgewiseMask, Dict[str, Any]]:
        \"\"\"Learn a mask and return (mask, metrics).\"\"\"
        raise NotImplementedError
