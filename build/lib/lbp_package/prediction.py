
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Type

# think about having a single prediction model class that takes all performances into account.


class MLModel(ABC):
    def __init__(
        self,
        folder_path: str,
        exp_code: str,
        model_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        # common, always‐required
        self.folder_path = folder_path
        self.exp_code = exp_code

        # delegate to subclass for model‐specific params
        self.model = {}
        self.params = {}
        self.set_parameters(model_config or {})

    @abstractmethod
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """Subclasses merge their DEFAULT_PARAMS + user overrides here."""
        ...

    @abstractmethod
    def train(self, data: Any) -> None:
        """Subclasses implement their own training here."""
        ...

    @abstractmethod
    def predict(self, data: Any) -> Any:
        """Subclasses implement their own prediction here."""
        ...
