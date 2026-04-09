"""DataObject type system for schema definitions — validation, typing, and value storage."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any
import numpy as np

from ..utils.enum import NormMethod, Roles

class DataObject(ABC):
    """Base class for schema type definitions — encapsulates dtype, constraints, and role."""

    _registry: dict[str, type['DataObject']] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._registry[cls.__name__] = cls

    def __init__(self, code: str, dtype: type, role: Roles, constraints: dict[str, Any] | None = None,
                 round_digits: int | None = None):
        self.code = code
        self.dtype = dtype
        self.constraints = constraints or {}
        self.round_digits = round_digits
        self.role = role
        self.runtime_adjustable: bool = False  # Set to True by Parameter factory for runtime params

    @property
    def normalize_strategy(self) -> NormMethod:
        """Normalization strategy for this type; subclasses override as needed."""
        return NormMethod.DEFAULT

    @abstractmethod
    def validate(self, value: Any) -> bool:
        """Validate value against type and constraints; raises TypeError/ValueError if invalid."""
        pass

    @abstractmethod
    def coerce(self, value: Any) -> Any:
        """Coerce raw input to this object's canonical runtime dtype."""
        ...

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for schema storage."""
        data = {
            "code": self.code,
            "role": self.role.value if self.role else None,
            "type": self.__class__.__name__,
            "dtype": self.dtype.__name__,
            "constraints": self.constraints
        }
        if self.round_digits is not None:
            data["round_digits"] = self.round_digits
        if self.runtime_adjustable:
            data["runtime_adjustable"] = True
        return data

    def to_hash_dict(self) -> dict[str, Any]:
        """Serialize structural identity for schema hashing; excludes operational metadata (e.g. runtime_adjustable)."""
        data = {
            "code": self.code,
            "role": self.role.value if self.role else None,
            "type": self.__class__.__name__,
            "dtype": self.dtype.__name__,
            "constraints": self.constraints
        }
        if self.round_digits is not None:
            data["round_digits"] = self.round_digits
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'DataObject':
        """Reconstruct from dictionary."""
        obj_type = data["type"]
        code = data["code"]
        constraints = data["constraints"]
        round_digits = data.get("round_digits")
        role_raw = data.get("role")

        # Deserialize role from stored enum value string.
        role: Roles | None
        if role_raw is None:
            role = None
        elif isinstance(role_raw, Roles):
            role = role_raw
        elif isinstance(role_raw, str):
            try:
                role = Roles(role_raw)
            except ValueError as e:
                raise TypeError(f"Invalid role value '{role_raw}' for DataObject '{code}'") from e
        else:
            raise TypeError(f"Role must be a Roles enum member, role string, or None, got {type(role_raw).__name__}")

        if obj_type not in cls._registry:
            raise ValueError(f"Unknown DataObject type: {obj_type}")
        if not issubclass(cls._registry[obj_type], DataObject):
            raise TypeError(f"Registered type {obj_type} is not a DataObject subclass")
        if role is None or not isinstance(role, Roles):
            raise TypeError(f"Role must be a Roles enum member or None, got {type(role).__name__}")

        # Instantiate object using registry lookups implementation
        obj = cls._registry[obj_type]._from_json_impl(code, role, constraints, round_digits)
        obj.role = role
        obj.runtime_adjustable = data.get("runtime_adjustable", False)
        return obj

    @classmethod
    @abstractmethod
    def _from_json_impl(cls, code: str, role: Roles, constraints: dict[str, Any], round_digits: int | None) -> 'DataObject':
        """Implementation-specific reconstruction. Override in subclasses."""
        ...


class DataReal(DataObject):
    """Floating-point numeric parameter."""

    def __init__(self, code: str, role: Roles, min_val: float | None = None, max_val: float | None = None,
                 round_digits: int | None = None):
        constraints = {}
        if min_val is not None:
            constraints["min"] = min_val
        if max_val is not None:
            constraints["max"] = max_val
        super().__init__(code, float, role, constraints, round_digits)

    @property
    def normalize_strategy(self) -> NormMethod:
        return NormMethod.DEFAULT

    def validate(self, value: Any) -> bool:
        """Validate float value against constraints."""
        if not isinstance(value, (int, float)):
            raise TypeError(f"{self.code} must be numeric, got {type(value).__name__}")

        value = float(value)

        if "min" in self.constraints and value < self.constraints["min"]:
            raise ValueError(f"{self.code}={value} below minimum {self.constraints['min']}")

        if "max" in self.constraints and value > self.constraints["max"]:
            raise ValueError(f"{self.code}={value} above maximum {self.constraints['max']}")

        return True

    def coerce(self, value: Any) -> float:
        """Coerce numeric input to float and apply configured rounding."""
        coerced = float(value)
        if self.round_digits is not None:
            coerced = round(coerced, self.round_digits)
        return coerced

    @classmethod
    def _from_json_impl(cls, code: str, role: Roles, constraints: dict[str, Any], round_digits: int | None) -> 'DataReal':
        return cls(code, role, constraints.get("min"), constraints.get("max"), round_digits)


class DataInt(DataObject):
    """Integer numeric parameter."""

    def __init__(self, code: str, role: Roles, min_val: int | None = None, max_val: int | None = None, round_digits: int | None = None):
        constraints = {}
        if min_val is not None:
            constraints["min"] = int(min_val)
        if max_val is not None:
            constraints["max"] = int(max_val)
        super().__init__(code, int, role, constraints, round_digits)

    @property
    def normalize_strategy(self) -> NormMethod:
        return NormMethod.DEFAULT

    def validate(self, value: Any) -> bool:
        """Validate integer value against constraints."""
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{self.code} must be int, got {type(value).__name__}")

        if "min" in self.constraints and value < self.constraints["min"]:
            raise ValueError(f"{self.code}={value} below minimum {self.constraints['min']}")

        if "max" in self.constraints and value > self.constraints["max"]:
            raise ValueError(f"{self.code}={value} above maximum {self.constraints['max']}")

        return True

    def coerce(self, value: Any) -> int:
        """Coerce numeric input to int using round-to-nearest."""
        return int(round(float(value)))

    @classmethod
    def _from_json_impl(cls, code: str, role: Roles, constraints: dict[str, Any], round_digits: int | None) -> 'DataInt':
        return cls(code, role, constraints.get("min"), constraints.get("max"))


class DataBool(DataObject):
    """Boolean parameter."""

    def __init__(self, code: str, role: Roles):
        super().__init__(code, bool, role, {})

    @property
    def normalize_strategy(self) -> NormMethod:
        return NormMethod.NONE

    def validate(self, value: Any) -> bool:
        """Validate boolean value."""
        if not isinstance(value, bool):
            raise TypeError(f"{self.code} must be bool, got {type(value).__name__}")
        return True

    def coerce(self, value: Any) -> bool:
        """Coerce bool-like input to bool using threshold semantics for numerics."""
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, float, np.integer, np.floating)):
            return float(value) >= 0.5
        raise TypeError(f"{self.code} must be bool-like, got {type(value).__name__}")

    @classmethod
    def _from_json_impl(cls, code: str, role: Roles, constraints: dict[str, Any], round_digits: int | None) -> 'DataBool':
        return cls(code, role)

class DataCategorical(DataObject):
    """Categorical parameter with a fixed set of allowed string values."""

    def __init__(self, code: str, categories: list[str], role: Roles):
        if not categories:
            raise ValueError("Categories list cannot be empty")
        super().__init__(code, str, role, {"categories": categories})

    @property
    def normalize_strategy(self) -> NormMethod:
        return NormMethod.CATEGORICAL

    def validate(self, value: Any) -> bool:
        """Validate value is in allowed categories."""
        if not isinstance(value, str):
            raise TypeError(f"{self.code} must be str, got {type(value).__name__}")

        if value not in self.constraints["categories"]:
            raise ValueError(
                f"{self.code}={value} not in allowed categories: {self.constraints['categories']}"
            )

        return True

    def coerce(self, value: Any) -> str:
        """Coerce categorical input to string."""
        return str(value)

    @classmethod
    def _from_json_impl(cls, code: str, role: Roles, constraints: dict[str, Any], round_digits: int | None) -> 'DataCategorical':
        return cls(code, constraints["categories"], role)


class Dimension:
    """User-facing declaration of a single domain axis: a named integer iteration parameter."""

    def __init__(self, code: str, iterator_code: str, min_val: int = 1, max_val: int | None = None):
        self.code = code
        self.iterator_code = iterator_code
        self.min_val = int(min_val)
        self.max_val = int(max_val) if max_val is not None else None


class DataDomainAxis(DataInt):
    """Integer parameter auto-created for a domain axis; carries iterator_code. Never instantiated by users."""

    def __init__(self, code: str, iterator_code: str, role: Roles, min_val: int = 1, max_val: int | None = None):
        super().__init__(code, role, min_val, max_val)
        self.iterator_code = iterator_code
        self.constraints["domain_iterator_code"] = iterator_code

    @property
    def normalize_strategy(self) -> NormMethod:
        return NormMethod.MIN_MAX

    @classmethod
    def _from_json_impl(cls, code: str, role: Roles, constraints: dict[str, Any], round_digits: int | None) -> 'DataDomainAxis':
        return cls(code, constraints["domain_iterator_code"], role, constraints.get("min", 1), constraints.get("max"))


class Domain:
    """Named ordered sequence of axes defining an iteration space for feature extraction."""

    def __init__(self, code: str, axes: list[Dimension]):
        self.code = code
        self._axes: list[Dimension] = list(axes)

    @property
    def axes(self) -> list[Dimension]:
        return self._axes

    def get_param_codes(self) -> list[str]:
        """Return list of param codes for each axis."""
        return [a.code for a in self._axes]

    def get_iterator_codes(self) -> list[str]:
        """Return list of iterator_code strings for each axis."""
        return [a.iterator_code for a in self._axes]

    @property
    def depth(self) -> int:
        """Number of axes in this domain."""
        return len(self._axes)

    def create_axis_params(self, role: Roles) -> list['DataDomainAxis']:
        """Instantiate DataDomainAxis objects for each axis in this domain."""
        return [DataDomainAxis(a.code, a.iterator_code, role, a.min_val, a.max_val) for a in self._axes]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for schema storage."""
        return {
            "code": self.code,
            "axes": [{"code": a.code, "iterator_code": a.iterator_code,
                       "min_val": a.min_val, "max_val": a.max_val} for a in self._axes]
        }

    def to_hash_dict(self) -> dict[str, Any]:
        """Serialize structural identity for schema hashing."""
        return self.to_dict()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'Domain':
        """Reconstruct from dictionary."""
        axes = [Dimension(a["code"], a["iterator_code"], a["min_val"], a.get("max_val")) for a in data["axes"]]
        return cls(data["code"], axes)


class DataArray(DataObject):
    """Numpy array DataObject with dtype validation, used for feature tensor storage."""

    def __init__(self, code: str, role: Roles, dtype: str | type | np.dtype = np.float64,
                 domain_code: str | None = None, feature_depth: int | None = None,
                 context: bool = False,
                 recursive_source: str | None = None,
                 recursive_dimensions: tuple[str, ...] | None = None,
                 recursive_depth: int | None = None):
        # Convert to dtype instance for consistent validation and serialization
        resolved_dtype = np.dtype(dtype) if dtype else np.dtype(np.float64)
        self.columns: list[str] = []
        self.domain_code: str | None = domain_code
        self.feature_depth: int | None = feature_depth
        self.context: bool = context  # Observable but uncontrollable covariate; input-only, never optimized.
        self.recursive_source: str | None = recursive_source
        self.recursive_dimensions: tuple[str, ...] | None = recursive_dimensions
        self.recursive_depth: int | None = recursive_depth

        constraints: dict[str, Any] = {
            "dtype": resolved_dtype.name,
            "columns": self.columns
        }
        if domain_code is not None:
            constraints["domain_code"] = domain_code
        if feature_depth is not None:
            constraints["feature_depth"] = feature_depth
        if context:
            constraints["context"] = True
        if recursive_source is not None:
            constraints["recursive_source"] = recursive_source
        if recursive_dimensions is not None:
            constraints["recursive_dimensions"] = list(recursive_dimensions)
        if recursive_depth is not None:
            constraints["recursive_depth"] = recursive_depth

        super().__init__(code, np.ndarray, role, constraints)

    @property
    def is_recursive(self) -> bool:
        """Whether this feature is derived by shifting a source feature tensor."""
        return self.recursive_source is not None

    @property
    def normalize_strategy(self) -> NormMethod:
        return NormMethod.DEFAULT

    def set_columns(self, columns: list[str]) -> None:
        """Set associated iterator column names for this DataArray."""
        self.columns = columns
        self.constraints["columns"] = columns

    def validate(self, value: Any) -> bool:
        """Validate numpy array against shape and dtype constraints."""
        if not isinstance(value, np.ndarray):
            raise TypeError(f"{self.code} must be np.ndarray, got {type(value).__name__}")

        # Validate dtype if specified
        target_dtype = np.dtype(self.constraints.get("dtype"))
        if value.dtype != target_dtype:
            raise ValueError(
                f"{self.code} dtype mismatch: expected {target_dtype}, got {value.dtype}"
            )
        return True

    def coerce(self, value: Any) -> np.ndarray:
        """Coerce array-like input to ndarray with configured dtype."""
        target_dtype = np.dtype(self.constraints.get("dtype"))
        return np.asarray(value, dtype=target_dtype)

    @classmethod
    def _from_json_impl(cls, code: str, role: Roles, constraints: dict[str, Any],
                       round_digits: int | None = None) -> 'DataArray':
        """Reconstruct DataArray from serialized constraints."""
        rec_dims = constraints.get("recursive_dimensions")
        obj = cls(
            code, role,
            dtype=constraints.get("dtype", np.float64),
            domain_code=constraints.get("domain_code"),
            feature_depth=constraints.get("feature_depth"),
            context=constraints.get("context", False),
            recursive_source=constraints.get("recursive_source"),
            recursive_dimensions=tuple(rec_dims) if rec_dims is not None else None,
            recursive_depth=constraints.get("recursive_depth"),
        )
        obj.round_digits = round_digits
        if "columns" in constraints:
            obj.set_columns(constraints["columns"])
        return obj

# -------- Factories for Parameter declaration -----------

class Parameter:
    """Factory for creating parameter DataObjects."""

    @staticmethod
    def real(code: str, min_val: float, max_val: float, round_digits: int = 3, runtime: bool = False) -> DataReal:
        """Create a real-valued parameter; runtime=True marks it as re-proposable during fabrication."""
        obj = DataReal(code=code, min_val=min_val, max_val=max_val, round_digits=round_digits, role=Roles.PARAMETER)
        obj.runtime_adjustable = runtime
        return obj

    @staticmethod
    def integer(code: str, min_val: int, max_val: int, runtime: bool = False) -> DataInt:
        """Create an integer parameter; runtime=True marks it as re-proposable during fabrication."""
        obj = DataInt(code=code, min_val=min_val, max_val=max_val, role=Roles.PARAMETER)
        obj.runtime_adjustable = runtime
        return obj

    @staticmethod
    def categorical(code: str, categories: list[str]) -> DataCategorical:
        """Create a categorical parameter."""
        return DataCategorical(code=code, categories=categories, role=Roles.PARAMETER)

    @staticmethod
    def boolean(code: str) -> DataBool:
        """Create a boolean parameter."""
        return DataBool(code=code, role=Roles.PARAMETER)



class Feature:
    """Factory for creating feature Feature objects."""

    @staticmethod
    def array(code: str, dtype: str | type | np.dtype = np.float64,
              domain: Domain | None = None, depth: int | None = None) -> DataArray:
        """Create a feature DataArray tied to a domain. depth=None means full domain depth."""
        return DataArray(code=code, role=Roles.FEATURE, dtype=dtype,
                         domain_code=domain.code if domain is not None else None, feature_depth=depth)

    @staticmethod
    def context(code: str, dtype: str | type | np.dtype = np.float64,
                domain: Domain | None = None, depth: int | None = None) -> DataArray:
        """Create a context feature DataArray — observable but uncontrollable covariate.

        Context features are extracted like regular features but passed as fixed inputs
        to the prediction model at calibration time; they are never optimized.
        """
        return DataArray(code=code, role=Roles.FEATURE, dtype=dtype,
                         domain_code=domain.code if domain is not None else None,
                         feature_depth=depth, context=True)

    @staticmethod
    def recursive(
        code: str,
        source: 'DataArray',
        dimensions: Sequence['Dimension'],
        depth: int = 1,
        max_depth: int | None = None,
    ) -> DataArray | list[DataArray]:
        """Derive recursive feature(s) from a source feature by shifting along domain dimensions.

        With ``depth`` only (default 1), returns a single DataArray referencing the source
        tensor shifted by ``depth`` steps along each listed dimension.  Boundary positions
        are filled with NaN (converted to 0 at DataModule batch time).

        With ``max_depth``, returns a list of DataArrays for steps 1..max_depth, suitable
        for ``*``-unpacking into a Features block.
        """
        rec_dims = tuple(d.iterator_code for d in dimensions)
        domain_code = source.domain_code

        def _make(suffix_depth: int) -> DataArray:
            arr_code = f"{code}_{suffix_depth}" if max_depth is not None else code
            return DataArray(
                code=arr_code,
                role=Roles.FEATURE,
                domain_code=domain_code,
                feature_depth=source.feature_depth,
                context=True,  # recursive features are not optimised
                recursive_source=source.code,
                recursive_dimensions=rec_dims,
                recursive_depth=suffix_depth,
            )

        if max_depth is not None:
            return [_make(d) for d in range(1, max_depth + 1)]
        return _make(depth)


class PerformanceAttribute:
    """Factory for creating Performance objects."""

    @staticmethod
    def score(code: str, round_digits: int = 3) -> DataReal:
        """Create a normalized score (0-1) performance attribute."""
        return DataReal(code=code, min_val=0, max_val=1, round_digits=round_digits, role=Roles.PERFORMANCE)

