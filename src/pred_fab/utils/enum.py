from enum import Enum

class SystemName(str, Enum):
    """Enumeration of system names in PFAB."""
    FEATURE = 'feature'
    EVALUATION = 'evaluation'
    PREDICTION = 'prediction'
    CALIBRATION = 'calibration'

class Roles(Enum):
    """
    Block-membership role for DataObjects.

    Identifies which DataBlock a DataObject belongs to. Used by DataBlock.add()
    to enforce that objects are registered in the correct block.

    Note: sub-classifications *within* a block (e.g. runtime-adjustable parameters)
    are expressed as attributes on DataObject directly, not as additional Roles values.
    """
    PARAMETER = 'parameter'
    PERFORMANCE = 'performance'
    FEATURE = 'feature'

class NormMethod(str, Enum):
    """Enumeration of normalization methods for data preprocessing."""
    NONE = 'none'
    DEFAULT = 'default'
    MIN_MAX = 'min_max'
    STANDARD = 'standard'
    ROBUST = 'robust'
    CATEGORICAL = 'categorical'

class SplitType(str, Enum):
    """Enumeration of dataset split types."""
    TRAIN = 'train'
    VAL = 'validation'
    TEST = 'test'

class BlockType(str, Enum):
    """Enumeration of data block types in a dataset/experiment."""
    PARAMETERS = 'parameters'
    PARAM_UPDATES = 'parameter_updates'
    PERF_ATTRS = 'performance_attrs'
    FEATURES = 'features'
    METADATA = 'metadata'

class WorkflowDomain(str, Enum):
    """Enumeration of workflow step domain (distinct from core.data_objects.Domain)."""
    OFFLINE = 'offline'
    ONLINE = 'online'

class Mode(str, Enum):
    """Enumeration of workflow modes."""
    DISCOVERY = 'discovery'
    EXPLORATION = 'exploration'
    INFERENCE = 'inference'

class Loaders(Enum):
    MEMORY = 'memory'
    LOCAL = 'local files'
    EXTERNAL = 'external sources'
    
class FileFormat(str, Enum):
    CSV = 'csv'
    JSON = 'json'

class SamplingStrategy(str, Enum):
    """Enumeration for sampling strategies in design of experiments."""
    NUMERICAL = 'numerical'
    CATEGORICAL = 'categorical'
    BOOL = 'bool'

class SourceStep(str, Enum):
    """Identifies which agent step produced a ParameterProposal or ParameterUpdateEvent."""
    DISCOVERY = 'discovery_step'
    EXPLORATION = 'exploration_step'
    INFERENCE = 'inference_step'
    ADAPTATION = 'adaptation_step'

    
