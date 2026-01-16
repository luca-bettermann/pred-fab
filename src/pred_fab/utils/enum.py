from enum import Enum

class SystemName(str, Enum):
    """Enumeration of system names in PFAB."""
    FEATURE = 'feature'
    EVALUATION = 'evaluation'
    PREDICTION = 'prediction'
    CALIBRATION = 'calibration'

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
    """Enumeration of block types in a neural network."""
    PARAMETERS = 'parameters'
    PERFORMANCE = 'performance_attrs'
    FEATURES = 'features'
    FEATURES_PRED = 'features_pred'

class Mode(str, Enum):
    """Enumeration of workflow step types."""
    OFFLINE = 'offline'
    ONLINE = 'online'

class Phase(str, Enum):
    """Enumeration of workflow stages."""
    LEARNING = 'learning'
    INFERENCE = 'inference'

class StepType(str, Enum):
    """Enumeration of workflow step types."""
    EVAL = 'evaluation_only'
    FULL = 'full_step'

class NormalizeStrategy(Enum):
    DEFAULT = 'default'
    STANDARD = 'standard'
    MINMAX = 'minmax'
    ROBUST = 'robust'
    NONE = 'none'
    CATEGORICAL = 'categorical'

class Roles(Enum):
    PARAMETER = 'parameter'
    DIMENSION = 'dimension'
    PERFORMANCE = 'performance'
    FEATURE = 'feature'
    