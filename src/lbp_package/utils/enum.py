import enum

class NormMethod(str, enum.Enum):
    """Enumeration of normalization methods for data preprocessing."""
    NONE = 'none'
    DEFAULT = 'default'
    MIN_MAX = 'min_max'
    STANDARD = 'standard'
    ROBUST = 'robust'
    CATEGORICAL = 'categorical'

class SplitType(str, enum.Enum):
    """Enumeration of dataset split types."""
    TRAIN = 'train'
    VAL = 'validation'
    TEST = 'test'

class BlockType(str, enum.Enum):
    """Enumeration of block types in a neural network."""
    PARAMETERS = 'parameters'
    PERFORMANCE = 'performance_attrs'
    FEATURES = 'features'
    FEATURES_PRED = 'features_pred'

class Mode(str, enum.Enum):
    """Enumeration of workflow step types."""
    EXPLORATION = 'exploration'
    EXPLOITATION = 'exploitation'

class StepType(str, enum.Enum):
    """Enumeration of workflow step types."""
    EVAL = 'evaluation_only'
    FULL = 'full_step'
