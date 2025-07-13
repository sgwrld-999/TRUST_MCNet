# Unified Trust Strategy Migration Guide

## Overview

The TRUST_MCNet project has been updated to use a **Unified Trust Strategy** that combines the functionality of both the original `TrustWeightedStrategy` and `AdaptiveTrustStrategy` into a single, more maintainable class.

## What Changed

### Before (Legacy)
- **TrustWeightedStrategy**: Basic trust-weighted aggregation with fixed thresholds
- **AdaptiveTrustStrategy**: Extended trust strategy with adaptive threshold adjustment
- **Two separate files**: Duplicated functionality and harder to maintain

### After (Unified)
- **UnifiedTrustStrategy**: Single strategy class with both standard and adaptive modes
- **Backward Compatibility**: Legacy class names still work via aliases
- **Single file**: Easier maintenance and consistent behavior

## Migration Guide

### For Existing Code

**No changes required!** Your existing code will continue to work:

```python
# This still works (backward compatibility)
from trust_mcnet.strategies import TrustWeightedStrategy, AdaptiveTrustStrategy

# Standard trust strategy
strategy = TrustWeightedStrategy(trust_evaluator=evaluator)

# Adaptive trust strategy  
strategy = AdaptiveTrustStrategy(trust_evaluator=evaluator)
```

### For New Code (Recommended)

Use the unified strategy for new implementations:

```python
from trust_mcnet.strategies import UnifiedTrustStrategy

# Standard mode (equivalent to TrustWeightedStrategy)
strategy = UnifiedTrustStrategy(
    trust_evaluator=evaluator,
    enable_adaptation=False  # Default
)

# Adaptive mode (equivalent to AdaptiveTrustStrategy)
strategy = UnifiedTrustStrategy(
    trust_evaluator=evaluator,
    enable_adaptation=True,
    target_accuracy=0.85,
    threshold_adaptation_rate=0.05
)
```

## Configuration Updates

### YAML Configuration

The configuration format remains the same, but you can now specify the strategy type:

```yaml
federated:
  # Standard mode
  use_adaptive_strategy: false
  
  # OR Adaptive mode  
  use_adaptive_strategy: true
  target_accuracy: 0.85
  adaptation_rate: 0.05
  max_threshold: 0.9
  min_threshold: 0.3
```

### Main Entry Point

The main script automatically uses the unified strategy when available:

```bash
# Standard mode
python main.py --mode flower_server

# Adaptive mode (via config)
python main.py --mode flower_server --config config/enhanced_federated.yaml
```

## Key Benefits

### 1. **Unified Codebase**
- Single implementation reduces maintenance burden
- Consistent behavior across modes
- Easier testing and debugging

### 2. **Backward Compatibility** 
- Existing code works without changes
- Gradual migration possible
- No breaking changes

### 3. **Enhanced Features**
- Better error handling and logging
- Improved performance monitoring
- More robust fallback mechanisms

### 4. **Simplified Configuration**
- Single strategy class to configure
- Clearer parameter structure
- Better documentation

## Implementation Details

### Class Structure

```python
class UnifiedTrustStrategy(fl.server.strategy.FedAvg):
    def __init__(
        self,
        trust_evaluator: TrustEvaluator,
        enable_adaptation: bool = False,  # Key parameter for mode selection
        # Adaptive parameters (only used if enable_adaptation=True)
        target_accuracy: float = 0.85,
        threshold_adaptation_rate: float = 0.05,
        # ... other parameters
    ):
```

### Mode Selection

The strategy automatically configures itself based on the `enable_adaptation` parameter:

- **Standard Mode** (`enable_adaptation=False`): Fixed trust thresholds
- **Adaptive Mode** (`enable_adaptation=True`): Dynamic threshold adjustment

### Backward Compatibility

```python
# These are aliases that map to UnifiedTrustStrategy
TrustWeightedStrategy = UnifiedTrustStrategy
AdaptiveTrustStrategy = lambda *args, **kwargs: UnifiedTrustStrategy(*args, enable_adaptation=True, **kwargs)
```

## Testing

The unified strategy includes comprehensive tests:

```bash
# Run unified strategy tests
python -m pytest tests/test_unified_trust_strategy.py -v

# Run all strategy tests
python -m pytest tests/ -k "strategy" -v
```

## Performance Impact

### Minimal Overhead
- Single class loads faster than multiple classes
- Conditional features only activate when needed
- Optimized parameter handling

### Memory Usage
- Adaptive state only allocated when `enable_adaptation=True`
- Shared trust evaluation logic
- More efficient parameter management

## Migration Timeline

### Phase 1: ✅ Complete
- Unified strategy implementation
- Backward compatibility aliases
- Updated main entry point

### Phase 2: In Progress  
- Updated documentation
- Enhanced test coverage
- Performance optimizations

### Phase 3: Future
- Legacy strategy deprecation warnings (optional)
- Advanced adaptive algorithms
- Extended monitoring capabilities

## Troubleshooting

### Import Issues

If you encounter import issues:

```python
# Fallback approach
try:
    from trust_mcnet.strategies import UnifiedTrustStrategy
    # Use unified strategy
except ImportError:
    from trust_mcnet.strategies import TrustWeightedStrategy
    # Use legacy strategy
```

### Configuration Problems

Check your YAML configuration:

```yaml
# Make sure these are properly set
federated:
  use_adaptive_strategy: true  # or false
  # ... other parameters
```

### Missing Dependencies

Ensure all required packages are installed:

```bash
pip install flwr torch numpy omegaconf
```

## Questions?

For questions about the unified strategy:

1. Check the inline documentation in `unified_trust_strategy.py`
2. Review the test cases in `test_unified_trust_strategy.py`
3. See examples in the `examples/` directory
4. Check the main script implementation

The unified strategy provides the same functionality as before with improved maintainability and enhanced features!
