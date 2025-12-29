# Ablation Studies Summary

## Overview
The KFujikawa pipeline conducted systematic ablation studies to validate component contributions to their FastFormer/transformer models. These experiments are located in `kfujikawa/src/exp/v2xxx_ablation/`.

## Feature Control System

### Leakage Configuration Encoding
KFujikawa uses a 3-digit naming system to control different types of temporal information:

```python
# Filename format: v2xxx_[ABC]_description.py
USE_FUTURE_IMP = FILE_NAME.split("_")[1][0] == "1"           # A: Future impression features
USE_FUTURE_ARTICLE_STATS = FILE_NAME.split("_")[1][1] == "1" # B: Future article statistics  
USE_PAST_IMP = FILE_NAME.split("_")[1][2] == "1"             # C: Past impression features
```

**Configuration Examples**:
- `000` = No future information (pure temporal model)
- `001` = Past impressions only 
- `111` = All temporal features (production baseline)

## Ablation Experiments

### Data Leakage Studies
| Experiment | Configuration | Purpose |
|------------|---------------|---------|
| `v2001_000_v1174_noleak.py` | `000` | **No future leakage** - Tests model without any future information |
| `v2002_001_v1174_use_past.py` | `001` | **Past only** - Tests with historical impressions but no future data |
| Production models | `111` | **All leakage enabled** - Full temporal feature set |

### Component Ablations
| Experiment | Target Component | Purpose |
|------------|------------------|---------|
| `v2003_111_v1174_no-drophist.py` | History dropout | Tests impact of dropout history mechanism |
| `v2004_111_v1174_no-imp.py` | Impression features | Tests without impression-level features |
| `v2005_111_v1174_no-transformer.py` | Transformer architecture | Tests without transformer components |
| `v2006_111_v1174_no-sample.py` | Sampling strategies | Tests without advanced sampling |
| `v2007_111_v1174_no-transformer-same-parameter.py` | Transformer (controlled) | Parameter-controlled transformer comparison |

## Key Findings

### Strategic Leakage Validation
- **Future information significantly improves performance**: The team chose `111` configuration (all leakage) for production
- **Past-only models underperform**: `001` configuration inferior to `111`
- **No-leakage models insufficient**: `000` configuration shows substantial performance drop

### Component Contributions
- **Transformer architecture essential**: `no-transformer` ablations demonstrate core importance
- **History dropout beneficial**: `no-drophist` shows dropout history mechanism adds value
- **Sampling strategies matter**: `no-sample` indicates advanced sampling improves results

## Validation Scope Limitations

### Comprehensive (KFujikawa)
- ✅ **Neural Networks**: Systematic ablation studies across all components
- ✅ **Temporal Features**: Thorough validation of leakage strategies
- ✅ **Architecture Components**: Individual component impact assessment

### Limited (KAMI)
- ⚠️ **Gradient Boosting**: No ablation studies despite having configuration flags
- ⚠️ **Feature Impact**: Infrastructure exists but experiments not conducted
- ⚠️ **Leakage Validation**: KAMI uses same leakage but without validation

## Strategic Implications

1. **Asymmetric Validation**: Only neural network components systematically validated
2. **Leakage Dependency**: Future information crucial for winning performance
3. **Component Synergy**: Multiple optimizations (dropout, sampling, attention) combine for best results
4. **Production Choices**: All final models use `111` configuration based on ablation results

## Experimental Rigor

The KFujikawa ablation studies demonstrate **scientific methodology** in deep learning for recommendation systems:
- Controlled experiments with single-variable changes
- Systematic evaluation of temporal information impact
- Component-wise architecture validation
- Evidence-based production configuration selection

However, the **incomplete validation across the full ensemble** (missing KAMI ablations) represents a gap in comprehensive system understanding.