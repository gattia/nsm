# [BREAKING CHANGE PROPOSAL] Standardize Sigma Sampling to Original Coordinate Space

## 🎯 Problem Statement

Currently, our SDF dataset sampling has **two fundamentally different coordinate space interpretations** for sigma values depending on the `scale_jointly` flag:

- **`scale_jointly=False` (default)**: Sigma sampling occurs **after** individual mesh normalization
  - Coordinate space: Normalized [-1,1] cube 
  - Required sigma values: Small (0.01-0.1)
  
- **`scale_jointly=True`**: Sigma sampling occurs **before** normalization  
  - Coordinate space: Original mesh units (e.g., millimeters)
  - Required sigma values: Large (0.5-5.0)

This dual-mode system creates several critical issues:

### 🚨 Current Problems

1. **Silent Failures**: Using wrong sigma values results in 100x over/under-sampling with no error
2. **Counter-Intuitive Behavior**: Same sigma value means completely different things in different modes
3. **Poor Cross-Dataset Compatibility**: Sigma values learned for one dataset don't transfer
4. **Documentation Complexity**: Requires extensive explanation of dual coordinate systems
5. **User Confusion**: Easy to use wrong sigma values, leading to poor model performance

### 📊 Impact Analysis

From our analysis of the codebase:
- **Old scripts**: Used small sigma values (0.01, 0.1) with `scale_jointly=False`
- **New scripts**: Use large sigma values (2.0+) with `scale_jointly=True` 
- **Breaking point**: ~100x difference in effective sampling density between modes

## 🎯 Proposed Solution

**Standardize all sigma sampling to occur in original coordinate space**, making the behavior intuitive and consistent.

### 🔄 New Unified Behavior

```python
# AFTER: Always intuitive, real-world sigma values
sigma_near = 0.74  # Always means 0.74mm from surface (for medical data)
sigma_far = 2.35   # Always means 2.35mm from surface

# Works the same regardless of other scaling options
dataset = SDFSamples(sigma_near=0.74, sigma_far=2.35, scale_jointly=True/False)
```

## 🛠️ Implementation Plan

### Phase 1: Foundation (Non-Breaking)
- [x] ✅ Add comprehensive documentation about current dual-mode behavior
- [x] ✅ Implement warning system for potentially incorrect sigma values  
- [ ] Add new parameter `sigma_coordinate_space` with options `["legacy", "original"]`
- [ ] Default to `"legacy"` (maintains current behavior) with deprecation warning

### Phase 2: Migration Tools (Non-Breaking)
- [ ] Add `convert_sigma_values()` utility function:
  ```python
  # Convert old normalized sigmas to new original-space sigmas
  new_sigma = convert_sigma_values(
      old_sigma=0.01, 
      typical_mesh_radius=40.0,  # mm
      from_space="normalized", 
      to_space="original"
  )
  ```
- [ ] Add `estimate_mesh_scale()` helper to automatically detect coordinate units

### Phase 3: Breaking Change (Major Version)
- [ ] Change default behavior: **always sample in original coordinate space**
- [ ] Update default sigma values to real-world scale:
  ```python
  # New defaults (for medical data in mm)
  sigma_near=0.74,    # Was: 0.01 
  sigma_far=2.35      # Was: 0.1
  ```
- [ ] Remove `scale_jointly` coordinate space dependency
- [ ] Simplify normalization: always apply after sampling if requested

### Phase 4: Cleanup (Major Version)
- [ ] Remove legacy `sigma_coordinate_space` parameter
- [ ] Remove dual-mode documentation
- [ ] Simplify codebase by eliminating coordinate space switching logic

## 📋 Migration Guide

### For Existing Users

#### Option 1: Keep Current Behavior (Temporary)
```python
# Keep existing scale_jointly-dependent behavior (deprecated)
dataset = SDFSamples(
    sigma_near=0.01, 
    sigma_far=0.1,
    scale_jointly=False,  # Uses normalized space
    sigma_coordinate_space="legacy"  # Explicit legacy mode (default)
)
```

#### Option 2: Force Original Space (Current scale_jointly=True behavior)  
```python
# Always use original coordinate space regardless of scale_jointly
dataset = SDFSamples(
    sigma_near=0.74,    # Real-world values (e.g., mm)
    sigma_far=2.35,
    scale_jointly=False,  # This gets ignored for sigma sampling  
    sigma_coordinate_space="original"  # Always original space
)
```

#### Option 3: Migrate to New Behavior (Recommended for Breaking Change)
```python
# After breaking change: always original space, convert old values
old_sigma_near = 0.01  # Your old normalized-space value
estimated_mesh_radius = 40.0  # mm, typical for your data

new_sigma_near = old_sigma_near * estimated_mesh_radius  # 0.01 * 40 = 0.4mm
new_sigma_far = 0.1 * estimated_mesh_radius  # 0.1 * 40 = 4.0mm

# Post-breaking-change approach (always original space)
dataset = SDFSamples(
    sigma_near=0.4,    # Real-world mm units
    sigma_far=4.0,     # Real-world mm units
    # No sigma_coordinate_space parameter - always original space
)
```

## 🎯 Benefits

### 1. **Intuitive Usage**
```python
# Clear, real-world interpretation
sigma_near = 1.0  # Always means 1mm from surface
sigma_far = 5.0   # Always means 5mm from surface
```

### 2. **Cross-Dataset Compatibility**
```python
# Same sigma values work across different anatomical structures
femur_dataset = SDFSamples(sigma_near=0.74, sigma_far=2.35)
tibia_dataset = SDFSamples(sigma_near=0.74, sigma_far=2.35)  # Same values!
```

### 3. **Eliminates Silent Failures**
- No more 100x coordinate space mismatches
- Sigma values have consistent physical meaning
- Easier to debug sampling issues

### 4. **Simplified Documentation**
- Single coordinate space explanation
- No need to explain dual modes
- Clear relationship between sigma and real-world distance

## ⚠️ Breaking Change Impact

### What Will Break
- Code using `scale_jointly=False` with small sigma values (0.01-0.1)
- Scripts that rely on normalized coordinate sampling behavior
- Any hard-coded sigma values optimized for normalized space

### What Won't Break  
- Code using `scale_jointly=True` (already uses original coordinates)
- Scripts that explicitly convert between coordinate spaces
- Most reconstruction and training pipelines (with updated sigma values)



---

**Breaking Change Label**: `breaking-change`, `v3.0.0`, `sigma-standardization`
