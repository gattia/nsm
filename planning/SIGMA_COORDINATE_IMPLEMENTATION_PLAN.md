# Implementation Plan: Sigma Coordinate Space Parameter

## 🎯 Goal
Add `sigma_coordinate_space` parameter to decouple sigma sampling from `scale_jointly` flag, enabling explicit control over coordinate space interpretation.

## 📋 Implementation Steps

### Step 1: Add Parameter to Class Constructors

#### 1.1 Update SDFSamples.__init__()
```python
def __init__(
    self,
    list_mesh_paths,
    subsample,
    n_pts=500000,
    p_near_surface=0.4,
    p_further_from_surface=0.4,
    sigma_near=0.01,
    sigma_far=0.1,
    rand_function="normal",
    center_pts=True,
    norm_pts=False,
    scale_method="max_rad",
    scale_jointly=False,
    joint_scale_buffer=0.1,
    sigma_coordinate_space="legacy",  # NEW PARAMETER
    # ... rest of parameters
):
```

#### 1.2 Update MultiSurfaceSDFSamples.__init__()
```python
def __init__(
    self,
    list_mesh_paths,
    # ... existing parameters ...
    sigma_coordinate_space="legacy",  # NEW PARAMETER
    # ... rest of parameters
):
```

### Step 2: Update Documentation

#### 2.1 Add Parameter Documentation
```python
sigma_coordinate_space (str, optional): Controls coordinate space for sigma sampling. Defaults to "legacy".
    - "legacy": Current behavior - depends on scale_jointly flag
      * scale_jointly=False: Sigma applied after normalization (small values: 0.01-0.1)
      * scale_jointly=True: Sigma applied before normalization (large values: 0.5-5.0)
    - "original": Always sample in original coordinate space (large values: 0.5-5.0)
      * Ignores scale_jointly for sigma sampling
      * Consistent with scale_jointly=True behavior
      * RECOMMENDED for new code
```

### Step 3: Store Parameter and Add Validation

#### 3.1 Store in __init__()
```python
# Add to both classes
self.sigma_coordinate_space = sigma_coordinate_space
```

#### 3.2 Add Validation in preprocess_inputs()
```python
def preprocess_inputs(self):
    """
    Preprocess inputs to ensure they are in the correct format.
    """
    
    # Validate sigma_coordinate_space parameter
    valid_modes = ["legacy", "original"]
    if self.sigma_coordinate_space not in valid_modes:
        raise ValueError(
            f"sigma_coordinate_space must be one of {valid_modes}, "
            f"got: {self.sigma_coordinate_space}"
        )
    
    # Add deprecation warning for legacy mode
    if self.sigma_coordinate_space == "legacy":
        warnings.warn(
            "sigma_coordinate_space='legacy' is deprecated and will be removed in v3.0.0. "
            "Use sigma_coordinate_space='original' with appropriately scaled sigma values. "
            "See documentation for migration guide.",
            DeprecationWarning,
            stacklevel=2
        )
    
    # ... rest of existing preprocess_inputs logic
```

### Step 4: Update Core Sampling Functions

#### 4.1 Modify get_sample_data_dict() in SDFSamples
```python
def get_sample_data_dict(self, loc_mesh):
    # ... existing code until pt_sample_combos loop ...
    
    for idx_, (n_pts_, sigma_) in enumerate(self.pt_sample_combos):
        # Determine center_pts and norm_pts based on coordinate space mode
        if self.sigma_coordinate_space == "original":
            # Force original coordinate space for sigma sampling
            center_pts_for_sigma = False
            norm_pts_for_sigma = False
        else:  # legacy mode
            # Use existing scale_jointly-dependent behavior
            center_pts_for_sigma = self.center_pts
            norm_pts_for_sigma = self.norm_pts
            
        result_ = read_mesh_get_sampled_pts(
            loc_mesh,
            mean=[0, 0, 0],
            sigma=sigma_,
            n_pts=n_pts_,
            rand_function=self.rand_function,
            center_pts=center_pts_for_sigma,  # Modified based on coordinate space
            norm_pts=norm_pts_for_sigma,     # Modified based on coordinate space
            scale_method=self.scale_method,
            get_random=True,
            return_orig_mesh=False,
            return_new_mesh=False,
            fix_mesh=self.fix_mesh,
            register_to_mean_first=False if reference_mesh is None else True,
            mean_mesh=reference_mesh,
            uniform_pts_buffer=self.uniform_pts_buffer,
        )
        
        # ... rest of processing ...
```

#### 4.2 Modify get_sample_data_dict() in MultiSurfaceSDFSamples
```python
def get_sample_data_dict(self, loc_meshes):
    # ... existing code until pt_sample_combos loop ...
    
    for idx_, (n_pts_, sigma_) in enumerate(self.pt_sample_combos):
        # Determine center_pts and norm_pts based on coordinate space mode
        if self.sigma_coordinate_space == "original":
            # Force original coordinate space for sigma sampling
            center_pts_for_sigma = False
            norm_pts_for_sigma = False
        else:  # legacy mode
            # Use existing scale_jointly-dependent behavior
            center_pts_for_sigma = self.center_pts
            norm_pts_for_sigma = self.norm_pts
            
        result_ = read_meshes_get_sampled_pts(
            loc_meshes,
            mean=[0, 0, 0],
            sigma=sigma_,
            n_pts=n_pts_,
            rand_function=self.rand_function,
            center_pts=center_pts_for_sigma,  # Modified based on coordinate space
            norm_pts=norm_pts_for_sigma,     # Modified based on coordinate space
            scale_method=self.scale_method,
            get_random=True,
            fix_mesh=self.fix_mesh,
            register_to_mean_first=False if reference_mesh is None else True,
            mean_mesh=reference_mesh,
            uniform_pts_buffer=self.uniform_pts_buffer,
            # Multi surface specific
            mesh_to_scale=self.mesh_to_scale,
            scale_all_meshes=self.scale_all_meshes,
            center_all_meshes=self.center_all_meshes,
            icp_transform=icp_transform,
        )
        
        # ... rest of processing ...
```

### Step 5: Update Hash Parameters

#### 5.1 Add to get_hash_params() in SDFSamples
```python
def get_hash_params(self):
    list_hash_params = [
        self.n_pts,
        self.p_near_surface,
        self.sigma_near,
        self.p_further_from_surface,
        self.sigma_far,
        self.center_pts,
        self.norm_pts,
        self.scale_method,
        self.rand_function,
        self.reference_mesh,
        self.fix_mesh,
        self.scale_jointly,
        self.sigma_coordinate_space,  # NEW - ensures different cache for different modes
    ]
    return list_hash_params
```

#### 5.2 Add to get_hash_params() in MultiSurfaceSDFSamples
```python
def get_hash_params(self):
    list_hash_params = [
        self.center_pts,
        self.norm_pts,
        self.scale_method,
        self.rand_function,
        self.scale_all_meshes,
        self.center_all_meshes,
        self.reference_mesh,
        self.reference_object,
        False,
        self.fix_mesh,
        self.scale_jointly,
        self.sigma_coordinate_space,  # NEW - ensures different cache for different modes
    ]
    
    # ... rest of existing hash params ...
    return list_hash_params
```

### Step 6: Update Warnings

#### 6.1 Enhance Sigma Warning Logic
```python
def preprocess_inputs(self):
    # ... existing validation ...
    
    if self.scale_jointly is True:
        # ... existing scale_jointly warnings ...
        
        # Enhanced sigma warning for different coordinate space modes
        sigma_threshold = 0.2
        small_sigma_detected = False
        
        # Handle both single values and lists of sigma values
        sigma_near_values = getattr(self, 'sigma_near', None)
        sigma_far_values = getattr(self, 'sigma_far', None)
        
        # Check for small sigmas
        if sigma_near_values is not None:
            if isinstance(sigma_near_values, (list, tuple)):
                small_sigma_detected = any(s is not None and s < sigma_threshold for s in sigma_near_values)
            else:
                small_sigma_detected = sigma_near_values < sigma_threshold
        
        if not small_sigma_detected and sigma_far_values is not None:
            if isinstance(sigma_far_values, (list, tuple)):
                small_sigma_detected = any(s is not None and s < sigma_threshold for s in sigma_far_values)
            else:
                small_sigma_detected = sigma_far_values < sigma_threshold
                
        if small_sigma_detected:
            print("\n" + "="*80)
            print("WARNING: Potentially incorrect sigma values detected!")
            print("="*80)
            print(f"Current sigma values - sigma_near: {sigma_near_values}, sigma_far: {sigma_far_values}")
            
            if self.sigma_coordinate_space == "legacy":
                print(f"Values < {sigma_threshold} detected with scale_jointly=True.")
                print("This combination uses original coordinate space.")
                print()
                print("RECOMMENDATIONS:")
                print("1. Use sigma_coordinate_space='original' with large sigma values (0.5-5.0)")
                print("2. Or use sigma_coordinate_space='legacy' with scale_jointly=False and small sigma values (0.01-0.1)")
            else:  # original mode
                print(f"Values < {sigma_threshold} detected with sigma_coordinate_space='original'.")
                print("Original coordinate space requires larger sigma values.")
                print()
                print("RECOMMENDATIONS:")
                print("Use larger sigma values appropriate for your coordinate units:")
                print("- For medical data in mm: sigma_near=0.74, sigma_far=2.35")
                print("- For other units: scale proportionally")
            
            print("="*80 + "\n")
```

### Step 7: Add Import for Warnings

#### 7.1 Add to imports at top of file
```python
import warnings  # Add this import
```

### Step 8: Testing Strategy

#### 8.1 Test Cases to Add
```python
# Test 1: Legacy mode with scale_jointly=False (normalized space)
dataset = SDFSamples(
    sigma_near=0.01, sigma_far=0.1,
    scale_jointly=False,
    sigma_coordinate_space="legacy"
)

# Test 2: Legacy mode with scale_jointly=True (original space) 
dataset = SDFSamples(
    sigma_near=0.74, sigma_far=2.35,
    scale_jointly=True,
    sigma_coordinate_space="legacy"
)

# Test 3: Original mode regardless of scale_jointly
dataset = SDFSamples(
    sigma_near=0.74, sigma_far=2.35,
    scale_jointly=False,  # This gets ignored for sigma sampling
    sigma_coordinate_space="original"
)

# Test 4: Verify different cache files are created
# (different hash due to sigma_coordinate_space parameter)

# Test 5: Warning system with various combinations
```

#### 8.2 Validation Points
- [ ] Legacy mode produces identical results to current behavior
- [ ] Original mode produces consistent results regardless of scale_jointly
- [ ] Warning system triggers appropriately
- [ ] Cache files are properly separated by coordinate space mode
- [ ] Documentation is clear and complete

### Step 9: Implementation Order

1. **Step 1-2**: Add parameter and documentation
2. **Step 3**: Add validation and warnings 
3. **Step 4**: Implement core logic changes
4. **Step 5**: Update hash parameters
5. **Step 6-7**: Enhance warnings and add imports
6. **Step 8**: Test thoroughly
7. **Step 9**: Update examples and user guides

### Step 10: Migration Utilities (Future)

#### 10.1 Add Conversion Helper
```python
def convert_sigma_values(sigma_near, sigma_far, typical_mesh_radius=40.0):
    """
    Convert normalized-space sigma values to original-space sigma values.
    
    Args:
        sigma_near (float): Sigma value for near-surface sampling in normalized space
        sigma_far (float): Sigma value for far-surface sampling in normalized space  
        typical_mesh_radius (float): Typical radius of meshes in real-world units (e.g., mm)
        
    Returns:
        tuple: (new_sigma_near, new_sigma_far) for original coordinate space
    """
    return sigma_near * typical_mesh_radius, sigma_far * typical_mesh_radius
```

## 🎯 Expected Behavior After Implementation

### Legacy Mode
```python
# Maintains current behavior
dataset = SDFSamples(sigma_coordinate_space="legacy")  # Default
# scale_jointly=False -> normalized space (small sigmas)
# scale_jointly=True -> original space (large sigmas)
```

### Original Mode  
```python
# Always uses original coordinate space
dataset = SDFSamples(sigma_coordinate_space="original")
# Always original space regardless of scale_jointly value
# Always requires large sigma values (0.5-5.0)
```

This implementation provides a clean migration path while maintaining backward compatibility during the transition period.


