import pymskt
from pymskt.mesh import BoneMesh, CartilageMesh
from scipy.stats import entropy

CART_REGIONS = (
    2, # med tib
    3, # lat tib
    4, # pat
    11, # troch
    12, # med wb fem
    13, # lat wb fem
    14, # med post fem
    15  # lat post fem
)


def compare_cart_thickness(
        orig_meshes,
        recon_meshes,
        cart_regions=CART_REGIONS,
        regions_label='labels',
):
    orig_bone, orig_cart = orig_meshes
    recon_bone, recon_cart = recon_meshes
    # Check Types & Create Bone/CartilageMesh objects as appropriate
    # RECON_BONE
    if isinstance(recon_bone, pymskt.mesh.BoneMesh):
        pass
    elif isinstance(recon_bone, pymskt.mesh.Mesh):
        recon_bone = BoneMesh(recon_bone.mesh)
    else:
        recon_bone = BoneMesh(recon_bone)

    # RECON_CART
    if isinstance(recon_cart, pymskt.mesh.CartilageMesh):
        pass
    elif isinstance(recon_cart, pymskt.mesh.Mesh):
        recon_cart = CartilageMesh(recon_cart.mesh)
    else:
        recon_cart = CartilageMesh(recon_cart)
    
    # ORIG_BONE
    if isinstance(orig_bone, pymskt.mesh.BoneMesh):
        pass
    elif isinstance(orig_bone, pymskt.mesh.Mesh):
        orig_bone = BoneMesh(orig_bone.mesh)
    else:
        orig_bone = BoneMesh(orig_bone)

    # transfer region scalars to recon_bone
    # should add 'labels' to the reconned bone (these are cartialge regions)
    recon_bone.copy_scalars_from_other_mesh_to_current(orig_bone, orig_scalars_name=regions_label)

    # compute cart thickness for bone
    # this should add a new caritalge thickness array to bone - test to make sure doesnt cause issues.
    recon_bone.list_cartilage_meshes = recon_cart
    recon_bone.calc_cartilage_thickness()

    dict_results = {}

    for cart_region in cart_regions:
        # MEAN difference
        orig_mean = orig_bone.get_cart_thickness_mean(cart_region)
        recon_mean = recon_bone.get_cart_thickness_mean(cart_region)
        mean_diff = orig_mean - recon_mean

        # STD difference
        orig_std = orig_bone.get_cart_thickness_std(cart_region)
        recon_std = recon_bone.get_cart_thickness_std(cart_region)
        std_diff = orig_std - recon_std

        dict_results[f'func_{cart_region}_mean_thick_diff'] = mean_diff
        dict_results[f'func_{cart_region}_std_thick_diff'] = std_diff

    orig_array = orig_bone.get_scalar('thickness (mm)')
    recon_array = recon_bone.get_scalar('thickness (mm)')

    # Compute KL divergence between two distributions
    thickness_kld = entropy(orig_array, qk=recon_array)
    
    dict_results['func_thickness_kld'] = thickness_kld

    return dict_results
