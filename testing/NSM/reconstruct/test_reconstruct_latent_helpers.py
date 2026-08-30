"""
Characterization of the latent-optimization helpers in ``reconstruct/main.py``, written
immediately before their move to ``latent_fit.py`` (plan §8.0.C). Everything asserts
behaviour as it stands, warts included: the bare-``Exception`` raises in the ``sdf_gt``
type check, and ``latent_norm_penalty`` silently computing the quadratic penalty when
asked for a barrier against a single target (documented in its docstring).

``preprocess_sdf_gt`` mutating its argument and also returning it was the third wart, and
it was fixed by plan §8.0.N (#55). The pin moved with the fix, to
``testing/NSM/test_caller_object_mutation.py``, where #55's other two sites are: one
issue, one file, rather than a copy per module.
"""

import numpy as np
import pytest
import torch

from NSM.reconstruct.main import (
    latent_norm_penalty,
    project_latent,
    reconstruct_latent_decoders_type_check,
    reconstruct_latent_get_lr_update_freq,
    reconstruct_latent_preprocess_sdf_gt,
    reconstruct_latent_pts_surface_type_check,
    reconstruct_latent_sdf_gt_type_check,
)


class TestSdfGtTypeCheck:
    def test_a_single_tensor_is_wrapped_in_a_list(self):
        sdf = torch.zeros(5, 1)
        result = reconstruct_latent_sdf_gt_type_check(sdf)
        assert result == [sdf]

    def test_a_single_ndarray_is_wrapped_in_a_list(self):
        sdf = np.zeros((5, 1))
        result = reconstruct_latent_sdf_gt_type_check(sdf)
        assert isinstance(result, list) and result[0] is sdf

    def test_a_list_passes_through_unwrapped(self):
        """
        Unwrapped, and — since §8.0.N (#55) — not the caller's own list. The elements are
        still the caller's objects; it is the container that is new, because the
        preprocess below assigns into it by index.
        """
        sdf_list = [torch.zeros(5, 1), None]
        result = reconstruct_latent_sdf_gt_type_check(sdf_list)
        assert result == sdf_list and result is not sdf_list
        assert result[0] is sdf_list[0]

    def test_an_unhandled_type_raises_a_bare_exception(self):
        with pytest.raises(Exception, match="Invalid sdf_gt type") as excinfo:
            reconstruct_latent_sdf_gt_type_check(42)
        assert type(excinfo.value) is Exception


class TestPtsSurfaceTypeCheck:
    def test_a_list_becomes_a_tensor(self):
        result = reconstruct_latent_pts_surface_type_check([0, 0, 1], device="cpu")
        assert isinstance(result, torch.Tensor)
        assert result.tolist() == [0, 0, 1]

    def test_an_ndarray_becomes_a_tensor(self):
        result = reconstruct_latent_pts_surface_type_check(np.array([0, 1]), device="cpu")
        assert isinstance(result, torch.Tensor)

    def test_a_tensor_passes_through_as_is(self):
        pts = torch.tensor([0, 1])
        assert reconstruct_latent_pts_surface_type_check(pts, device="cpu") is pts

    def test_an_unhandled_type_raises_value_error(self):
        with pytest.raises(ValueError, match="pts_surface must be"):
            reconstruct_latent_pts_surface_type_check(42, device="cpu")


class TestDecodersTypeCheck:
    def test_a_single_module_is_wrapped_in_a_list(self):
        decoder = torch.nn.Linear(2, 1)
        assert reconstruct_latent_decoders_type_check(decoder) == [decoder]

    def test_a_list_of_modules_passes_through(self):
        decoders = [torch.nn.Linear(2, 1)]
        assert reconstruct_latent_decoders_type_check(decoders) is decoders

    def test_a_list_containing_a_non_module_raises(self):
        with pytest.raises(ValueError, match="list of torch.nn.Module"):
            reconstruct_latent_decoders_type_check([torch.nn.Linear(2, 1), 42])

    def test_a_non_module_raises(self):
        with pytest.raises(ValueError, match="must be a torch.nn.Module"):
            reconstruct_latent_decoders_type_check(42)


class TestLrUpdateFreq:
    """``adjust_lr_every`` arithmetic: 0/None disable by overshooting the loop bound."""

    @pytest.mark.parametrize("n_updates", [0, None])
    def test_zero_or_none_means_never(self, n_updates):
        assert reconstruct_latent_get_lr_update_freq(n_updates, 100) == 101

    def test_even_division(self):
        assert reconstruct_latent_get_lr_update_freq(4, 100) == 25

    def test_uneven_division_floors(self):
        assert reconstruct_latent_get_lr_update_freq(7, 100) == 14

    def test_more_updates_than_iterations_clamps_to_every_step(self):
        assert reconstruct_latent_get_lr_update_freq(200, 100) == 1


class TestPreprocessSdfGt:
    def test_clamps_to_plus_minus_clamp_dist(self):
        sdf_gt = [torch.tensor([-2.0, -0.05, 0.05, 2.0])]
        result = reconstruct_latent_preprocess_sdf_gt(sdf_gt, clamp_dist=0.1, device="cpu")
        assert torch.equal(result[0], torch.tensor([-0.1, -0.05, 0.05, 0.1]))

    def test_clamp_dist_none_leaves_values_untouched(self):
        sdf_gt = [torch.tensor([-2.0, 2.0])]
        result = reconstruct_latent_preprocess_sdf_gt(sdf_gt, clamp_dist=None, device="cpu")
        assert torch.equal(result[0], torch.tensor([-2.0, 2.0]))

    def test_none_surfaces_are_skipped_not_dropped(self):
        sdf_gt = [None, torch.tensor([0.5])]
        result = reconstruct_latent_preprocess_sdf_gt(sdf_gt, clamp_dist=0.1, device="cpu")
        assert result[0] is None and len(result) == 2


class TestProjectLatent:
    """In-place norm clamp; returns None. Production never sets ``latent_norm``."""

    def _latent_with_norm(self, norm):
        latent = torch.zeros(1, 4)
        latent[0, 0] = norm
        return latent

    def test_a_norm_above_the_range_is_pulled_down_to_max(self):
        latent = self._latent_with_norm(5.0)
        assert project_latent(latent, (1.0, 2.0)) is None
        assert latent.norm(p=2).item() == pytest.approx(2.0, rel=1e-6)

    def test_a_norm_below_the_range_is_pushed_up_to_min(self):
        latent = self._latent_with_norm(0.5)
        project_latent(latent, (1.0, 2.0))
        assert latent.norm(p=2).item() == pytest.approx(1.0, rel=1e-6)

    def test_a_norm_inside_the_range_is_left_alone(self):
        latent = self._latent_with_norm(1.5)
        project_latent(latent, (1.0, 2.0))
        assert latent.norm(p=2).item() == pytest.approx(1.5, rel=1e-6)

    def test_a_single_value_projects_onto_that_norm(self):
        latent = self._latent_with_norm(5.0)
        project_latent(latent, 3.0)
        assert latent.norm(p=2).item() == pytest.approx(3.0, rel=1e-6)

    @pytest.mark.parametrize("bad_spec", [(1.0, 2.0, 3.0), "1.0"])
    def test_a_bad_spec_raises_value_error(self, bad_spec):
        with pytest.raises(ValueError, match="latent_norm must be"):
            project_latent(self._latent_with_norm(1.5), bad_spec)


class TestLatentNormPenalty:
    def _latent_with_norm(self, norm):
        latent = torch.zeros(1, 4)
        latent[0, 0] = norm
        return latent

    def test_quadratic_inside_the_range_is_zero(self):
        penalty = latent_norm_penalty(self._latent_with_norm(1.5), (1.0, 2.0))
        assert float(penalty) == 0.0

    def test_quadratic_below_the_range_grows_from_min(self):
        penalty = latent_norm_penalty(self._latent_with_norm(0.5), (1.0, 2.0))
        assert float(penalty) == pytest.approx(0.25, rel=1e-6)

    def test_quadratic_above_the_range_grows_from_max(self):
        penalty = latent_norm_penalty(self._latent_with_norm(3.0), (1.0, 2.0))
        assert float(penalty) == pytest.approx(1.0, rel=1e-6)

    def test_the_weight_scales_the_penalty(self):
        penalty = latent_norm_penalty(self._latent_with_norm(3.0), (1.0, 2.0), penalty_weight=0.5)
        assert float(penalty) == pytest.approx(0.5, rel=1e-6)

    def test_huber_inside_the_range_is_zero(self):
        penalty = latent_norm_penalty(self._latent_with_norm(1.5), (1.0, 2.0), penalty_type="huber")
        assert float(penalty) == 0.0

    def test_huber_within_delta_is_quadratic(self):
        # delta = 10% of the (1.0, 2.0) range = 0.1; diff 0.05 <= delta -> 0.5 * diff**2
        penalty = latent_norm_penalty(
            self._latent_with_norm(2.05), (1.0, 2.0), penalty_type="huber"
        )
        assert float(penalty) == pytest.approx(0.5 * 0.05**2, rel=1e-5)

    def test_huber_beyond_delta_is_linear(self):
        # diff 0.5 > delta 0.1 -> delta * (diff - 0.5 * delta)
        penalty = latent_norm_penalty(self._latent_with_norm(2.5), (1.0, 2.0), penalty_type="huber")
        assert float(penalty) == pytest.approx(0.1 * (0.5 - 0.05), rel=1e-5)

    def test_single_target_quadratic(self):
        penalty = latent_norm_penalty(self._latent_with_norm(3.0), 2.0)
        assert float(penalty) == pytest.approx(1.0, rel=1e-6)

    def test_single_target_barrier_silently_computes_the_quadratic(self):
        """Documented in the docstring; pinned so a rewrite of the branch is deliberate."""
        barrier = latent_norm_penalty(self._latent_with_norm(3.0), 2.0, penalty_type="barrier")
        quadratic = latent_norm_penalty(self._latent_with_norm(3.0), 2.0, penalty_type="quadratic")
        assert float(barrier) == float(quadratic)

    @pytest.mark.parametrize("target", [2.0, (1.0, 2.0)])
    def test_an_unknown_penalty_type_raises_on_both_branches(self, target):
        with pytest.raises(ValueError, match="Unknown penalty_type"):
            latent_norm_penalty(self._latent_with_norm(1.5), target, penalty_type="cubic")
