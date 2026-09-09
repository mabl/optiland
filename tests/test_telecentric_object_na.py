"""Focused object-space telecentric NA regressions for issue #797."""

import math
from typing import TYPE_CHECKING
from unittest.mock import Mock

import numpy as np
import pytest

import optiland.backend as be
from optiland.materials.ideal import IdealMaterial
from optiland.optic import Optic
from optiland.rays.ray_aiming.paraxial import ParaxialRayAimer
from tests.utils import assert_allclose

if TYPE_CHECKING:
    from optiland._types import ScalarOrArray


def _plane_optic(
    material: IdealMaterial,
    aperture_value: "ScalarOrArray" = 0.6,
    *,
    telecentric: bool = True,
) -> Optic:
    """Build homogeneous planes with stop/image 10/15 mm from the object."""
    optic = Optic()
    optic.surfaces.add(index=0, thickness=10.0, material=material)
    optic.surfaces.add(index=1, thickness=5.0, material=material, is_stop=True)
    optic.surfaces.add(index=2, material=material)
    optic.set_aperture("objectNA", aperture_value)
    optic.fields.set_type("object_height")
    optic.fields.add(y=0.0)
    optic.fields.add(y=2.0)
    optic.obj_space_telecentric = telecentric
    optic.fields.set_telecentric(telecentric)
    optic.wavelengths.add(0.45)
    optic.wavelengths.add(0.55, is_primary=True)
    optic.ray_tracer.set_aiming("paraxial")
    return optic


@pytest.mark.parametrize("index,na", [(1.0, 0.6), (1.5, 0.6), (1.5, 1.2)])
@pytest.mark.parametrize("telecentric", [True, False], ids=["telecentric", "control"])
def test_public_plane_trace_has_physical_na_and_heights(
    set_test_backend: None, index: float, na: float, telecentric: bool
) -> None:
    """Public tracing obeys NA = n sin(theta) and free-propagation geometry."""
    optic = _plane_optic(IdealMaterial(index), na, telecentric=telecentric)
    px = be.array([0.0, 1.0, 0.0])
    py = be.array([0.0, 0.0, -1.0])
    rays = optic.trace_generic(be.zeros(3), be.zeros(3), px, py, 0.55)

    sine = na / index
    expected_l = np.array([0.0, sine, 0.0])
    expected_m = np.array([0.0, 0.0, -sine])
    expected_n = np.array([1.0, math.sqrt(1 - sine**2), math.sqrt(1 - sine**2)])
    for actual, expected in zip(
        (rays.L, rays.M, rays.N), (expected_l, expected_m, expected_n), strict=True
    ):
        assert np.isfinite(be.to_numpy(actual)).all()
        assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
    assert_allclose(index * be.sqrt(rays.L**2 + rays.M**2), [0.0, na, na])
    assert_allclose(rays.L**2 + rays.M**2 + rays.N**2, 1.0)
    assert_allclose(rays.i, 1.0)
    for surface, distance in ((optic.surfaces[1], 10.0), (optic.surfaces[2], 15.0)):
        assert_allclose(surface.x, distance * expected_l / expected_n)
        assert_allclose(surface.y, distance * expected_m / expected_n)
    assert_allclose(rays.x, 15.0 * expected_l / expected_n)
    assert_allclose(rays.y, 15.0 * expected_m / expected_n)


@pytest.mark.parametrize("index", [1.0, 1.5], ids=["air", "immersion"])
@pytest.mark.parametrize("vignetted", [False, True])
def test_pupil_sampling_and_vignetting_preserve_slopes(
    set_test_backend: None, index: float, vignetted: bool
) -> None:
    """Chief, interior and cardinal pupils scale slope, not numerical aperture."""
    na = 0.6
    optic = _plane_optic(IdealMaterial(index), be.array([na]) if vignetted else na)
    vx, vy = (0.2, 0.4) if vignetted else (0.0, 0.0)
    for field in optic.fields:
        field.vx, field.vy = vx, vy
    px = np.array([0.0, 0.3, 1.0, -1.0, 0.0, 0.0])
    py = np.array([0.0, -0.4, 0.0, 0.0, 1.0, -1.0])
    x, y, z, L, M, N = ParaxialRayAimer(optic).aim_rays(
        (be.full(6, 0.3), be.full(6, 0.4)),
        be.full(6, 0.55),
        (be.array(px), be.array(py)),
    )

    marginal_slope = math.tan(math.asin(na / index))
    slope_x = px * (1 - vx) * marginal_slope
    slope_y = py * (1 - vy) * marginal_slope
    expected_n = 1 / np.sqrt(1 + slope_x**2 + slope_y**2)
    assert_allclose(x, 0.6)
    assert_allclose(y, 0.8)
    assert_allclose(z, optic.object_surface.geometry.cs.z)
    assert_allclose(L / N, slope_x, rtol=1e-10, atol=1e-12)
    assert_allclose(M / N, slope_y, rtol=1e-10, atol=1e-12)
    assert_allclose(L, slope_x * expected_n)
    assert_allclose(M, slope_y * expected_n)
    assert_allclose(N, expected_n)
    assert_allclose(L**2 + M**2 + N**2, 1.0)
    interior_na = index * be.sqrt(L[1] ** 2 + M[1] ** 2)
    linear_na = na * math.hypot(px[1] * (1 - vx), py[1] * (1 - vy))
    assert not np.isclose(be.to_numpy(interior_na), linear_na, rtol=1e-3)


def test_launch_uses_object_medium_not_first_surface_post_medium(
    set_test_backend: None,
) -> None:
    """Launch NA belongs to the incident object medium, before any refraction."""
    optic = _plane_optic(IdealMaterial(1.5))
    optic.updater.set_material(IdealMaterial(2.0), 1)
    _, _, _, L, M, N = ParaxialRayAimer(optic).aim_rays(
        (be.zeros(2), be.zeros(2)),
        be.full(2, 0.55),
        (be.array([1.0, 0.0]), be.array([0.0, -1.0])),
    )

    assert_allclose(optic.object_surface.material_post.n(0.55), 1.5)
    assert_allclose(optic.surfaces[1].material_post.n(0.55), 2.0)
    assert_allclose(L, [0.4, 0.0])
    assert_allclose(M, [0.0, -0.4])
    assert_allclose(N, math.sqrt(1 - 0.4**2))
    assert_allclose(1.5 * be.sqrt(L**2 + M**2), 0.6)


def test_launch_uses_primary_wavelength_not_requested_wavelength(
    set_test_backend: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dispersive object medium defines launch NA at the primary wavelength."""
    material = IdealMaterial(1.5)

    def refractive_index(wavelength: "ScalarOrArray") -> "ScalarOrArray":
        """Separate the primary index 1.5 from the requested-ray index 1.7."""
        return 1.5 + 2.0 * (wavelength - 0.55)

    index_lookup = Mock(side_effect=refractive_index)
    monkeypatch.setattr(material, "n", index_lookup)
    optic = _plane_optic(material)
    _, _, _, L, M, N = ParaxialRayAimer(optic).aim_rays(
        (be.zeros(2), be.zeros(2)),
        be.full(2, 0.65),
        (be.array([1.0, 0.0]), be.array([0.0, 1.0])),
    )

    index_lookup.assert_called_once_with(0.55)
    assert_allclose(L, [0.4, 0.0])
    assert_allclose(M, [0.0, 0.4])
    assert_allclose(N, math.sqrt(1 - 0.4**2))
    assert_allclose(1.5 * be.sqrt(L**2 + M**2), 0.6)


def test_torch_na_and_index_gradients_for_chief_interior_and_marginal(
    set_test_backend: None,
) -> None:
    """Fresh direct launches retain analytical NA/index derivatives in normal mode."""
    if be.get_backend() != "torch":
        pytest.skip("Autograd requires the torch backend.")
    import torch

    na = torch.tensor(0.6, dtype=torch.float64, requires_grad=True)
    index = torch.tensor([1.5], dtype=torch.float64, requires_grad=True)
    material = IdealMaterial(1.5)
    # Install the parameter before the first material lookup, avoiding issue #794.
    material.index = index
    optic = _plane_optic(material, na)
    aimer = ParaxialRayAimer(optic)
    px = np.array([0.0, 0.3, 1.0])
    py = np.array([0.0, -0.4, 0.0])

    # Differentiate the unit vector (px*t, py*t, 1), t = NA/sqrt(n^2 - NA^2).
    t = 0.6 / math.sqrt(1.5**2 - 0.6**2)
    radius_squared = px**2 + py**2
    denominator = (1 + radius_squared * t**2) ** 1.5
    direction_dt = np.array([px, py, -radius_squared * t]) / denominator
    dt_dna = 1.5**2 / (1.5**2 - 0.6**2) ** 1.5
    dt_dn = -0.6 * 1.5 / (1.5**2 - 0.6**2) ** 1.5

    for _ in range(2):
        _, _, _, L, M, N = aimer.aim_rays(
            (be.zeros(3), be.zeros(3)),
            be.full(3, 0.55),
            (be.array(px), be.array(py)),
        )
        components = torch.stack((L, M, N)).flatten()
        for i, component in enumerate(components):
            grad_na, grad_index = torch.autograd.grad(
                component, (na, index), retain_graph=i < len(components) - 1
            )
            derivative = direction_dt.flat[i]
            assert_allclose(grad_na, derivative * dt_dna, rtol=1e-10, atol=1e-12)
            assert_allclose(grad_index, derivative * dt_dn, rtol=1e-10, atol=1e-12)


def test_float_by_stop_retains_legacy_telecentric_slope(
    set_test_backend: None,
) -> None:
    """Positive sub-unit FloatByStop values retain the legacy index-free slope."""
    optic = _plane_optic(IdealMaterial(1.5))
    optic.set_aperture("float_by_stop_size", 0.6)
    px = be.array([0.0, 0.3, 1.0])
    py = be.array([0.0, -0.4, 0.0])
    _, _, _, L, M, N = ParaxialRayAimer(optic).aim_rays(
        (be.zeros(3), be.zeros(3)), be.full(3, 0.55), (px, py)
    )

    legacy_slope = 0.6 / math.sqrt(1 - 0.6**2)
    assert_allclose(L / N, px * legacy_slope)
    assert_allclose(M / N, py * legacy_slope)
    assert_allclose(L**2 + M**2 + N**2, 1.0)
    assert_allclose(L[2], 0.6)
    assert_allclose(N[2], 0.8)


def test_zero_object_na_launches_finite_axial_rays(set_test_backend: None) -> None:
    """Zero NA naturally collapses chief and nonchief pupils onto the +z axis."""
    optic = _plane_optic(IdealMaterial(1.5), 0.0)
    x, y, z, L, M, N = ParaxialRayAimer(optic).aim_rays(
        (be.zeros(3), be.ones(3)),
        be.full(3, 0.55),
        (be.array([0.0, 0.3, 1.0]), be.array([0.0, -0.4, 0.0])),
    )

    for value in (x, y, z, L, M, N):
        assert np.isfinite(be.to_numpy(value)).all()
    assert_allclose(x, 0.0)
    assert_allclose(y, 2.0)
    assert_allclose(L, 0.0)
    assert_allclose(M, 0.0)
    assert_allclose(N, 1.0)
