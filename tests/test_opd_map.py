"""OPD map values, intensity selection, and interpolation compatibility."""

# ruff: noqa: I002

from collections.abc import Sequence
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.interpolate import griddata

import optiland.backend as be
from optiland.samples.objectives import DoubleGauss
from optiland.wavefront import OPD, WavefrontData
from optiland.wavefront.opd import OPDData

FIELD = (0.0, 0.0)
WAVELENGTH = 0.55
X = np.array([-0.5, 0.5, -0.5, 0.5])
Y = np.array([-0.5, -0.5, 0.5, 0.5])


def _make_analysis(
    opd: Sequence[float] | np.ndarray,
    intensity: Sequence[float] | np.ndarray,
    x: np.ndarray = X,
    y: np.ndarray = Y,
) -> tuple[OPD, WavefrontData]:
    """Supply controlled cached data without tracing or refitting a reference."""
    data = WavefrontData(
        pupil_x=be.array(x),
        pupil_y=be.array(y),
        pupil_z=be.array(np.zeros_like(x)),
        opd=be.array(np.asarray(opd, dtype=float)),
        intensity=be.array(np.asarray(intensity, dtype=float)),
        radius=1.0,
    )
    analysis = OPD.__new__(OPD)
    analysis.fields = [FIELD]
    analysis.wavelengths = [WAVELENGTH]
    analysis.distribution = SimpleNamespace(x=be.array(x), y=be.array(y))
    analysis.data = {(FIELD, WAVELENGTH): data}
    if be.get_backend() == "torch":
        import torch

        for value in (
            analysis.distribution.x,
            analysis.distribution.y,
            data.opd,
            data.intensity,
        ):
            assert isinstance(value, torch.Tensor)
    return analysis, data


def _assert_same_map(actual: OPDData, expected: OPDData) -> None:
    """Compare interpolation domains and values without accepting all-NaN maps."""
    for key in ("x", "y"):
        np.testing.assert_array_equal(actual[key], expected[key])
    assert np.isfinite(expected["z"]).any()
    np.testing.assert_array_equal(np.isnan(actual["z"]), np.isnan(expected["z"]))
    np.testing.assert_allclose(
        actual["z"], expected["z"], rtol=0, atol=1e-12, equal_nan=True
    )


@pytest.mark.parametrize(
    "opd",
    [[0.5] * 4, [-0.5, 0.25, 0.75, -0.5]],
    ids=["constant", "signed-nonconstant"],
)
@pytest.mark.parametrize(
    "intensity",
    [[0.2] * 4, [0.2, 0.4, 0.6, 0.8], [1e-12, 0.3, 0.9, 1.0], [2.0, 0.3, 1.2, 4.0]],
    ids=["uniform-attenuation", "nonuniform", "tiny-positive", "relative-above-one"],
)
def test_positive_intensities_do_not_rescale_opd(set_test_backend, opd, intensity):
    analysis, data = _make_analysis(opd, np.ones(4))
    baseline = analysis.generate_opd_map(num_points=9)
    baseline_rms = be.to_numpy(analysis.rms()).copy()
    data.intensity = be.array(intensity)

    _assert_same_map(analysis.generate_opd_map(num_points=9), baseline)
    np.testing.assert_allclose(
        be.to_numpy(analysis.rms()), baseline_rms, rtol=0, atol=1e-12
    )


def test_traced_map_is_unchanged_by_cached_attenuation(set_test_backend):
    analysis = OPD(DoubleGauss(), (0.0, 1.0), WAVELENGTH)
    data = analysis.get_data(analysis.fields[0], analysis.wavelengths[0])
    baseline = analysis.generate_opd_map(num_points=32)
    assert np.any(baseline["z"][np.isfinite(baseline["z"])] != 0.0)
    data.intensity = data.intensity * 0.25

    _assert_same_map(analysis.generate_opd_map(num_points=32), baseline)


@pytest.mark.parametrize("constant", [-0.5, 0.0, 0.5])
def test_constant_opd_preserved_over_interpolation_domain(set_test_backend, constant):
    analysis, _ = _make_analysis([constant] * 4, [0.2, 0.4, 0.6, 0.8])
    result = analysis.generate_opd_map(num_points=9)
    inside = (np.abs(result["x"]) <= 0.5) & (np.abs(result["y"]) <= 0.5)

    assert inside.sum() == 25
    np.testing.assert_array_equal(np.isfinite(result["z"]), inside)
    np.testing.assert_allclose(result["z"][inside], constant, rtol=0, atol=1e-12)
    assert np.isnan(result["z"][~inside]).all()


@pytest.mark.parametrize("excluded_intensity", [0.0, -0.2])
@pytest.mark.parametrize(
    "excluded_xy", [(0.0, 0.0), (0.9, 0.0)], ids=["interior", "outside-hull"]
)
def test_excluded_samples_do_not_affect_values_or_support(
    set_test_backend, excluded_intensity, excluded_xy
):
    values = np.array([-0.5, 0.25, 0.75, -0.5])
    retained, _ = _make_analysis(values, np.ones(4))
    augmented, _ = _make_analysis(
        np.r_[values, 1e6],
        np.r_[np.ones(4), excluded_intensity],
        x=np.r_[X, excluded_xy[0]],
        y=np.r_[Y, excluded_xy[1]],
    )

    _assert_same_map(
        augmented.generate_opd_map(num_points=21),
        retained.generate_opd_map(num_points=21),
    )


def test_sample_node_values_are_unscaled_signed_opd(set_test_backend):
    values = np.array([-0.5, 0.25, 0.75, -0.5])
    analysis, _ = _make_analysis(values, [0.2, 0.4, 0.6, 0.8])
    result = analysis.generate_opd_map(num_points=5)
    # Grid [-1, -.5, 0, .5, 1] includes all four original sample coordinates.
    at_nodes = result["z"][[1, 1, 3, 3], [1, 3, 1, 3]]

    np.testing.assert_allclose(at_nodes, values, rtol=0, atol=1e-12)


def test_input_arrays_and_cache_not_mutated(set_test_backend):
    analysis, data = _make_analysis(
        [-0.5, 0.25, 0.75, -0.5, 1e6],
        [0.2, 0.4, 0.6, 0.8, 0.0],
        x=np.r_[X, 0.0],
        y=np.r_[Y, 0.0],
    )
    objects = [
        (analysis.distribution, "x"),
        (analysis.distribution, "y"),
        (data, "pupil_x"),
        (data, "pupil_y"),
        (data, "pupil_z"),
        (data, "opd"),
        (data, "intensity"),
    ]
    snapshots = [
        (obj, key, getattr(obj, key), be.to_numpy(getattr(obj, key)).copy())
        for obj, key in objects
    ]
    for _ in range(2):
        analysis.generate_opd_map(num_points=9)

    assert analysis.get_data(FIELD, WAVELENGTH) is data
    for obj, key, original, snapshot in snapshots:
        assert getattr(obj, key) is original
        np.testing.assert_array_equal(be.to_numpy(getattr(obj, key)), snapshot)


@pytest.mark.parametrize("num_points", [5, 9, 17])
def test_grid_and_cubic_interpolation_contract(set_test_backend, num_points):
    x, y = np.r_[X, 0.0], np.r_[Y, 0.0]
    values = np.array([0.1, -0.45, 0.8, -0.2, 0.7])
    analysis, _ = _make_analysis(values, np.ones(5), x=x, y=y)
    actual = analysis.generate_opd_map(num_points=num_points)
    xx, yy = np.meshgrid(np.linspace(-1, 1, num_points), np.linspace(-1, 1, num_points))
    points = np.column_stack((x, y))
    expected_z = griddata(points, values, (xx, yy), method="cubic")

    assert set(actual) == {"x", "y", "z"}
    for key in ("x", "y", "z"):
        assert isinstance(actual[key], np.ndarray)
        assert actual[key].shape == (num_points, num_points)
    _assert_same_map(actual, {"x": xx, "y": yy, "z": expected_z})
    if num_points > 5:
        linear_z = griddata(points, values, (xx, yy), method="linear")
        assert np.nanmax(np.abs(expected_z - linear_z)) > 1e-3


def test_default_resolution_preserved(set_test_backend):
    analysis, _ = _make_analysis([0.5] * 4, np.ones(4))
    actual = analysis.generate_opd_map()

    for key in ("x", "y", "z"):
        assert isinstance(actual[key], np.ndarray)
        assert actual[key].shape == (256, 256)
    np.testing.assert_array_equal(actual["x"][0], np.linspace(-1, 1, 256))
    np.testing.assert_array_equal(actual["y"][:, 0], np.linspace(-1, 1, 256))


def test_distribution_coordinates_define_interpolation_domain(set_test_backend):
    analysis, data = _make_analysis([0.5] * 4, np.ones(4))
    # Exit-pupil coordinates need not equal the distribution's sampling chart.
    data.pupil_x = be.array(2 * X)
    data.pupil_y = be.array(3 * Y)
    result = analysis.generate_opd_map(num_points=9)
    inside = (np.abs(result["x"]) <= 0.5) & (np.abs(result["y"]) <= 0.5)

    np.testing.assert_array_equal(np.isfinite(result["z"]), inside)


def test_torch_requires_grad_inputs_return_numpy_without_mutation(set_test_backend):
    if be.get_backend() != "torch":
        pytest.skip("Torch-only conversion boundary check.")
    analysis, data = _make_analysis([0.5] * 4, [0.2, 0.4, 0.6, 0.8])
    tensors = [
        analysis.distribution.x,
        analysis.distribution.y,
        data.opd,
        data.intensity,
    ]
    for tensor in tensors:
        tensor.requires_grad_(True)
    snapshots = [tensor.detach().clone() for tensor in tensors]
    result = analysis.generate_opd_map(num_points=9)

    assert all(isinstance(result[key], np.ndarray) for key in ("x", "y", "z"))
    np.testing.assert_allclose(result["z"][4, 4], 0.5, rtol=0, atol=1e-12)
    for tensor, snapshot in zip(tensors, snapshots, strict=True):
        assert tensor.requires_grad
        assert tensor.grad is None
        np.testing.assert_array_equal(be.to_numpy(tensor), be.to_numpy(snapshot))
