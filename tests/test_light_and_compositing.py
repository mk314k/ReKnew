import numpy as np

from reknew.compositing import Layer, composite_layers, order_quad, warp_to_quad
from reknew.light import estimate_reflectance, linear_to_srgb, srgb_to_linear


def test_srgb_round_trip_is_lossless():
    values = np.arange(256, dtype=np.uint8)
    assert np.array_equal(linear_to_srgb(srgb_to_linear(values)), values)


def test_reflectance_is_normalized(room):
    reflectance = estimate_reflectance(srgb_to_linear(room))
    assert reflectance.min() >= 0.0
    assert 0.95 <= reflectance.max() <= 1.0


def test_reflectance_ignores_light_level(room):
    room_lin = srgb_to_linear(room)
    assert np.allclose(estimate_reflectance(room_lin), estimate_reflectance(room_lin * 0.2), atol=1e-5)


def test_order_quad_is_clockwise_from_top_left_for_any_click_order():
    expected = [[100, 100], [300, 100], [300, 300], [100, 300]]
    for clicks in ([(100, 300), (300, 300), (300, 100), (100, 100)],
                   [(300, 100), (100, 300), (100, 100), (300, 300)]):
        assert order_quad(clicks).tolist() == expected


def test_warp_to_quad_alpha_covers_only_the_quad(memory):
    _, alpha = warp_to_quad(memory, order_quad([(100, 100), (300, 100), (300, 300), (100, 300)]), (640, 480))
    assert alpha[200, 200] == 1.0
    assert alpha[50, 50] == 0.0 and alpha[400, 500] == 0.0


def test_composite_layers_coverage_combines_layers(memory):
    left = Layer(memory, order_quad([(0, 0), (100, 0), (100, 100), (0, 100)]), opacity=0.5)
    right = Layer(memory, order_quad([(50, 0), (150, 0), (150, 100), (50, 100)]), opacity=0.5)
    _, coverage = composite_layers([left, right], (200, 120))
    assert np.isclose(coverage[50, 25], 0.5)
    assert np.isclose(coverage[50, 75], 0.75)
    assert coverage[50, 175] == 0.0
