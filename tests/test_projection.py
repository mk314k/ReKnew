import cv2
import numpy as np

from reknew.compositing import Layer, composite_layers
from reknew.light import estimate_reflectance, linear_to_srgb, srgb_to_linear
from reknew.projection import compensation_target, solve_projector_light
from reknew.projectors import SimulatedProjector
from tests.scenery import CAMERA_SIZE


def test_projector_light_is_zero_where_target_is_the_room(room):
    ambient = srgb_to_linear(room) * 0.3
    light, unreachable = solve_projector_light(ambient.copy(), ambient, estimate_reflectance(ambient), 1.0)
    assert not light.any()
    assert not unreachable.any()


def test_darker_than_room_is_unreachable(room):
    ambient = srgb_to_linear(room) * 0.3
    _, unreachable = solve_projector_light(ambient * 0.5, ambient, estimate_reflectance(ambient), 1.0)
    assert unreachable[50, 50]


def test_full_uniformity_trades_contrast_for_evenness(room, memory, overlay_corners):
    ambient = srgb_to_linear(room) * 0.3
    reflectance = estimate_reflectance(ambient)
    content, coverage = composite_layers([Layer(memory, overlay_corners)], CAMERA_SIZE)
    _, even = compensation_target(content, coverage, ambient, reflectance, 1.0, uniformity=1.0)
    _, contrasty = compensation_target(content, coverage, ambient, reflectance, 1.0, uniformity=0.5)
    assert even.ratio < contrasty.ratio


def test_compensation_is_closer_to_target_than_raw_projection(room, memory, overlay_corners):
    room_light, gain = 0.3, 1.0
    room_lin = srgb_to_linear(room)
    ambient = room_lin * room_light
    reflectance = estimate_reflectance(ambient)
    projector = SimulatedProjector()
    projector.mount_next_to_camera(CAMERA_SIZE)
    cam_to_proj = np.linalg.inv(projector.true_proj_to_cam)

    content, coverage = composite_layers([Layer(memory, overlay_corners)], CAMERA_SIZE)
    target, _ = compensation_target(content, coverage, ambient, reflectance, gain)
    light, _ = solve_projector_light(target, ambient, reflectance, gain)
    projector.show(cv2.warpPerspective(light, cam_to_proj, projector.size))
    compensated = projector.capture(room_lin, room_light, gain)

    projector_quad = cv2.perspectiveTransform(overlay_corners.reshape(-1, 1, 2), cam_to_proj).reshape(4, 2)
    raw, _ = composite_layers([Layer(memory, projector_quad)], projector.size)
    projector.show(np.clip(raw, 0, 1))
    uncompensated = projector.capture(room_lin, room_light, gain)

    area = coverage > 0
    target_srgb = linear_to_srgb(target).astype(float)

    def error(observed):
        return np.abs(linear_to_srgb(observed).astype(float) - target_srgb)[area].mean()

    assert error(compensated) < 0.75 * error(uncompensated)
