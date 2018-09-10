from __future__ import division, print_function
import pytest

from highway_env.road.road import Road
from highway_env.vehicle.control import ControlledVehicle

FPS = 15


def test_step():
    v = ControlledVehicle(road=None, position=[0, 0], velocity=20, heading=0)
    for _ in range(2 * FPS):
        v.step(dt=1/FPS)
    assert v.position[0] == pytest.approx(40)
    assert v.position[1] == pytest.approx(0)
    assert v.velocity == pytest.approx(20)
    assert v.heading == pytest.approx(0)


def test_lane_change():
    lane_width = 4.0
    road = Road.create_random_road(lanes_count=4, lane_width=lane_width, vehicles_count=0)
    lane = road.network.graph.values()[0].values()[0][0]
    v = ControlledVehicle(road=road, position=lane.position(0, 0), velocity=20, heading=0)
    v.act('LANE_RIGHT')
    for _ in range(3 * FPS):
        v.act()
        v.step(dt=1/FPS)
    assert v.velocity == pytest.approx(20)
    assert v.position[1] == pytest.approx(lane_width, abs=lane_width/4)
    assert v.lane_index == 1


def test_velocity_control():
    lane_width = 4.0
    road = Road.create_random_road(lanes_count=4, lane_width=lane_width, vehicles_count=0)
    lane = road.network.graph.values()[0].values()[0][0]
    v = ControlledVehicle(road=road, position=lane.position(0, 0), velocity=20, heading=0)
    v.act('FASTER')
    for _ in range(1 * FPS):
        v.act()
        v.step(dt=1/FPS)
    assert v.velocity == pytest.approx(20+v.DELTA_VELOCITY, abs=0.5)
    assert v.position[1] == pytest.approx(0)
    assert v.lane_index == 0
