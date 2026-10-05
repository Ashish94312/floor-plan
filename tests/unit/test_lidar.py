"""LiDAR tier reader (Stray Scanner): poses, intrinsics, depth -> world points, on synthetic box rooms."""

import numpy as np
import pytest

from scan.config import load_config
from scan.io.lidar import lidar_views, load_recording, pick_frames, read_odometry
from scan.synth.stray import write_box_room

CFG = load_config(overrides={"lidar": {"max_frames_per_room": 40, "min_dt_s": 0.0, "min_confidence": 2, "max_depth_m": 5.0}})


def test_pose_is_read_in_opencv_camera_axes(tmp_path):
    # Stray Scanner writes OpenCV camera axes (E25a: real recordings agree frame to frame only so): identity pose
    # = camera looking along world +z, image down along world +y
    (tmp_path / "odometry.csv").write_text("timestamp, frame, x, y, z, qx, qy, qz, qw, fx, fy, cx, cy\n"
                                           "0.0, 000000, 1.0, 2.0, 3.0, 0, 0, 0, 1, 100, 100, 50, 40\n")
    T = read_odometry(tmp_path / "odometry.csv")["T_wc"][0]
    assert np.allclose(T[:3, :3], np.eye(3)) and np.allclose(T[:3, 3], [1, 2, 3])


def test_pick_frames_spreads_over_time():
    t = np.arange(0, 60, 1 / 60)  # one minute at 60 fps
    idx = pick_frames(t, n_max=40, min_dt=0.5)
    assert len(idx) == 40 and idx[0] == 0 and idx[-1] == len(t) - 1
    assert len(pick_frames(t[:60], n_max=40, min_dt=0.5)) == 2  # 0 to 0.98 s, at least 0.5 s apart -> 2 frames


@pytest.mark.parametrize("size", [(4.0, 2.6, 3.0), (2.4, 2.8, 2.9)])
def test_box_room_points_land_on_its_walls(tmp_path, size):
    rec = write_box_room(tmp_path / "rec", size=size, cam=(size[0] / 2, 1.4, size[2] / 2))
    frames, warns = load_recording(rec, "room1", CFG)
    assert not warns and len(frames) == 24
    v = lidar_views(frames, CFG)
    P = v["pts"][v["mask"]]
    assert v["pts"].shape[1:3] == frames[0].depth.shape
    # every point on a face of the box (to the 1 mm of the depth PNGs)
    d = np.minimum(np.abs(P), np.abs(P - np.array(size))).min(1)
    assert np.percentile(d, 99) < 0.003
    assert np.allclose(P.max(0) - P.min(0), size, atol=0.01)


@pytest.mark.parametrize("turns", [0, 1, 2, 3])
def test_detector_box_maps_back_from_upright(turns):
    from scan.damage.detect import unrotate_box

    img = np.zeros((6, 10), np.uint8)
    img[1:3, 4:9] = 1  # box x 4-9, y 1-3 (pixel edges)
    r = np.rot90(img, turns)
    ys, xs = np.nonzero(r)
    assert unrotate_box([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], turns, img.shape) == [4, 1, 9, 3]


def test_upright_turns_from_gravity():
    from scan.io.lidar import upright_turns

    T = np.eye(4)
    # image right = world up (phone held portrait, sensor landscape): one counter-clockwise turn
    T[:3, 0], T[:3, 1], T[:3, 2] = [0, 1, 0], [1, 0, 0], [0, 0, -1]
    assert upright_turns(T) == 1
    T[:3, 0], T[:3, 1], T[:3, 2] = [1, 0, 0], [0, -1, 0], [0, 0, 1]  # image down = world down: upright already
    assert upright_turns(T) == 0
