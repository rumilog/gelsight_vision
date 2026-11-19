"""Weld-bead inspection: D555 image -> bead centerline -> GelSight touch stations.

ROS-free modules (importable and testable without rclpy):
    camera_model  pixel <-> plate-frame geometry (pinhole + plumb_bob)
    centerline    bead segmentation + centerline in a plate-frame ortho image
    planner       touch stations, tooltip yaw, plate reference points
    gel_features  locate features / the bead crown inside a GelSight frame
    calib         solvers for the spot-touch and bead-sweep calibrations
    viz           overlay drawing (cv2)

ROS nodes built on these: touch_bead_stations, calib_spot_touch,
calib_bead_sweep (all share bead.rig_node.BeadRigNode).
"""
