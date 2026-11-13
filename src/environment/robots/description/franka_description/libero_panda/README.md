# LIBERO-compatible Panda model

This directory vendors the composed `Panda` + `PandaGripper` MuJoCo model and
mesh assets from `robosuite==1.4.0`, the robosuite version pinned by LIBERO.
`panda.xml` was produced through robosuite's model composer and then made
standalone by using repository-relative mesh paths and explicitly declaring
radian joint angles. Runtime does not import robosuite.

Robot contract used by this project:

- 7 Panda revolute joints: `robot0_joint1` ... `robot0_joint7`
- parallel two-finger `PandaGripper`
- torque-controlled arm and position-controlled fingers
- `robot0_eye_in_hand` wrist camera
- exact LIBERO `OnTheGroundPanda` initial joint pose
- robosuite `OSC_POSE` scaling: 0.05 m translation and 0.5 rad rotation for a
  normalized action of magnitude 1

The upstream robosuite license is preserved in
[`ROBOSUITE_LICENSE`](ROBOSUITE_LICENSE). The corresponding ROS visualization
description is generated from `../panda_description/panda.urdf.in`.

