# Kangaroo and wombat integration

Use this entire blue_mountains_gazebo folder in place of your previous package. Your supplied landscape meshes, textures, model SDF, model config and setup script are unchanged.

- `demo_animal` is now the kangaroo at (2, 3, 0.003) metres.
- `wombat1` is at (4, 2, 0.003) metres.
- Both are static, face +Y, and use scale 1.0. Feet are 3 mm above sampled flat terrain.
- Kangaroo: 1.6 m tall, 3,116 visual triangles, 8 box colliders.
- Wombat: 1.05 m long, about 0.50 m tall, 5,904 visual triangles, 6 box colliders.
- The models use embedded COLLADA material colours; no external animal textures or network downloads are required.
- Collision shapes approximate body parts; fine ears, claws and muzzle detail remain visual geometry. No animation, walking, hopping, thermal signatures or controllers were added.
- The old animal was moved off its sloping rock to prevent clipping. All other world entities and settings are preserved.

## Launch (Gazebo Fortress)

```bash
source /path/to/blue_mountains_gazebo/setup_env.sh
ign sdf -k "$BLUE_MOUNTAINS_ROOT/models/kangaroo/model.sdf"
ign sdf -k "$BLUE_MOUNTAINS_ROOT/models/wombat/model.sdf"
ign sdf -k "$BLUE_MOUNTAINS_WORLD"
ign gazebo -v 4 "$BLUE_MOUNTAINS_WORLD"
```

For your existing drone launch, source setup_env.sh in the same terminal and keep using this package's worlds/large_demo.sdf. Gazebo/ROS are unavailable on this Mac, so simulator loading, contact and sensor behaviour still require a run on your Gazebo machine. The existing standing-person Fuel dependency is unchanged.

To move or duplicate an animal, edit its `<include>` in worlds/large_demo.sdf; use a unique `<name>` for every instance and set Z to the local terrain height. Pose values are x y z roll pitch yaw (metres/radians).

## Local verification

```bash
python3 validate_package.py
python3 validate_animals.py
```
