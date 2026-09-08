# Blue Mountains replacement environment — Gazebo Fortress 6.18

This package replaces the scenery in your supplied `large_demo.sdf`. It contains the environment only; keep your existing ROS 2 launch, Parrot model, controllers, lidar, bridges and search code.

## Use with your current simulation

1. Copy this entire `blue_mountains_gazebo` folder to your ROS/Gazebo machine, preserving its internal folders.
2. In the Bash terminal where you normally launch the simulation, run:

   ```bash
   source /path/to/blue_mountains_gazebo/setup_env.sh
   ```

3. Point your existing launch's world-file setting at the path printed by that script (`$BLUE_MOUNTAINS_WORLD`). If your launch uses a hard-coded package world path, replace that package's `large_demo.sdf` with the supplied `worlds/large_demo.sdf`, retaining your usual recovery/backup workflow. The resource-path setup is still required unless `models/blue_mountains` is placed in an already configured model directory.
4. Launch using your existing command. No drone or lidar is included in this environment package.

For an environment-only inspection, after sourcing the script:

```bash
ign sdf -k "$BLUE_MOUNTAINS_WORLD"
ign sdf -k "$BLUE_MOUNTAINS_ROOT/models/blue_mountains/model.sdf"
ign gazebo -v 4 "$BLUE_MOUNTAINS_WORLD"
```

The world retains the original world-level plugin setup (no explicit plugins). It relies on the same server/launch configuration as your original world. Do not replace your existing sensor-system configuration with this inspection command. The original Fuel `Standing person` asset remains an external dependency; it must already be cached or downloadable.

## Scale and coordinates

- Terrain footprint: **100 × 80.3 m**. Scenery, including trees and hills, is uniformly scaled from the Blender landscape.
- Terrain XY limits: approximately **X −33.77 to 66.23 m; Y −8.77 to 71.54 m**.
- The Three Sisters remain the primary landmark; they are now approximately 25–30 m tall above their local base, not real-world geographic scale.
- World origin `(0, 0, 0)` lies on a flattened launch clearing, with a 7 m flat radius and a blended outer slope. Trees are cleared within an 8 m radius.
- The drone, person and original demo models are **not scaled**. Existing target and demo poses are retained, including the hidden fire/burnt trees at Z = −30.
- Existing waypoint/search limits that only cover the old ~25 × 25 m world will not automatically cover the entire new landscape. Outside the launch clearing, altitude commands must account for terrain height.
- The human remains at its original position near the launch area to avoid changing your task code's assumptions. If you relocate it for a search trial, move `person1` and `person1_heat` together, maintaining their original 0.85 m Z offset and accounting for local ground height.

## What changed in the world

Preserved: `large_demo` world name, gravity, physics settings, lighting, spherical coordinates, `person1`, `person1_heat` and its thermal plugin, `demo_animal`, `demo_tree_healthy`, `demo_tree_fire`, and `demo_tree_burnt`.

Replaced: old forest plane, forest walls, oak/pine scenery, platypus, gazebo and decorative rock includes. Scripts explicitly referencing those removed scenery names would need updates; the named dynamic-demo models remain intact.

The existing Blender file `three-sisters-smaller-ground.blend` was updated in place to match the new landscape scale and clearing. These SDF/DAE files are the required simulator export, not another Blender copy. The original Downloads `large_demo.sdf` was only read.

## Geometry, collision and lidar

- About **347,608 visual triangles**, in **135 static links**, including **96 spatial forest sections**. No dynamic links or joints are added to the environment.
- **1,604 trees** have a cylinder trunk collider and two convex canopy-section meshes. Smaller scrub has canopy collisions too. Total: **5,719 collision shapes**, including the terrain and rock meshes.
- Visual forest meshes are batched by 10 m sections. Collision shapes retain each plant's individual position, rotation and scale, leaving paths between trees open.
- Canopy proxies approximate the outer foliage volumes. Small branch gaps are simplified and the visual/collision surfaces are not pixel-identical. Inspect their fit near intended flight routes and set the avoidance planner's clearance for your drone's envelope.
- Existing low-poly terrain and rock surfaces are used for static mesh collisions, including the carved riverbed. Water and thin gravel/shelf overlays are visual only. There is no fluid physics.
- Materials use embedded colours and two small portable PNG textures. These approximate the Blender procedural materials; they are not exact shader bakes.
- Fortress GPU lidar sees visual surfaces. Collision geometry controls physical contact, not the obstacle-avoidance controller. The river is rendered as a simple opaque surface, so its lidar return will not reproduce real water optics.

## Validation status and first run

XML, COLLADA indices, local mesh/texture references, collider scales, retained world entities and launch-clearance geometry were checked locally. **Gazebo Fortress and ROS were not available on the authoring Mac**, so this is not a runtime-certified or performance-benchmarked export.

On your Fortress machine, verify the SDF commands above, check for asset-loading warnings, then use your existing drone launch. Inspect lidar returns from a trunk, canopy, cliff and ground; perform a slow contact test against each with the simulator paused/reset between trials. Check real-time factor with your actual scan resolution and update rate before running the full search. The obstacle count and lidar settings can dominate runtime even with low triangle counts.

Reference: [Fortress sensor setup](https://gazebosim.org/docs/fortress/sensors/), [SDF collision and visual shapes](https://sdformat.org/tutorials/specification/spec_shapes/).
