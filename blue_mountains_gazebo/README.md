# Blue Mountains — trimmed Three Sisters map

Updated environment package for **Gazebo Fortress 6.x**, using SDF 1.8 and COLLADA meshes. Ready to commit to GitHub. Replace the old package folder in full so obsolete meshes are removed.

## Included

- `models/blue_mountains/model.sdf`: static scenery, visual meshes and collision shapes.
- `models/blue_mountains/model.config`: Gazebo model metadata.
- `models/blue_mountains/meshes/`: 71 COLLADA files, including reusable canopy collision meshes.
- `models/blue_mountains/materials/textures/`: portable sandstone and forest-floor textures.
- `worlds/large_demo.sdf`: previous demo world with the new scenery included.
- `setup_env.sh`: resource-path setup for your existing launch.
- `validate_package.py`, `validation.json`, `manifest.json`: checks and export statistics.

## Launch on the ROS/Gazebo machine

```bash
source /path/to/blue_mountains_gazebo/setup_env.sh
ign sdf -k "$BLUE_MOUNTAINS_WORLD"
ign sdf -k "$BLUE_MOUNTAINS_ROOT/models/blue_mountains/model.sdf"
ign gazebo -v 4 "$BLUE_MOUNTAINS_WORLD"
```

For the existing drone simulation, source the setup script in its launch terminal and set its world-file argument to `$BLUE_MOUNTAINS_WORLD`. Keep your existing ROS 2 controllers, drone, lidar, bridges and server/sensor plugin configuration. This package does not include them. The world keeps its previous world-level plugin setup (no explicit server plugins).

The Standing person still uses the previous OpenRobotics Fuel URL and must be cached or downloadable. All landscape resources are local and use portable model-relative paths.

## Changes

- Ground reduced to approximately **70 × 45.27 m**: X −18.77 to 51.2333; Y −8.7667 to 36.5.
- Rear landscape and equal strips from both sides removed; Three Sisters centered across the width.
- Separate low outcrop moved 6 m right, along with its vegetation.
- Buried portions of the main rock formations cut to the terrain surface.
- River and bank faces now face upward and sit above the terrain. Water has explicit opaque blue-green SDF colour as well as embedded COLLADA colour.
- **148,598 visual triangles**, **53 static links**, **39 forest batches**, **465 tree trunks**, **2,092 collision shapes**. Previous export: 347,608 triangles and 5,719 collision shapes. These reductions are not measured frame-rate improvements.

## Collision behaviour

Terrain and rocks use their current static triangle meshes for contact. Tree trunks use cylinders; foliage and shrubs use two convex canopy sections per plant, following the previous package. Forest visuals are batched in 10 m tiles; individual collision placement preserves gaps between plants. Proxies simplify branch detail, so allow flight clearance around foliage.

Water, gravel banks and thin decorative shelves are visual only. The underlying terrain supplies riverbed collision; there is no water or buoyancy physics. Fortress GPU lidar primarily uses visual geometry, while collisions control physical contact.

## Demo compatibility

World name, lighting, physics, person and thermal target, demo tree poses and demo animal XY are retained. **The demo animal Z is now 3.82 m**, because the repositioned outcrop covers its former ground-level pose. Launch `(0,0)` and the person position remain on the existing flat clearing. Do not use the old landscape bounds for waypoints: the rear and side terrain no longer exist.

## Validation

```bash
python3 /path/to/blue_mountains_gazebo/validate_package.py
```

Local checks cover XML structure, COLLADA arrays and indices, material and texture references, positive collider scales, unique names, river face direction and colour, and model counts. Source-scene ray checks verified the launch/person clearance and identified the demo animal adjustment.

**Gazebo/ROS are not installed on the authoring Mac. Runtime loading, contact behaviour, lidar and performance must still be checked on your Fortress machine.** Run the `ign sdf -k` checks above, inspect the river colour, and test contact with ground, rock, trunk and canopy before a full flight.

References: [Fortress resource lookup](https://gazebosim.org/api/gazebo/6/resources.html), [SDF collisions and visuals](https://sdformat.org/tutorials/specification/spec_shapes/).
