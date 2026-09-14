# dimos-module-pointlio

Point-LIO (non-ROS) source for the DimOS native module, in two implementations:

- `src/`, `include/`, `vendor_pointlio/` — the C++, consumed as a source input by the pointlio flake in the main dimos repo. Not built standalone.
- `rust/` — `pointlio-core`, a Rust port of the same estimator. A plain cargo library with no I/O and no dimos dependency, consumed as a git dependency.

## Upstream

Non-ROS port of [hku-mars/Point-LIO](https://github.com/hku-mars/Point-LIO). The non-ROS harness is derived from the DimOS FAST-LIO2 non-ROS work (dimos-module-fastlio2).

DimOS modifications:
- Parameterized config via CLI arguments (no hardcoded paths)
- Removed Python dependency
- LCM integration for point cloud and odometry publishing
- Livox Mid-360 direct SDK integration (no ROS driver)
- Nix flake for reproducible builds

## rust/

`pointlio-core` is a line-by-line port of the C++ estimator: `laserMapping.hpp`,
`Estimator.cpp`, the two `esekfom::esekf` instantiations, `IMU_Processing.hpp`,
`preprocess.cpp` (Livox path), `ivox3d`, and the `pcl::VoxelGrid` filter. Pure
f64, no threads, no I/O — the same feed sequence gives the same output, on any
machine and under any load.

What it does not port: ROS publishing, PCD saving, the plot and log files, and
the dead ikd-Tree FOV code.

## License

GPL-2.0 — inherited from upstream [Point-LIO](https://github.com/hku-mars/Point-LIO).
Both implementations, `rust/` included: the port is a derivative work of the
same upstream.
