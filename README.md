# dimos-module-pointlio

Point-LIO (non-ROS) source for the DimOS native module. This repo is consumed as a source input by the pointlio flake in the main dimos repo — it is not built standalone.

## Upstream

Non-ROS port of [hku-mars/Point-LIO](https://github.com/hku-mars/Point-LIO). The non-ROS harness is derived from the DimOS FAST-LIO2 non-ROS work (dimos-module-fastlio2).

DimOS modifications:
- Parameterized config via CLI arguments (no hardcoded paths)
- Removed Python dependency
- LCM integration for point cloud and odometry publishing
- Livox Mid-360 direct SDK integration (no ROS driver)
- Nix flake for reproducible builds

## License

GPL-2.0 — inherited from upstream [Point-LIO](https://github.com/hku-mars/Point-LIO).
