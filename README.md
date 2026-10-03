# dimos-module-pointlio

`pointlio-core`: a Rust port of the [Point-LIO](https://github.com/hku-mars/Point-LIO)
estimator for the DimOS native module. A plain cargo library with no I/O and no
dimos dependency, consumed as a git dependency.

The crate lives in `rust/`.

## Upstream

Non-ROS port of [hku-mars/Point-LIO](https://github.com/hku-mars/Point-LIO).

`pointlio-core` is a line-by-line port of the upstream C++ estimator:
`laserMapping.cpp`, `Estimator.cpp`, the two `esekfom::esekf` instantiations,
`IMU_Processing.hpp`, `preprocess.cpp` (Livox path), `ivox3d`, and the
`pcl::VoxelGrid` filter. Pure f64, no threads, no I/O — the same feed sequence
gives the same output, on any machine and under any load.

What it does not port: ROS publishing, PCD saving, the plot and log files, and
the dead ikd-Tree FOV code.

## Building

```sh
cd rust
cargo build
cargo test
```

A Nix dev shell with the Rust toolchain is provided (`nix develop`, or direnv
via `.envrc`).

## License

GPL-2.0 — inherited from upstream [Point-LIO](https://github.com/hku-mars/Point-LIO).
The port is a derivative work of the same upstream.
