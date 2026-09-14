// Copyright 2026 Dimensional Inc.
//
// This file is part of a Rust port of Point-LIO (https://github.com/hku-mars/Point-LIO)
// and is therefore a derivative work of it.
//
// This program is free software; you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free Software
// Foundation; either version 2 of the License, or (at your option) any later
// version.
//
// This program is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for details.
//
// SPDX-License-Identifier: GPL-2.0-or-later

// pcl::VoxelGrid<PointXYZINormal>::applyFilter (PCL 1.15, downsample_all_data): one centroid per
// occupied leaf, output ordered by leaf index, points summed in spreadsort order.
use crate::common::PointXYZI;

#[derive(Clone, Copy)]
struct IndexIdx {
    idx: u32,
    point: u32,
}

/// `setLeafSize(leaf, leaf, leaf)` + `filter`. `leaf` is the float PCL stores.
pub fn voxel_grid(input: &[PointXYZI], leaf: f32) -> Vec<PointXYZI> {
    if input.is_empty() {
        return Vec::new();
    }
    let inv = 1.0f32 / leaf;
    let mut min_p = [f32::MAX; 3];
    let mut max_p = [f32::MIN; 3];
    for p in input {
        for (k, v) in [p.x, p.y, p.z].into_iter().enumerate() {
            min_p[k] = min_p[k].min(v);
            max_p[k] = max_p[k].max(v);
        }
    }
    let d = |k: usize| ((max_p[k] - min_p[k]) * inv) as i64 + 1;
    if d(0) * d(1) * d(2) > i32::MAX as i64 {
        return input.to_vec();
    }
    let min_b: [i32; 3] = std::array::from_fn(|k| (min_p[k] * inv).floor() as i32);
    let max_b: [i32; 3] = std::array::from_fn(|k| (max_p[k] * inv).floor() as i32);
    let div_b: [i32; 3] = std::array::from_fn(|k| max_b[k] - min_b[k] + 1);
    let divb_mul = [1, div_b[0], div_b[0] * div_b[1]];
    let mut index_vector: Vec<IndexIdx> = input
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let ijk = |v: f32, k: usize| ((v * inv).floor() - min_b[k] as f32) as i32;
            let idx =
                ijk(p.x, 0) * divb_mul[0] + ijk(p.y, 1) * divb_mul[1] + ijk(p.z, 2) * divb_mul[2];
            IndexIdx {
                idx: idx as u32,
                point: i as u32,
            }
        })
        .collect();
    index_vector.sort_unstable_by_key(|t| t.idx);
    let mut out = Vec::new();
    let mut index = 0;
    while index < index_vector.len() {
        let mut i = index + 1;
        while i < index_vector.len() && index_vector[i].idx == index_vector[index].idx {
            i += 1;
        }
        // CentroidPoint accumulators, each field summed in leaf order then divided by the count.
        let mut sum = PointXYZI::default();
        for e in &index_vector[index..i] {
            let p = input[e.point as usize];
            sum.x += p.x;
            sum.y += p.y;
            sum.z += p.z;
            sum.intensity += p.intensity;
            sum.curvature += p.curvature;
        }
        let n = (i - index) as f32;
        out.push(PointXYZI {
            x: sum.x / n,
            y: sum.y / n,
            z: sum.z / n,
            intensity: sum.intensity / n,
            curvature: sum.curvature / n,
        });
        index = i;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pt(x: f32, y: f32, z: f32, i: f32, t: f32) -> PointXYZI {
        PointXYZI {
            x,
            y,
            z,
            intensity: i,
            curvature: t,
        }
    }

    #[test]
    fn centroids_in_leaf_index_order() {
        // Leaf 0.5: two points in (1,0,0)-ish cell, one in the min cell, one far along z.
        let pts = [
            pt(0.6, 0.1, 0.1, 10.0, 1.0),
            pt(0.1, 0.1, 0.1, 20.0, 2.0),
            pt(0.7, 0.2, 0.1, 30.0, 3.0),
            pt(0.1, 0.1, 1.1, 40.0, 4.0),
        ];
        let out = voxel_grid(&pts, 0.5);
        assert_eq!(out.len(), 3);
        assert_eq!(out[0], pts[1]);
        assert_eq!(
            out[1],
            pt((0.6f32 + 0.7) / 2.0, (0.1f32 + 0.2) / 2.0, 0.1, 20.0, 2.0)
        );
        assert_eq!(out[2], pts[3]);
        assert!(voxel_grid(&[], 0.5).is_empty());
        // Big cloud takes the spreadsort path; every leaf must still come out once, sorted.
        let big: Vec<_> = (0..5000)
            .map(|i| {
                let f = i as f32;
                pt((f * 7.3) % 13.0, (f * 3.1) % 11.0, (f * 1.7) % 5.0, 1.0, f)
            })
            .collect();
        let out = voxel_grid(&big, 0.5);
        assert!(out.len() > 100 && out.len() < 5000);
        let key = |p: &PointXYZI| {
            let g = |v: f32| (v * 2.0).floor() as i32;
            (g(p.z), g(p.y), g(p.x))
        };
        assert!(out.windows(2).all(|w| key(&w[0]) < key(&w[1])));
    }
}
