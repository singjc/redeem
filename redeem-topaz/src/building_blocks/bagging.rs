//! Convert flat candidate rows into fixed-size MIL bags.
//!
//! Shape notation used in this module:
//!
//! - `B`: number of bags.
//! - `K`: padded candidate capacity per bag.
//! - `D`: heuristic feature dimension.
//! - `C`: trace-channel count.
//! - `L`: fixed trace length.

use std::collections::HashMap;

/// Output of bagging.
///
/// All buffers are stored in flattened row-major order so they can be passed
/// directly into Candle tensor constructors without an intermediate copy.
#[derive(Debug)]
pub struct Bags {
    /// Flattened `(B, K, D)` feature tensor.
    pub x_bag: Vec<f32>,
    pub b: usize,
    pub k: usize,
    pub d: usize,

    /// Flattened `(B, K, C, L)` trace tensor.
    pub t_bag: Vec<f32>,
    pub c: usize,
    pub l: usize,

    /// Flattened `(B, K)` validity mask.
    pub mask: Vec<bool>,

    /// `(B,)` bag labels in TOPAZ convention: target=1, decoy=0.
    pub y_bag: Vec<f32>,

    /// `(B,)` bag identifiers preserving first-seen group order.
    pub bag_pid: Vec<String>,
}

/// Build fixed-size bags of `K` candidates.
///
/// # Inputs
/// - `x_rows`: flattened `(N, D)` heuristic feature matrix.
/// - `t_rows`: flattened `(N, C, L)` trace tensor.
/// - `y_rows`: row labels using the Python convention `0=target`, `1=decoy`.
/// - `pid_rows`: bagging key for each row, for example `RUN_ID_PRECURSOR_ID`.
/// - `k`: padded candidate capacity per bag.
///
/// # Behavior
/// - group order is stable and follows first occurrence in `pid_rows`
/// - bag labels are derived from the first row in each group
/// - TOPAZ label convention is `target=1`, `decoy=0`
///
/// # Output
/// Returns the bagged tensors and metadata in flattened row-major layout.
pub fn make_bags_with_traces(
    x_rows: &[f32],
    n: usize,
    d: usize,
    t_rows: &[f32],
    c: usize,
    l: usize,
    y_rows: &[u8],
    pid_rows: &[String],
    k: usize,
) -> Bags {
    // preserve encounter order of groups
    let mut pid_to_indices: HashMap<&str, Vec<usize>> = HashMap::new();
    let mut pid_order: Vec<&str> = Vec::new();

    for i in 0..n {
        let pid = pid_rows[i].as_str();
        if !pid_to_indices.contains_key(pid) {
            pid_order.push(pid);
            pid_to_indices.insert(pid, Vec::new());
        }
        pid_to_indices.get_mut(pid).unwrap().push(i);
    }

    let b = pid_order.len();

    let mut x_bag = vec![0f32; b * k * d];
    let mut t_bag = vec![0f32; b * k * c * l];
    let mut mask = vec![false; b * k];
    let mut y_bag = vec![0f32; b];
    let mut bag_pid = Vec::with_capacity(b);

    for (bi, &pid) in pid_order.iter().enumerate() {
        let idxs = &pid_to_indices[pid];
        let take = idxs.len().min(k);
        bag_pid.push(pid.to_string());

        // label from first row (same as Python)
        let yi = y_rows[idxs[0]];
        y_bag[bi] = if yi == 0 { 1.0 } else { 0.0 };

        for kk in 0..take {
            let ri = idxs[kk];

            // X
            let src_x = &x_rows[ri * d..(ri + 1) * d];
            let dst_x0 = (bi * k + kk) * d;
            x_bag[dst_x0..dst_x0 + d].copy_from_slice(src_x);

            // T
            let src_t = &t_rows[ri * c * l..(ri + 1) * c * l];
            let dst_t0 = (bi * k + kk) * c * l;
            t_bag[dst_t0..dst_t0 + c * l].copy_from_slice(src_t);

            // mask
            mask[bi * k + kk] = true;
        }
    }

    Bags {
        x_bag,
        b,
        k,
        d,
        t_bag,
        c,
        l,
        mask,
        y_bag,
        bag_pid,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_make_bags_order_mask_labels() {
        let n = 6usize;
        let d = 2usize;
        let c = 1usize;
        let l = 3usize;
        let k = 2usize;

        // rows: B, A, B, A, B, C
        let x_rows: Vec<f32> = vec![
            1.0, 1.0, // 0 B
            2.0, 2.0, // 1 A
            3.0, 3.0, // 2 B
            4.0, 4.0, // 3 A
            5.0, 5.0, // 4 B
            6.0, 6.0, // 5 C
        ];
        let t_rows: Vec<f32> = vec![
            0.1, 0.2, 0.3, // 0
            1.1, 1.2, 1.3, // 1
            2.1, 2.2, 2.3, // 2
            3.1, 3.2, 3.3, // 3
            4.1, 4.2, 4.3, // 4
            5.1, 5.2, 5.3, // 5
        ];
        let y_rows: Vec<u8> = vec![1, 0, 1, 0, 1, 0]; // B decoy, A target, C target
        let pid_rows: Vec<String> = vec!["B", "A", "B", "A", "B", "C"]
            .into_iter()
            .map(|s| s.to_string())
            .collect();

        let bags = make_bags_with_traces(&x_rows, n, d, &t_rows, c, l, &y_rows, &pid_rows, k);

        assert_eq!(bags.b, 3);
        assert_eq!(bags.k, 2);
        assert_eq!(bags.d, 2);
        assert_eq!(bags.c, 1);
        assert_eq!(bags.l, 3);

        // Stable group order (first occurrence).
        assert_eq!(bags.bag_pid, vec!["B", "A", "C"]);

        // Mask: B has 2, A has 2, C has 1.
        assert_eq!(&bags.mask[0..2], &[true, true]);
        assert_eq!(&bags.mask[2..4], &[true, true]);
        assert_eq!(&bags.mask[4..6], &[true, false]);

        // Label mapping: y==0 -> bag_y=1, y==1 -> bag_y=0.
        assert_eq!(bags.y_bag, vec![0.0, 1.0, 1.0]);

        // Spot-check bag content (C first candidate).
        let c0 = (2 * k + 0) * d;
        assert_eq!(&bags.x_bag[c0..c0 + d], &[6.0, 6.0]);
        let t0 = (2 * k + 0) * c * l;
        assert_eq!(&bags.t_bag[t0..t0 + l], &[5.1, 5.2, 5.3]);
    }
}
