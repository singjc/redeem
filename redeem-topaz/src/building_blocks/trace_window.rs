// redeem-topaz/src/building_blocks/trace_window.rs

/// Return index of nearest element in sorted `rt` to `center_rt`.
pub fn nearest_index_sorted(rt: &[f32], center_rt: f32) -> usize {
    if rt.is_empty() {
        return 0;
    }
    // lower_bound
    let mut lo = 0usize;
    let mut hi = rt.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        if rt[mid] < center_rt {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    let i = lo;
    if i == 0 {
        0
    } else if i >= rt.len() {
        rt.len() - 1
    } else {
        let a = rt[i - 1];
        let b = rt[i];
        if (center_rt - a).abs() <= (b - center_rt).abs() { i - 1 } else { i }
    }
}

/// Copy a centered window of length L from `x` into a zero-padded output.
pub fn pad_or_crop_centered(x: &[f32], center_idx: isize, l: usize) -> Vec<f32> {
    let mut out = vec![0f32; l];
    if x.is_empty() || l == 0 {
        return out;
    }
    let half = (l as isize) / 2;
    let start = center_idx - half;
    let end = start + l as isize;

    let src_start = start.max(0) as usize;
    let src_end = end.min(x.len() as isize).max(0) as usize;

    if src_end <= src_start {
        return out;
    }

    let dst_start = (src_start as isize - start) as usize;
    let n = src_end - src_start;
    out[dst_start..dst_start + n].copy_from_slice(&x[src_start..src_end]);
    out
}

/// Build (Cmax,L) trace tensor for a set of transition series.
///
/// - `series` must be ordered by ordinal then annotation (same as Python).
/// - normalization: "max" divides by global max over (C,L).
pub fn extract_trace_tensor_centered(
    series: &[crate::io::xic::TransitionTrace],
    center_rt: f32,
    l: usize,
    cmax: usize,
    normalize_max: bool,
) -> Vec<f32> {
    let mut out = vec![0f32; cmax * l];
    let take = cmax.min(series.len());
    for c in 0..take {
        let pts = &series[c].points;
        if pts.is_empty() {
            continue;
        }
        // split points into rt/int arrays
        let mut rt: Vec<f32> = Vec::with_capacity(pts.len());
        let mut inten: Vec<f32> = Vec::with_capacity(pts.len());
        for p in pts {
            rt.push(p.rt);
            inten.push(p.intensity);
        }
        let ci = nearest_index_sorted(&rt, center_rt) as isize;
        let win = pad_or_crop_centered(&inten, ci, l);
        out[c * l..(c + 1) * l].copy_from_slice(&win);
    }

    if normalize_max {
        let mut m = 0f32;
        for v in &out {
            if *v > m {
                m = *v;
            }
        }
        if m > 0f32 {
            for v in &mut out {
                *v /= m;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::xic::{TransitionTrace, XicPoint};

    #[test]
    fn test_nearest_index_sorted_basic() {
        let rt = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        assert_eq!(nearest_index_sorted(&rt, 2.1), 2);
        assert_eq!(nearest_index_sorted(&rt, 2.6), 3);
        assert_eq!(nearest_index_sorted(&rt, -1.0), 0);
        assert_eq!(nearest_index_sorted(&rt, 10.0), 4);
        let empty: Vec<f32> = vec![];
        assert_eq!(nearest_index_sorted(&empty, 1.0), 0);
    }

    #[test]
    fn test_pad_or_crop_centered_padding() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let out = pad_or_crop_centered(&x, 2, 3);
        assert_eq!(out, vec![2.0, 3.0, 4.0]);

        let out = pad_or_crop_centered(&x, 0, 4);
        assert_eq!(out, vec![0.0, 0.0, 1.0, 2.0]);

        let x_short = vec![5.0, 6.0];
        let out = pad_or_crop_centered(&x_short, 1, 5);
        assert_eq!(out, vec![0.0, 5.0, 6.0, 0.0, 0.0]);
    }

    #[test]
    fn test_extract_trace_tensor_centered_shape_and_centering() {
        let trace_a = TransitionTrace {
            annotation: "a".to_string(),
            ordinal: 0,
            ms_level: Some(2),
            points: vec![
                XicPoint { rt: 0.0, intensity: 0.0 },
                XicPoint { rt: 1.0, intensity: 1.0 },
                XicPoint { rt: 2.0, intensity: 2.0 },
                XicPoint { rt: 3.0, intensity: 3.0 },
                XicPoint { rt: 4.0, intensity: 4.0 },
            ],
        };
        let trace_b = TransitionTrace {
            annotation: "b".to_string(),
            ordinal: 1,
            ms_level: Some(2),
            points: vec![
                XicPoint { rt: 0.0, intensity: 10.0 },
                XicPoint { rt: 1.0, intensity: 11.0 },
                XicPoint { rt: 2.0, intensity: 12.0 },
                XicPoint { rt: 3.0, intensity: 13.0 },
                XicPoint { rt: 4.0, intensity: 14.0 },
            ],
        };

        let series = vec![trace_a, trace_b];
        let out = extract_trace_tensor_centered(&series, 2.0, 3, 3, false);
        assert_eq!(out.len(), 9);

        let expect0 = vec![1.0, 2.0, 3.0];
        let expect1 = vec![11.0, 12.0, 13.0];
        assert_eq!(&out[0..3], &expect0[..]);
        assert_eq!(&out[3..6], &expect1[..]);
        assert_eq!(&out[6..9], &[0.0, 0.0, 0.0]);
    }
}
