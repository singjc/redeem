//! Pure helpers for extracting fixed-width trace windows around an expected RT.
//!
//! Shape notation used in this module:
//!
//! - `L`: requested fixed output length in retention-time samples.
//! - `Cmax`: maximum number of channels/traces kept for one precursor.
//!
//! A returned trace tensor with shape `(Cmax, L)` is stored as a flat
//! row-major vector where each channel contributes one contiguous window of
//! length `L`.

/// Return the index of the RT sample nearest to `center_rt`.
///
/// `rt` must be sorted in ascending order.
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
        if (center_rt - a).abs() <= (b - center_rt).abs() {
            i - 1
        } else {
            i
        }
    }
}

/// Copy a centered window of length `L` from `x` into a zero-padded output.
///
/// The returned vector always has length `L`. If the requested window would
/// extend past either end of `x`, the missing values are filled with zeros.
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

/// Build a fixed-size `(Cmax, L)` trace tensor for a set of traces.
///
/// # Inputs
/// - `series`: traces already ordered deterministically, typically by ordinal
///   then annotation to match the Python implementation.
/// - `center_rt`: expected apex RT around which the window is centered.
/// - `l`: output trace length `L`.
/// - `cmax`: number of channels kept in the output. Missing channels are
///   zero-padded.
/// - `normalize_max`: when `true`, divide the full `(Cmax, L)` block by its
///   global maximum.
///
/// # Output
/// Returns a row-major flat representation of a `(Cmax, L)` tensor.
pub fn extract_trace_tensor_centered(
    series: &[crate::io::xic::TransitionTrace],
    center_rt: f32,
    l: usize,
    cmax: usize,
    normalize_max: bool,
) -> Vec<f32> {
    extract_trace_tensor_centered_with_bounds(series, center_rt, None, l, cmax, normalize_max)
}

/// Build a fixed-size `(Cmax, L)` trace tensor, optionally masking samples
/// outside a valid RT interval.
pub fn extract_trace_tensor_centered_with_bounds(
    series: &[crate::io::xic::TransitionTrace],
    center_rt: f32,
    rt_bounds: Option<(f32, f32)>,
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
        if let Some((left, right)) = rt_bounds {
            let half = (l as isize) / 2;
            let start = ci - half;
            for dst_idx in 0..l {
                let src_idx = start + dst_idx as isize;
                if src_idx < 0 || src_idx >= inten.len() as isize {
                    continue;
                }
                let src_idx = src_idx as usize;
                let keep = {
                    let rt_value = rt[src_idx];
                    rt_value >= left && rt_value <= right
                };
                if keep {
                    out[c * l + dst_idx] = inten[src_idx];
                }
            }
        } else {
            let win = pad_or_crop_centered(&inten, ci, l);
            out[c * l..(c + 1) * l].copy_from_slice(&win);
        }
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
                XicPoint {
                    rt: 0.0,
                    intensity: 0.0,
                },
                XicPoint {
                    rt: 1.0,
                    intensity: 1.0,
                },
                XicPoint {
                    rt: 2.0,
                    intensity: 2.0,
                },
                XicPoint {
                    rt: 3.0,
                    intensity: 3.0,
                },
                XicPoint {
                    rt: 4.0,
                    intensity: 4.0,
                },
            ],
        };
        let trace_b = TransitionTrace {
            annotation: "b".to_string(),
            ordinal: 1,
            ms_level: Some(2),
            points: vec![
                XicPoint {
                    rt: 0.0,
                    intensity: 10.0,
                },
                XicPoint {
                    rt: 1.0,
                    intensity: 11.0,
                },
                XicPoint {
                    rt: 2.0,
                    intensity: 12.0,
                },
                XicPoint {
                    rt: 3.0,
                    intensity: 13.0,
                },
                XicPoint {
                    rt: 4.0,
                    intensity: 14.0,
                },
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
