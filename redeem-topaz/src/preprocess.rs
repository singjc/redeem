use serde::{Deserialize, Serialize};

/// Preprocessor: impute non-finite values with column medians, then standardize.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Preprocessor {
    pub med: Vec<f32>,
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
    pub eps: f32,
}

impl Preprocessor {
    pub fn fit(x: &[f32], n: usize, d: usize) -> Self {
        let eps = 1e-8f32;
        let mut med = vec![0f32; d];
        let mut mean = vec![0f32; d];
        let mut std = vec![1f32; d];

        for j in 0..d {
            let mut col: Vec<f32> = Vec::new();
            col.reserve(n);
            for i in 0..n {
                let v = x[i * d + j];
                if v.is_finite() {
                    col.push(v);
                }
            }
            if col.is_empty() {
                med[j] = 0.0;
            } else {
                col.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let m = if col.len() % 2 == 1 {
                    col[col.len() / 2]
                } else {
                    let a = col[col.len() / 2 - 1];
                    let b = col[col.len() / 2];
                    0.5 * (a + b)
                };
                med[j] = if m.is_finite() { m } else { 0.0 };
            }
        }

        for j in 0..d {
            let mut sum = 0f64;
            let mut count = 0f64;
            for i in 0..n {
                let mut v = x[i * d + j];
                if !v.is_finite() {
                    v = med[j];
                }
                if v.is_finite() {
                    sum += v as f64;
                    count += 1.0;
                }
            }
            if count > 0.0 {
                mean[j] = (sum / count) as f32;
            } else {
                mean[j] = 0.0;
            }

            let mut var_sum = 0f64;
            for i in 0..n {
                let mut v = x[i * d + j];
                if !v.is_finite() {
                    v = med[j];
                }
                if v.is_finite() {
                    let diff = v as f64 - mean[j] as f64;
                    var_sum += diff * diff;
                }
            }
            let var = if count > 0.0 { var_sum / count } else { 0.0 };
            let s = (var as f32).sqrt();
            std[j] = if s > eps { s } else { 1.0 };
        }

        Self { med, mean, std, eps }
    }

    pub fn transform_in_place(&self, x: &mut [f32], n: usize, d: usize) {
        if d == 0 || n == 0 {
            return;
        }
        for i in 0..n {
            let row = i * d;
            for j in 0..d {
                let mut v = x[row + j];
                if !v.is_finite() {
                    v = self.med[j];
                }
                v = (v - self.mean[j]) / self.std[j];
                if !v.is_finite() {
                    v = 0.0;
                }
                x[row + j] = v;
            }
        }
    }

    pub fn transform(&self, x: &[f32], n: usize, d: usize) -> Vec<f32> {
        let mut out = x.to_vec();
        self.transform_in_place(&mut out, n, d);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preprocessor_impute_and_scale() {
        let n = 3usize;
        let d = 2usize;
        let x = vec![
            1.0, f32::NAN, //
            3.0, 4.0,     //
            f32::INFINITY, 6.0,
        ];
        let pre = Preprocessor::fit(&x, n, d);
        let y = pre.transform(&x, n, d);
        assert_eq!(y.len(), x.len());
        for v in y {
            assert!(v.is_finite());
        }
    }
}
