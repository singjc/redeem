//! Ranking and statistical post-processing for scored rows.

/// Rank scores within each key, descending (best=1).
pub fn rank_within_key(keys: &[String], scores: &[f32]) -> Vec<i32> {
    let n = scores.len();
    let mut ranks = vec![0i32; n];

    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| {
        let ka = &keys[a];
        let kb = &keys[b];
        if ka == kb {
            scores[b]
                .partial_cmp(&scores[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        } else {
            ka.cmp(kb)
        }
    });

    let mut i = 0usize;
    while i < n {
        let key = &keys[idx[i]];
        let mut j = i;
        while j < n && &keys[idx[j]] == key {
            j += 1;
        }
        for (r, &k) in idx[i..j].iter().enumerate() {
            ranks[k] = (r + 1) as i32;
        }
        i = j;
    }

    ranks
}

/// Simple decoy-tail p-values: `p = (#decoy score >= s) / (#decoy)`.
pub fn decoy_tail_pvalues(scores: &[f32], is_decoy: &[bool]) -> Vec<f32> {
    let n = scores.len();
    let mut decoy_scores: Vec<f32> = scores
        .iter()
        .zip(is_decoy.iter())
        .filter_map(|(&s, &d)| if d { Some(s) } else { None })
        .collect();
    decoy_scores.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    let m = decoy_scores.len();
    if m == 0 {
        return vec![1.0; n];
    }

    let mut out = vec![1.0f32; n];
    for i in 0..n {
        let s = scores[i];
        let cnt = decoy_scores.iter().take_while(|&&d| d >= s).count() as f32;
        out[i] = (cnt / m as f32).max(1.0 / (m as f32 + 1.0));
    }
    out
}

/// Target-decoy competition q-values from scores sorted descending.
pub fn tdc_qvalues(scores: &[f32], is_decoy: &[bool]) -> Vec<f32> {
    let n = scores.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());

    let mut q = vec![1.0f32; n];
    let mut dec = 0f32;
    let mut tar = 0f32;
    let mut fdrs = vec![1.0f32; n];
    for (i, &k) in idx.iter().enumerate() {
        if is_decoy[k] {
            dec += 1.0;
        } else {
            tar += 1.0;
        }
        let fdr = if tar > 0.0 { dec / tar } else { 1.0 };
        fdrs[i] = fdr;
    }
    // monotonic from tail
    let mut min_q = 1.0f32;
    for (i_rev, &k) in idx.iter().enumerate().rev() {
        let fdr = fdrs[i_rev];
        if fdr < min_q {
            min_q = fdr;
        }
        q[k] = min_q;
    }
    q
}

/// Binned PEP based on the decoy fraction in score bins.
pub fn binned_pep(scores: &[f32], is_decoy: &[bool], bins: usize) -> Vec<f32> {
    let n = scores.len();
    if n == 0 || bins == 0 {
        return vec![];
    }
    let min_s = scores.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let span = (max_s - min_s).max(1e-6);

    let mut bin_dec = vec![0f32; bins];
    let mut bin_all = vec![0f32; bins];
    let mut bin_idx = vec![0usize; n];

    for i in 0..n {
        let mut b = (((scores[i] - min_s) / span) * bins as f32) as usize;
        if b >= bins {
            b = bins - 1;
        }
        bin_idx[i] = b;
        bin_all[b] += 1.0;
        if is_decoy[i] {
            bin_dec[b] += 1.0;
        }
    }

    let mut pep = vec![1.0f32; n];
    for i in 0..n {
        let b = bin_idx[i];
        let denom = bin_all[b].max(1.0);
        pep[i] = (bin_dec[b] / denom).min(1.0);
    }
    pep
}

/// Summary of identifications at a chosen target q-value threshold.
#[derive(Debug, Clone)]
pub struct TdcSummary {
    pub cutoff: f32,
    pub n_targets: usize,
    pub n_decoys: usize,
}

/// Summarize identifications at a chosen q-value threshold.
pub fn tdc_summary(scores: &[f32], is_decoy: &[bool], q: f32) -> TdcSummary {
    let qvals = tdc_qvalues(scores, is_decoy);
    let mut cutoff = f32::INFINITY;
    for (s, qv) in scores.iter().zip(qvals.iter()) {
        if *qv <= q && *s < cutoff {
            cutoff = *s;
        }
    }
    if cutoff == f32::INFINITY {
        cutoff = f32::INFINITY;
    }
    let mut n_targets = 0usize;
    let mut n_decoys = 0usize;
    for (s, d) in scores.iter().zip(is_decoy.iter()) {
        if *s > cutoff {
            if *d {
                n_decoys += 1;
            } else {
                n_targets += 1;
            }
        }
    }
    TdcSummary {
        cutoff,
        n_targets,
        n_decoys,
    }
}
