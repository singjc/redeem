//! Sequence construction for XRUN calibration.

use std::collections::HashMap;

/// Dense precursor-by-run tensorization used by the XRUN calibrator.
#[derive(Debug, Clone)]
pub struct XrunSeq {
    /// (P,R,Din)
    pub xseq: Vec<f32>,
    pub p: usize,
    pub r: usize,
    pub din: usize,

    /// (P,R)
    pub mask: Vec<bool>,

    /// (P,)
    pub y_prec: Vec<f32>,

    /// (P,R) mapping back into original bag index (or -1)
    pub idx_mat: Vec<i64>,

    pub prec_ids: Vec<u64>,
}

/// Parse a bag/group identifier of the form `RUN_ID_PRECURSOR_ID`.
pub fn split_group_id_run_prec(group_id: &str) -> Option<(u64, u64)> {
    let mut it = group_id.splitn(2, '_');
    let a = it.next()?;
    let b = it.next()?;
    Some((a.parse().ok()?, b.parse().ok()?))
}

/// Build precursor-aligned run sequences from per-bag base TOPAZ outputs.
pub fn build_xrun_sequences_from_bags(
    bag_pid: &[String],
    bag_score: &[f32],
    bag_hidden: &[f32],
    h: usize,
    bag_y: &[f32],
    max_runs: usize,
    sort_by: &str, // "run" | "score"
) -> XrunSeq {
    let b = bag_pid.len();
    let din = 1 + h;

    let mut run_ids = vec![0u64; b];
    let mut prec_ids = vec![0u64; b];
    for i in 0..b {
        if let Some((r, p)) = split_group_id_run_prec(&bag_pid[i]) {
            run_ids[i] = r;
            prec_ids[i] = p;
        }
    }

    // group bag indices by precursor_id
    let mut prec_to_idx: HashMap<u64, Vec<usize>> = HashMap::new();
    for (i, &p) in prec_ids.iter().enumerate() {
        prec_to_idx.entry(p).or_default().push(i);
    }

    let mut prec_keys: Vec<u64> = prec_to_idx.keys().copied().collect();
    prec_keys.sort_unstable();

    let p = prec_keys.len();
    let r = max_runs;

    let mut xseq = vec![0f32; p * r * din];
    let mut mask = vec![false; p * r];
    let mut y_prec = vec![0f32; p];
    let mut idx_mat = vec![-1i64; p * r];
    let mut prec_out = vec![0u64; p];

    for (pi, &prec) in prec_keys.iter().enumerate() {
        let mut idx = prec_to_idx[&prec].clone();

        // if >R, keep top by score
        if idx.len() > r {
            idx.sort_by(|&a, &b| bag_score[b].partial_cmp(&bag_score[a]).unwrap());
            idx.truncate(r);
        }

        if sort_by == "score" {
            idx.sort_by(|&a, &b| bag_score[b].partial_cmp(&bag_score[a]).unwrap());
        } else {
            idx.sort_by(|&a, &b| run_ids[a].cmp(&run_ids[b]));
        }

        for (ri, &bi) in idx.iter().enumerate() {
            idx_mat[pi * r + ri] = bi as i64;
            mask[pi * r + ri] = true;

            // Xseq[pi,ri,0] = bag_score
            xseq[(pi * r + ri) * din + 0] = bag_score[bi];

            // Xseq[pi,ri,1:] = hidden
            let src = &bag_hidden[bi * h..(bi + 1) * h];
            let dst0 = (pi * r + ri) * din + 1;
            xseq[dst0..dst0 + h].copy_from_slice(src);
        }

        // y_prec = max(bag_y over all runs for that precursor)
        let mut y = 0f32;
        for &bi in prec_to_idx[&prec].iter() {
            y = y.max(bag_y[bi]);
        }
        y_prec[pi] = y;
        prec_out[pi] = prec;
    }

    XrunSeq {
        xseq,
        p,
        r,
        din,
        mask,
        y_prec,
        idx_mat,
        prec_ids: prec_out,
    }
}
