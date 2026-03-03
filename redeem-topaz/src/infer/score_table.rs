// redeem-topaz/src/infer/score_table.rs

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::infer::{binned_pep, decoy_tail_pvalues, rank_within_key, tdc_qvalues};
use crate::io::osw::FeatureRow;

#[derive(Debug, Clone)]
pub struct ScoreTableRow {
    pub feature_id: u64,
    pub score: f32,
    pub rank: i32,
    pub pvalue: f32,
    pub qvalue: f32,
    pub pep: f32,
}

pub fn build_score_table(
    feature_ids: &[u64],
    scores: &[f32],
    ranks: &[i32],
    pvalues: &[f32],
    qvalues: &[f32],
    peps: &[f32],
) -> Vec<ScoreTableRow> {
    let n = feature_ids.len();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(ScoreTableRow {
            feature_id: feature_ids[i],
            score: scores[i],
            rank: ranks[i],
            pvalue: pvalues[i],
            qvalue: qvalues[i],
            pep: peps[i],
        });
    }
    out
}

/// Build score table rows from feature rows + scores.
pub fn build_score_table_from_rows(
    rows: &[FeatureRow],
    scores: &[f32],
    pep_bins: usize,
) -> Vec<ScoreTableRow> {
    let n = rows.len();
    if n == 0 || scores.len() != n {
        return Vec::new();
    }
    let feature_ids: Vec<u64> = rows.iter().map(|r| r.feature_id).collect();
    let keys: Vec<String> = rows.iter().map(|r| r.group_id.clone()).collect();
    let is_decoy: Vec<bool> = rows.iter().map(|r| r.is_decoy).collect();

    let ranks = rank_within_key(&keys, scores);
    let pvalues = decoy_tail_pvalues(scores, &is_decoy);
    let qvalues = tdc_qvalues(scores, &is_decoy);
    let pep = binned_pep(scores, &is_decoy, pep_bins.max(1));

    build_score_table(&feature_ids, scores, &ranks, &pvalues, &qvalues, &pep)
}

/// Write SCORE table to TSV with header:
/// FEATURE_ID, SCORE, RANK, PVALUE, QVALUE, PEP
pub fn write_score_tsv<P: AsRef<Path>>(path: P, rows: &[ScoreTableRow]) -> std::io::Result<()> {
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);
    writeln!(w, "FEATURE_ID\tSCORE\tRANK\tPVALUE\tQVALUE\tPEP")?;
    for r in rows {
        writeln!(
            w,
            "{}\t{}\t{}\t{}\t{}\t{}",
            r.feature_id, r.score, r.rank, r.pvalue, r.qvalue, r.pep
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn tmp_path(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        p.push(format!("redeem_topaz_{name}_{stamp}.tsv"));
        p
    }

    #[test]
    fn test_score_table_tsv_integration() {
        let rows = vec![
            FeatureRow {
                feature_id: 1,
                precursor_id: 10,
                run_id: 100,
                group_id: "A".to_string(),
                exp_rt: 0.0,
                is_decoy: false,
                features: vec![],
            },
            FeatureRow {
                feature_id: 2,
                precursor_id: 10,
                run_id: 100,
                group_id: "A".to_string(),
                exp_rt: 0.0,
                is_decoy: true,
                features: vec![],
            },
            FeatureRow {
                feature_id: 3,
                precursor_id: 11,
                run_id: 100,
                group_id: "B".to_string(),
                exp_rt: 0.0,
                is_decoy: false,
                features: vec![],
            },
        ];
        let scores = vec![3.0f32, 1.0, 2.0];
        let table = build_score_table_from_rows(&rows, &scores, 5);
        assert_eq!(table.len(), rows.len());

        let path = tmp_path("score_table");
        write_score_tsv(&path, &table).unwrap();
        let text = fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), rows.len() + 1);
        assert_eq!(lines[0], "FEATURE_ID\tSCORE\tRANK\tPVALUE\tQVALUE\tPEP");

        // cleanup
        let _ = fs::remove_file(&path);
    }
}
