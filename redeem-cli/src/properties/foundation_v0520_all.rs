//! One opt-in forward output from the independently qualified native v0.52 paths.
//! Keeps original checkpoint inference, Validation gating, and annotated MS2 intact.
use crate::properties::inference::input::PropertyInferenceConfig;
use anyhow::{Result, ensure};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;

pub fn run_inference(config: &PropertyInferenceConfig) -> Result<()> {
    ensure!(
        config
            .foundation_v0350_teacher_checkpoint
            .as_deref()
            .is_some_and(|value| !value.trim().is_empty()),
        "v0520_native_all requires foundation_v0350_teacher_checkpoint"
    );
    // Both existing backends enforce the frozen Validation/2048/batch=32 contract.
    ensure!(
        config.foundation_partition == "validation",
        "v0520_native_all is Validation-only"
    );
    ensure!(
        config.foundation_max_records == Some(2048),
        "requires frozen 2048-record Validation prefix"
    );
    ensure!(
        config.batch_size == 32 && !config.foundation_spectra_only,
        "requires batch_size=32 and spectra_only=false"
    );

    let output = &config.output_file;
    let annotated = format!("{output}.annotated_fragments.tsv");
    let mobility = format!("{output}.mobility_ccs.tsv");
    let forward_tmp = format!("{output}.v0520_forward_stage");
    let forward_frag_tmp = format!("{forward_tmp}.annotated_fragments.tsv");
    let mobility_tmp = format!("{output}.v0520_mobility_stage");
    let merged_tmp = format!("{output}.v0520_merged_stage");

    for path in [
        output,
        &annotated,
        &mobility,
        &forward_tmp,
        &forward_frag_tmp,
        &mobility_tmp,
        &merged_tmp,
    ] {
        ensure!(
            !Path::new(path).exists(),
            "output or staging file already exists: {path}"
        );
    }

    let mut forward_config = config.clone();
    forward_config.output_file = forward_tmp.clone();
    let mut mobility_config = config.clone();
    mobility_config.output_file = mobility_tmp.clone();

    // ponytail: two proven forward passes; share loaded tensors only if profiling warrants it.
    let result = (|| -> Result<usize> {
        super::foundation_v0520_native::run_inference(&forward_config)?;
        super::foundation_v0520_ccs::run_inference(&mobility_config)?;
        let (joined, count) = join_ccs(
            &fs::read_to_string(&forward_tmp)?,
            &fs::read_to_string(&mobility_tmp)?,
        )?;
        fs::write(&merged_tmp, joined)?;
        fs::rename(&mobility_tmp, &mobility)?;
        fs::rename(&forward_frag_tmp, &annotated)?;
        fs::rename(&merged_tmp, output)?;
        Ok(count)
    })();

    for path in [&forward_tmp, &forward_frag_tmp, &mobility_tmp, &merged_tmp] {
        if Path::new(path).exists() {
            let _ = fs::remove_file(path);
        }
    }
    let ccs_count = result?;
    println!("foundation_v0520_native_all=PASS");
    println!("ccs_joined_records={ccs_count}");
    println!("output={output}");
    println!("annotated_ms2={annotated}");
    println!("mobility_ccs={mobility}");
    println!("historical_test_consumed=NO");
    println!("train_holdout_read_by_this_inference=NO");
    Ok(())
}

fn column(header: &[&str], name: &str) -> Result<usize> {
    header
        .iter()
        .position(|field| *field == name)
        .ok_or_else(|| anyhow::anyhow!("missing TSV column '{name}'"))
}

/// Join on *corpus* index, never Validation ordinal. Keep existing RT/MS2 text untouched.
fn join_ccs(forward: &str, mobility: &str) -> Result<(String, usize)> {
    let mut ccs_lines = mobility.lines();
    let ccs_header: Vec<_> = ccs_lines.next().unwrap_or("").split('\t').collect();
    let c_index = column(&ccs_header, "corpus_record_index")?;
    let c_seq = column(&ccs_header, "sequence")?;
    let c_charge = column(&ccs_header, "charge")?;
    let c_pred = column(&ccs_header, "predicted_ccs")?;
    let mut ccs = HashMap::new();
    for line in ccs_lines {
        let cells: Vec<_> = line.split('\t').collect();
        ensure!(
            cells.len() == ccs_header.len(),
            "malformed mobility/CCS row"
        );
        ensure!(!cells[c_pred].is_empty(), "empty CCS prediction");
        ensure!(
            ccs.insert(
                cells[c_index],
                (cells[c_seq], cells[c_charge], cells[c_pred])
            )
            .is_none(),
            "duplicate CCS corpus_record_index: {}",
            cells[c_index]
        );
    }
    ensure!(!ccs.is_empty(), "mobility/CCS output has no records");
    let ccs_count = ccs.len();

    let mut forward_lines = forward.lines();
    let forward_header = forward_lines.next().unwrap_or("");
    let header: Vec<_> = forward_header.split('\t').collect();
    let f_index = column(&header, "corpus_record_index")?;
    let f_seq = column(&header, "sequence")?;
    let f_charge = column(&header, "charge")?;
    let f_pred = column(&header, "predicted_ccs")?;
    let mut seen = HashSet::new();
    let mut joined = String::with_capacity(forward.len() + ccs_count * 12);
    joined.push_str(forward_header);
    joined.push('\n');
    for line in forward_lines {
        let cells: Vec<_> = line.split('\t').collect();
        ensure!(cells.len() == header.len(), "malformed native forward row");
        let index = cells[f_index];
        ensure!(
            seen.insert(index),
            "duplicate native corpus_record_index: {index}"
        );
        ensure!(
            cells[f_pred].is_empty(),
            "native forward CCS column unexpectedly populated"
        );
        let (seq, charge, prediction) = ccs.remove(index).ok_or_else(|| {
            anyhow::anyhow!("missing CCS prediction for corpus_record_index: {index}")
        })?;
        ensure!(
            cells[f_seq] == seq && cells[f_charge] == charge,
            "native forward and mobility identity mismatch: {index}"
        );
        for (i, value) in cells.iter().enumerate() {
            if i > 0 {
                joined.push('\t');
            }
            joined.push_str(if i == f_pred { prediction } else { value });
        }
        joined.push('\n');
    }
    ensure!(
        ccs.is_empty(),
        "CCS predictions missing from native RT/MS2 output: {}",
        ccs.len()
    );
    ensure!(!seen.is_empty(), "native RT/MS2 output has no records");
    Ok((joined, ccs_count))
}

#[cfg(test)]
mod tests {
    use super::join_ccs;

    #[test]
    fn joins_by_corpus_identity_and_rejects_mismatch() {
        let forward = "record_index\tcorpus_record_index\tsequence\tcharge\tpredicted_rt\tpredicted_ccs\n0\t13428\tPEPTIDEK\t2\t10.00000000\t\n1\t30699\tOTHERTIDEK\t3\t20.00000000\t\n";
        let ccs = "corpus_record_index\tsequence\tcharge\tpredicted_ccs\n30699\tOTHERTIDEK\t3\t299.12345678\n13428\tPEPTIDEK\t2\t222.12345678\n";
        let (joined, count) = join_ccs(forward, ccs).unwrap();
        assert_eq!(count, 2);
        assert!(joined.contains("0\t13428\tPEPTIDEK\t2\t10.00000000\t222.12345678\n"));
        assert!(joined.contains("1\t30699\tOTHERTIDEK\t3\t20.00000000\t299.12345678\n"));
        assert!(join_ccs(forward, &ccs.replace("PEPTIDEK", "WRONGSEQ")).is_err());
        assert!(join_ccs(forward, &ccs.replace("13428", "99999")).is_err());
        assert!(join_ccs(forward, &ccs.lines().take(2).collect::<Vec<_>>().join("\n")).is_err());
        assert!(
            join_ccs(
                forward,
                &format!("{ccs}30699\tOTHERTIDEK\t3\t299.12345678\n")
            )
            .is_err()
        );
        assert!(join_ccs(forward, &ccs.replace("299.12345678", "")).is_err());
    }
}
