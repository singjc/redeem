//! Label-independent forward predictions for caller-supplied unmodified precursors.
//! The frozen v0.52 Validation backends remain unchanged.
use crate::properties::inference::input::PropertyInferenceConfig;
use anyhow::{Context, Result, ensure};
use redeem_properties::foundation::{
    FoundationTrainingRecord, PeptidoformInput, ResearchV0520Forward, ResearchV0520Mobility,
    RetentionTimeLabels, TrainingContext, foundation_fragment_cleavage_geometry,
    foundation_peptidoform_neutral_mass,
};
use redeem_properties::utils::utils::get_device;
use std::fs;
use std::path::Path;

const BATCH: usize = 32;
const MAX_PRECURSORS: usize = 2048;
const PROTON_MASS_DA: f64 = 1.007_276_466_77;
const ORIGINAL_CORPUS_FINGERPRINT: u64 = 0xa2a6_f57d_3106_4ba6;
const ORIGINAL_BENCHMARK_FINGERPRINT: u64 = 0x2133_c039_625f_77df;
const CORE_IONS: [&str; 4] = ["b_z1", "b_z2", "y_z1", "y_z2"];

fn required_column(header: &[&str], name: &str) -> Result<usize> {
    let indices: Vec<_> = header
        .iter()
        .enumerate()
        .filter(|(_, h)| **h == name)
        .collect();
    ensure!(indices.len() == 1, "expected exactly one '{name}' column");
    Ok(indices[0].0)
}

fn optional_column(header: &[&str], name: &str) -> Result<Option<usize>> {
    let indices: Vec<_> = header
        .iter()
        .enumerate()
        .filter(|(_, h)| **h == name)
        .collect();
    ensure!(indices.len() <= 1, "duplicate '{name}' column");
    Ok(indices.first().map(|(index, _)| *index))
}

fn optional_number(cells: &[&str], index: Option<usize>, name: &str) -> Result<Option<f32>> {
    let Some(index) = index else { return Ok(None) };
    let text = cells[index];
    if text.is_empty() {
        return Ok(None);
    }
    let value: f32 = text
        .parse()
        .with_context(|| format!("invalid {name}: {text}"))?;
    ensure!(value.is_finite(), "{name} must be finite");
    Ok(Some(value))
}

fn parse_panel(input: &str) -> Result<Vec<FoundationTrainingRecord>> {
    let mut lines = input.lines();
    let header: Vec<_> = lines
        .next()
        .context("empty precursor panel")?
        .split('\t')
        .collect();
    let seq_col = required_column(&header, "sequence")?;
    let charge_col = required_column(&header, "charge")?;
    let mz_col = optional_column(&header, "precursor_mz")?;
    let nce_col = optional_column(&header, "nce")?;
    ensure!(
        header.len() == 2 + usize::from(mz_col.is_some()) + usize::from(nce_col.is_some()),
        "unsupported panel column; use sequence, charge, optional precursor_mz, nce"
    );
    let mut records = Vec::new();
    for (row, line) in lines.enumerate() {
        ensure!(!line.trim().is_empty(), "blank row at line {}", row + 2);
        ensure!(
            records.len() < MAX_PRECURSORS,
            "panel exceeds {MAX_PRECURSORS} precursors"
        );
        let cells: Vec<_> = line.split('\t').collect();
        ensure!(
            cells.len() == header.len(),
            "line {}: wrong number of columns",
            row + 2
        );
        let sequence = cells[seq_col];
        ensure!(
            (7..=35).contains(&sequence.len())
                && sequence
                    .bytes()
                    .all(|b| b"ACDEFGHIKLMNPQRSTVWY".contains(&b)),
            "line {}: expected unmodified uppercase canonical peptide of length 7..35",
            row + 2
        );
        let charge: i32 = cells[charge_col]
            .parse()
            .with_context(|| format!("line {}: invalid charge", row + 2))?;
        ensure!(
            (1..=8).contains(&charge),
            "line {}: charge must be 1..8",
            row + 2
        );
        let peptidoform = PeptidoformInput::unmodified(sequence);
        let neutral_mass = foundation_peptidoform_neutral_mass(&peptidoform)
            .map_err(|e| anyhow::anyhow!("line {}: {e}", row + 2))?;
        let calculated_mz = (neutral_mass + f64::from(charge) * PROTON_MASS_DA) / f64::from(charge);
        let supplied_mz = optional_number(&cells, mz_col, "precursor_mz")?;
        if let Some(mz) = supplied_mz {
            ensure!(
                mz > 0.0 && (f64::from(mz) - calculated_mz).abs() <= calculated_mz * 25e-6,
                "line {}: precursor_mz disagrees with unmodified sequence/charge by >25 ppm",
                row + 2
            );
        }
        let nce = optional_number(&cells, nce_col, "nce")?;
        if let Some(nce) = nce {
            ensure!(
                nce > 0.0 && nce <= 100.0,
                "line {}: nce must be in (0,100]",
                row + 2
            );
        }
        records.push(FoundationTrainingRecord {
            peptidoform,
            retention_time: RetentionTimeLabels::default(),
            ccs: None,
            fragments: Vec::new(),
            observed_spectrum_peaks: Vec::new(),
            context: TrainingContext {
                charge: Some(charge),
                precursor_mz: Some(supplied_mz.unwrap_or(calculated_mz as f32)),
                nce,
                ..TrainingContext::default()
            },
            run_id: None,
        });
    }
    ensure!(!records.is_empty(), "precursor panel has no rows");
    Ok(records)
}

pub fn run_inference(config: &PropertyInferenceConfig) -> Result<()> {
    ensure!(
        config.batch_size == BATCH,
        "v0520_peptide_panel requires batch_size=32"
    );
    ensure!(
        config.foundation_partition == "external",
        "v0520_peptide_panel requires foundation_partition=external"
    );
    ensure!(
        config.foundation_max_records.is_none() && !config.foundation_spectra_only,
        "v0520_peptide_panel does not support Validation limits or spectra_only"
    );
    let teacher = config
        .foundation_v0350_teacher_checkpoint
        .as_deref()
        .context("v0520_peptide_panel requires foundation_v0350_teacher_checkpoint")?;
    ensure!(
        !teacher.trim().is_empty(),
        "empty v0.35 teacher checkpoint path"
    );
    ensure!(
        !config.model_path.is_empty(),
        "missing v0.52 model checkpoint path"
    );
    let output = Path::new(&config.output_file);
    let fragments_path = format!("{}.predicted_fragments.tsv", config.output_file);
    ensure!(
        !output.exists() && !Path::new(&fragments_path).exists(),
        "prediction output already exists; refusing overwrite"
    );
    let records = parse_panel(
        &fs::read_to_string(&config.inference_data)
            .with_context(|| format!("read precursor panel {}", config.inference_data))?,
    )?;
    let device = get_device(&config.device)?;
    let forward = ResearchV0520Forward::load(Path::new(&config.model_path), device.clone())?;
    let mobility =
        ResearchV0520Mobility::load(Path::new(&config.model_path), Path::new(teacher), device)?;
    for fingerprints in [
        forward.verify_fingerprints(ORIGINAL_CORPUS_FINGERPRINT, ORIGINAL_BENCHMARK_FINGERPRINT),
        mobility.verify_fingerprints(ORIGINAL_CORPUS_FINGERPRINT, ORIGINAL_BENCHMARK_FINGERPRINT),
    ] {
        fingerprints.context("checkpoint does not match frozen v0.52 research provenance")?;
    }

    let mut main = String::from(
        "input_index\tsequence\tcharge\tprecursor_mz\tnce\tpredicted_rt\tpredicted_native_mobility\tpredicted_ccs\tccs_factor\tms2_scope\n",
    );
    let mut fragments = String::from(
        "input_index\tsequence\tcharge\tcleavage_index\tchannel\tion_type\tfragment_number\ttheoretical_mz\tpredicted_intensity\n",
    );
    for (chunk_number, chunk) in records.chunks(BATCH).enumerate() {
        let rt_ms2 = forward.infer(chunk)?;
        let ccs = mobility.infer(chunk)?;
        ensure!(
            rt_ms2.len() == chunk.len() && ccs.len() == chunk.len(),
            "model output count mismatch"
        );
        for (local_index, ((record, property), native)) in
            chunk.iter().zip(rt_ms2).zip(ccs).enumerate()
        {
            let input_index = chunk_number * BATCH + local_index;
            ensure!(
                property.rt.is_finite()
                    && native.ccs.is_finite()
                    && native.native_mobility.is_finite()
                    && native.factor.is_finite()
                    && native.factor > 0.0,
                "nonfinite prediction at input_index={input_index}"
            );
            let sequence = &record.peptidoform.sequence;
            let charge = record.context.charge.unwrap();
            let mz = record.context.precursor_mz.unwrap();
            let nce = record
                .context
                .nce
                .map(|v| format!("{v:.4}"))
                .unwrap_or_default();
            main.push_str(&format!(
                "{input_index}\t{sequence}\t{charge}\t{mz:.8}\t{nce}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\ttheoretical_b_y_1_2_only\n",
                property.rt, native.native_mobility, native.ccs, native.factor
            ));
            let geometry = foundation_fragment_cleavage_geometry(&record.peptidoform)
                .map_err(anyhow::Error::msg)?;
            for cleavage in geometry {
                let i = cleavage.cleavage_index;
                let intensities = property
                    .ms2
                    .get(i)
                    .context("missing predicted cleavage row")?;
                for channel in 0..4 {
                    if charge == 1 && (channel == 1 || channel == 3) {
                        continue;
                    }
                    let intensity = *intensities
                        .get(channel)
                        .context("missing predicted fragment channel")?;
                    ensure!(
                        intensity.is_finite(),
                        "nonfinite MS2 prediction at row {input_index}, cleavage {i}, channel {channel}"
                    );
                    let number = if channel < 2 {
                        i + 1
                    } else {
                        sequence.len() - i - 1
                    };
                    fragments.push_str(&format!(
                        "{input_index}\t{sequence}\t{charge}\t{i}\t{channel}\t{}\t{number}\t{:.8}\t{intensity:.8}\n",
                        CORE_IONS[channel], cleavage.core_mz[channel]
                    ));
                }
            }
        }
    }
    // Finish inference before creating output files; never persist partially inferred panels.
    fs::write(&fragments_path, fragments).with_context(|| format!("write {fragments_path}"))?;
    if let Err(err) = fs::write(output, main) {
        let _ = fs::remove_file(&fragments_path);
        return Err(err).with_context(|| format!("write {}", output.display()));
    }
    println!("foundation_inference=PASS");
    println!("foundation_backend=v0520_peptide_panel");
    println!("input_precursors={}", records.len());
    println!("ms2_scope=theoretical_b_y_1_2_only");
    println!("output={}", output.display());
    println!("predicted_fragments={fragments_path}");
    println!("benchmark_partitions_read=NO");
    println!("historical_test_consumed=NO");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn external_panel_is_label_free_and_fail_closed() {
        let records =
            parse_panel("sequence\tcharge\tnce\nPEPTIDEK\t2\t27\nACDEFGHIK\t3\t\n").unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].context.nce, Some(27.0));
        assert!(records.iter().all(|r| r.ccs.is_none()
            && r.fragments.is_empty()
            && r.retention_time.normalized.is_none()
            && r.retention_time.harmonized.is_none()
            && r.retention_time.observed_seconds.is_none()));
        assert!(parse_panel("sequence\tcharge\nPEPTIDEK\t2\n").is_ok());
        for invalid in [
            "sequence\tcharge\nPEPTIDEK\t0\n",
            "sequence\tcharge\nPEPTXDEK\t2\n",
            "sequence\tcharge\nPEPTIDEK\t2\n\n",
            "sequence\tcharge\tobserved_rt\nPEPTIDEK\t2\t30\n",
            "sequence\tcharge\tprecursor_mz\nPEPTIDEK\t2\t999\n",
            "sequence\tcharge\nPEPTIDEK\t2\textra\n",
            "sequence\tsequence\tcharge\nPEPTIDEK\tPEPTIDEK\t2\n",
        ] {
            assert!(
                parse_panel(invalid).is_err(),
                "accepted invalid panel {invalid:?}"
            );
        }
    }
}
