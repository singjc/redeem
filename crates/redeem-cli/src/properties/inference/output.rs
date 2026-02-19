use anyhow::{Context, Result};
use redeem_properties::utils::data_handling::{PeptideData, TargetNormalization};
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

/// Write a vector of PeptideData to a CSV or TSV file based on file extension.
/// Also include the original (denormalized) target so we can compare predicted vs. true.
pub fn write_peptide_data<P: AsRef<Path>>(
    predicted: &[PeptideData],
    originals: &[PeptideData],
    norm: TargetNormalization,
    normalize_field: &str,
    output_path: P,
) -> Result<()> {
    let path = output_path.as_ref();
    let extension = path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("csv");
    let delimiter = match extension {
        "tsv" => '\t',
        _ => ',',
    };

    let file =
        File::create(path).with_context(|| format!("Failed to create output file: {:?}", path))?;
    let mut writer = csv::WriterBuilder::new()
        .delimiter(delimiter as u8)
        .from_writer(BufWriter::new(file));

    let (target_col, pred_col) = match normalize_field {
        "ccs" => ("target_ccs", "predicted_ccs"),
        _ => ("target_retention_time", "predicted_retention_time"),
    };

    // Write headers (keep retention_time/ccs for compatibility, add explicit target/predicted columns)
    let mut headers = vec![
        "modified_sequence",
        "naked_sequence",
        "mods",
        "mod_sites",
        "charge",
        "precursor_mass",
        "nce",
        "instrument",
        "retention_time",
        "ion_mobility",
        "ccs",
        "ms2_intensities",
    ];
    headers.push(target_col);
    headers.push(pred_col);
    writer.write_record(&headers)?;

    for (predicted_entry, original_entry) in predicted.iter().zip(originals.iter()) {
        let ms2_str = predicted_entry
            .ms2_intensities
            .as_ref()
            .map(|intensities| {
                intensities
                    .iter()
                    .map(|v| {
                        v.iter()
                            .map(|f| f.to_string())
                            .collect::<Vec<_>>()
                            .join(",")
                    })
                    .collect::<Vec<_>>()
                    .join("|")
            })
            .unwrap_or_default();

        // Denormalize the original target so we can persist the true value alongside predictions.
        let denorm = |v: Option<f32>| -> Option<f32> {
            let val = v?;
            match norm {
                TargetNormalization::ZScore(mean, std) => Some(val * std + mean),
                TargetNormalization::MinMax(min, max) => Some(val * (max - min) + min),
                TargetNormalization::None => Some(val),
            }
        };

        let (target_val, predicted_val) = match normalize_field {
            "ccs" => (denorm(original_entry.ccs), predicted_entry.ccs),
            _ => (
                denorm(original_entry.retention_time),
                predicted_entry.retention_time,
            ),
        };

        writer.write_record(&[
            predicted_entry.modified_sequence_str(),
            predicted_entry.naked_sequence_str(),
            predicted_entry.mods_str(),
            predicted_entry.mod_sites_str(),
            &predicted_entry
                .charge
                .map_or(String::new(), |c| c.to_string()),
            &predicted_entry
                .precursor_mass
                .map_or(String::new(), |m| format!("{:.4}", m)),
            &predicted_entry.nce.map_or(String::new(), |n| n.to_string()),
            &predicted_entry
                .instrument_str()
                .unwrap_or_default()
                .to_string(),
            // keep compatibility: populate retention_time/ccs columns with predicted values
            &predicted_entry
                .retention_time
                .map_or(String::new(), |r| format!("{:.4}", r)),
            &predicted_entry
                .ion_mobility
                .map_or(String::new(), |im| format!("{:.4}", im)),
            &predicted_entry
                .ccs
                .map_or(String::new(), |c| format!("{:.4}", c)),
            &ms2_str,
            &target_val.map_or(String::new(), |t| format!("{:.4}", t)),
            &predicted_val.map_or(String::new(), |p| format!("{:.4}", p)),
        ])?;
    }

    writer.flush()?;
    Ok(())
}
