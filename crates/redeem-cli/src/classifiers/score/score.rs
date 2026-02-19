//! CLI scoring helpers for redeem-classifiers.
use std::path::Path;

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use plotly::layout::BarMode;
use plotly::{Histogram, Layout, Plot};
use report_builder::{Report, ReportSection};

use redeem_classifiers::config::ModelConfig;
use redeem_classifiers::data_handling::{PsmMetadata, RankGrouping};
use redeem_classifiers::io::read_pin_tsv;
use redeem_classifiers::math::Array1;
use redeem_classifiers::psm_scorer::SemiSupervisedLearner;
use redeem_classifiers::stats::tdc;

/// Parameters for running the semi-supervised scorer.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ScoreConfig {
    pub model: ModelConfig,
    pub train_fdr: f32,
    pub xeval_num_iter: usize,
    pub max_iterations: usize,
    pub class_pct: Option<(f64, f64)>,
    pub scale_features: bool,
    pub normalize_scores: bool,
    pub rank_grouping: RankGrouping,
    pub deduplicate: bool,
}

impl Default for ScoreConfig {
    fn default() -> Self {
        Self {
            model: ModelConfig::default(),
            train_fdr: 0.01,
            xeval_num_iter: 5,
            max_iterations: 10,
            class_pct: None,
            scale_features: false,
            normalize_scores: false,
            rank_grouping: RankGrouping::Percolator,
            deduplicate: false,
        }
    }
}

/// Scoring outputs from the semi-supervised learner.
#[derive(Debug)]
pub struct ScoreResult {
    pub predictions: Array1<f32>,
    pub qvalues: Array1<f32>,
    pub ranks: Array1<u32>,
    pub metadata: PsmMetadata,
    pub default_direction: Option<Vec<f32>>,
    pub row_mapping: Option<Vec<Option<usize>>>,
    pub labels: Array1<i32>,
}

/// Load a scorer configuration from a JSON file.
pub fn load_score_config<P: AsRef<Path>>(path: P) -> Result<ScoreConfig> {
    let content = std::fs::read_to_string(&path)
        .with_context(|| format!("Failed to read config: {}", path.as_ref().display()))?;
    let config: ScoreConfig = serde_json::from_str(&content)
        .with_context(|| format!("Failed to parse config: {}", path.as_ref().display()))?;
    Ok(config)
}

/// Run the semi-supervised scorer using a Percolator .pin TSV input.
pub fn score_pin_with_config<P: AsRef<Path>>(pin_path: P, config_path: P) -> Result<ScoreResult> {
    let config = load_score_config(config_path)?;
    score_pin(pin_path, &config)
}

/// Run the semi-supervised scorer using a Percolator .pin TSV input.
pub fn score_pin<P: AsRef<Path>>(pin_path: P, config: &ScoreConfig) -> Result<ScoreResult> {
    let pin_data = read_pin_tsv(pin_path)?;
    let targets = pin_data.y.mapv(|&v| v == 1);
    let labels = pin_data.y.clone();
    let mut learner = SemiSupervisedLearner::new(
        config.model.model_type.clone(),
        config.model.learning_rate,
        config.train_fdr,
        config.xeval_num_iter,
        config.max_iterations,
        config.class_pct,
        config.scale_features,
        config.normalize_scores,
        config.rank_grouping,
    );

    let (predictions, ranks) = learner.fit(pin_data.x, pin_data.y, pin_data.metadata.clone())?;
    let default_direction = learner.feature_weights().map(|weights| weights.to_vec());

    let (predictions, ranks, targets, labels, metadata, row_mapping) = if config.deduplicate {
        let keep = dedup_keep_mask(&pin_data.metadata, &predictions, config.rank_grouping);
        let mapping = build_row_mapping(&keep);
        (
            filter_array1(&predictions, &keep),
            filter_array1(&ranks, &keep),
            filter_array1(&targets, &keep),
            filter_array1(&labels, &keep),
            pin_data.metadata.filter_by_indices(&keep_indices(&keep)),
            Some(mapping),
        )
    } else {
        (
            predictions,
            ranks,
            targets,
            labels,
            pin_data.metadata,
            None,
        )
    };

    let qvalues = tdc(&predictions, &targets, true);

    Ok(ScoreResult {
        predictions,
        qvalues,
        ranks,
        metadata,
        default_direction,
        row_mapping,
        labels,
    })
}

/// Write scoring results to stdout or a tab-delimited file.
pub fn write_score_output<P: AsRef<Path>>(
    pin_path: P,
    results: &ScoreResult,
    output_path: Option<P>,
) -> Result<()> {
    let mut reader = csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .has_headers(true)
        .flexible(true)
        .from_path(&pin_path)
        .with_context(|| format!("Failed to open PIN file: {}", pin_path.as_ref().display()))?;

    let headers = reader
        .headers()
        .context("Failed to read PIN header row")?
        .clone();

    let label_idx = headers
        .iter()
        .position(|header| header.eq_ignore_ascii_case("Label"))
        .ok_or_else(|| anyhow!("Missing label column 'Label' in PIN header"))?;

    let feature_indices: Vec<Option<usize>> = results
        .metadata
        .feature_names
        .iter()
        .map(|name| {
            headers
                .iter()
                .position(|header| header.eq_ignore_ascii_case(name))
        })
        .collect();
    let feature_weights = results.default_direction.as_ref().and_then(|weights| {
        if weights.len() == feature_indices.len() {
            Some(weights)
        } else {
            None
        }
    });

    let writer: Box<dyn std::io::Write> = match output_path {
        Some(path) => {
            let file = std::fs::File::create(&path)
                .with_context(|| format!("Failed to create output: {}", path.as_ref().display()))?;
            Box::new(file)
        }
        None => Box::new(std::io::stdout()),
    };
    let mut writer = csv::WriterBuilder::new().delimiter(b'\t').from_writer(writer);

    let mut out_headers = headers.clone();
    out_headers.push_field("d_score");
    out_headers.push_field("spectrum_q");
    writer.write_record(&out_headers)?;

    let mut row_idx = 0usize;
    let mut pred_idx = 0usize;
    let mut label_row_idx = 0usize;
    for result in reader.records() {
        let record = result.with_context(|| format!("Failed to read row {}", row_idx + 1))?;
        let is_default_direction = record
            .get(0)
            .map(|value| value.eq_ignore_ascii_case("DefaultDirection"))
            .unwrap_or(false);
        let label_value = record.get(label_idx).unwrap_or("").trim();
        let label = label_value.parse::<i32>().ok();
        let mut out_fields = normalize_record_fields(&record, headers.len());
        if is_default_direction {
            if let Some(weights) = feature_weights {
                for (idx_opt, weight) in feature_indices.iter().zip(weights.iter()) {
                    if let Some(idx) = idx_opt {
                        out_fields[*idx] = weight.to_string();
                    }
                }
            }
            out_fields.push(String::new());
            out_fields.push(String::new());
        } else if label.is_some() {
            if let Some(mapping) = results.row_mapping.as_ref() {
                let mapped = mapping.get(label_row_idx).copied().unwrap_or(None);
                label_row_idx += 1;
                let Some(prediction_idx) = mapped else {
                    row_idx += 1;
                    continue;
                };
                out_fields.push(results.predictions[prediction_idx].to_string());
                out_fields.push(results.qvalues[prediction_idx].to_string());
            } else {
                if pred_idx >= results.predictions.len() {
                    return Err(anyhow::anyhow!(
                        "PIN rows exceed predictions length (row {} > {})",
                        row_idx + 1,
                        results.predictions.len()
                    ));
                }
                out_fields.push(results.predictions[pred_idx].to_string());
                out_fields.push(results.qvalues[pred_idx].to_string());
                pred_idx += 1;
                label_row_idx += 1;
            }
        } else {
            out_fields.push(String::new());
            out_fields.push(String::new());
        }
        writer.write_record(&out_fields)?;
        row_idx += 1;
    }

    if results.row_mapping.is_none() && pred_idx != results.predictions.len() {
        return Err(anyhow::anyhow!(
            "Prediction length {} does not match PIN rows {}",
            results.predictions.len(),
            pred_idx
        ));
    }

    writer.flush()?;
    Ok(())
}

fn normalize_record_fields(record: &csv::StringRecord, header_len: usize) -> Vec<String> {
    let mut fields: Vec<String> = record.iter().map(|field| field.to_string()).collect();
    if fields.len() > header_len && header_len > 0 {
        let mut merged = Vec::with_capacity(header_len);
        merged.extend(fields.drain(..header_len.saturating_sub(1)));
        let mut last = String::new();
        if header_len > 0 {
            if let Some(base) = fields.first() {
                last.push_str(base);
            }
            for extra in fields.iter().skip(1) {
                if !extra.is_empty() {
                    if !last.is_empty() {
                        last.push(';');
                    }
                    last.push_str(extra);
                }
            }
            merged.push(last);
        }
        fields = merged;
    }
    while fields.len() < header_len {
        fields.push(String::new());
    }
    fields
}

/// Save an HTML report with a d_score histogram colored by label.
pub fn write_score_report<P: AsRef<Path>>(results: &ScoreResult, report_path: P) -> Result<()> {
    let mut report = Report::new(
        "ReDeeM Classifier Report",
        "1",
        None,
        "Classifier scoring summary",
    );

    let mut section = ReportSection::new("Score Distribution");
    let plot = plot_dscore_histogram(&results.predictions, &results.labels);
    section.add_plot(plot);
    let qvalue_plot = plot_qvalue_histogram(&results.qvalues, &results.labels);
    section.add_plot(qvalue_plot);
    report.add_section(section);

    report.save_to_file(&report_path.as_ref().to_string_lossy().to_string())?;
    Ok(())
}

fn plot_dscore_histogram(scores: &Array1<f32>, labels: &Array1<i32>) -> Plot {
    let mut targets = Vec::new();
    let mut decoys = Vec::new();

    for (score, label) in scores.iter().zip(labels.iter()) {
        if *label == 1 {
            targets.push(*score as f64);
        } else if *label == -1 {
            decoys.push(*score as f64);
        }
    }

    let mut plot = Plot::new();
    plot.add_trace(
        Histogram::new(targets)
            .name("Target")
            .opacity(0.7)
            .marker(plotly::common::Marker::new().color("rgba(31, 119, 180, 0.7)")),
    );
    plot.add_trace(
        Histogram::new(decoys)
            .name("Decoy")
            .opacity(0.7)
            .marker(plotly::common::Marker::new().color("rgba(214, 39, 40, 0.7)")),
    );

    plot.set_layout(
        Layout::new()
            .title("d_score Histogram")
            .x_axis(plotly::layout::Axis::new().title("d_score"))
            .y_axis(plotly::layout::Axis::new().title("Count"))
            .bar_mode(BarMode::Overlay),
    );

    plot
}

fn plot_qvalue_histogram(qvalues: &Array1<f32>, labels: &Array1<i32>) -> Plot {
    let mut targets = Vec::new();
    let mut decoys = Vec::new();

    for (qvalue, label) in qvalues.iter().zip(labels.iter()) {
        if *label == 1 {
            targets.push(*qvalue as f64);
        } else if *label == -1 {
            decoys.push(*qvalue as f64);
        }
    }

    let mut plot = Plot::new();
    plot.add_trace(
        Histogram::new(targets)
            .name("Target")
            .opacity(0.7)
            .marker(plotly::common::Marker::new().color("rgba(31, 119, 180, 0.7)")),
    );
    plot.add_trace(
        Histogram::new(decoys)
            .name("Decoy")
            .opacity(0.7)
            .marker(plotly::common::Marker::new().color("rgba(214, 39, 40, 0.7)")),
    );

    plot.set_layout(
        Layout::new()
            .title("spectrum_q Histogram")
            .x_axis(plotly::layout::Axis::new().title("spectrum_q"))
            .y_axis(plotly::layout::Axis::new().title("Count"))
            .bar_mode(BarMode::Overlay),
    );

    plot
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum DedupScanOrSpec {
    Scan(i32),
    Spec(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct DedupKey {
    file_id: usize,
    scan_or_spec: DedupScanOrSpec,
    exp_mass_bits: Option<u32>,
}

fn dedup_keep_mask(
    metadata: &PsmMetadata,
    scores: &Array1<f32>,
    grouping: RankGrouping,
) -> Vec<bool> {
    let mut best: std::collections::HashMap<DedupKey, (usize, f32)> = std::collections::HashMap::new();
    for idx in 0..scores.len() {
        let scan_value = metadata
            .scan_nr
            .as_ref()
            .and_then(|values| values.get(idx).copied())
            .flatten();
        let exp_mass_bits = metadata
            .exp_mass
            .as_ref()
            .and_then(|values| values.get(idx).copied())
            .flatten()
            .and_then(|value| if value.is_finite() { Some(value.to_bits()) } else { None });
        let scan_or_spec = match grouping {
            RankGrouping::SpecId => DedupScanOrSpec::Spec(metadata.spec_id[idx].clone()),
            RankGrouping::Percolator => match scan_value {
                Some(scan) => DedupScanOrSpec::Scan(scan),
                None => DedupScanOrSpec::Spec(metadata.spec_id[idx].clone()),
            },
        };
        let key = DedupKey {
            file_id: metadata.file_id[idx],
            scan_or_spec,
            exp_mass_bits: if matches!(grouping, RankGrouping::Percolator) {
                exp_mass_bits
            } else {
                None
            },
        };
        let score = scores[idx];
        match best.get_mut(&key) {
            Some((best_idx, best_score)) => {
                if score > *best_score {
                    *best_idx = idx;
                    *best_score = score;
                }
            }
            None => {
                best.insert(key, (idx, score));
            }
        }
    }

    let mut keep = vec![false; scores.len()];
    for (idx, _) in best.values() {
        keep[*idx] = true;
    }
    keep
}

fn build_row_mapping(keep: &[bool]) -> Vec<Option<usize>> {
    let mut mapping = Vec::with_capacity(keep.len());
    let mut next = 0usize;
    for &flag in keep {
        if flag {
            mapping.push(Some(next));
            next += 1;
        } else {
            mapping.push(None);
        }
    }
    mapping
}

fn keep_indices(keep: &[bool]) -> Vec<usize> {
    keep.iter()
        .enumerate()
        .filter_map(|(idx, &flag)| if flag { Some(idx) } else { None })
        .collect()
}

fn filter_array1<T: Copy>(values: &Array1<T>, keep: &[bool]) -> Array1<T> {
    let filtered = values
        .iter()
        .zip(keep.iter())
        .filter_map(|(&value, &flag)| if flag { Some(value) } else { None })
        .collect::<Vec<T>>();
    Array1::from_vec(filtered)
}
