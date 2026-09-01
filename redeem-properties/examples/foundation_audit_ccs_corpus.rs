//! Audit CCS labels derived from ion mobility across a multi-source foundation corpus.
//!
//! The report is intentionally descriptive and performs no training. It summarizes
//! CCS / ion-mobility / precursor-m/z distributions by source and charge, then
//! compares matched peptidoform+charge identities shared between source pairs.

use anyhow::{bail, Context, Result};
use redeem_properties::foundation::{
    load_foundation_corpus, FoundationCorpusConfig, FoundationModificationSite,
    FoundationTrainingRecord,
};
use std::collections::{BTreeMap, HashMap};
use std::env;
use std::fs::File;

#[derive(Debug, Clone, Default)]
struct Values {
    values: Vec<f64>,
}

impl Values {
    fn push(&mut self, value: f64) {
        if value.is_finite() {
            self.values.push(value);
        }
    }

    fn summary(&self) -> Option<Summary> {
        if self.values.is_empty() {
            return None;
        }
        let mut sorted = self.values.clone();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let n = sorted.len();
        let mean = sorted.iter().sum::<f64>() / n as f64;
        let variance = sorted
            .iter()
            .map(|value| {
                let d = *value - mean;
                d * d
            })
            .sum::<f64>()
            / n as f64;
        Some(Summary {
            n,
            min: sorted[0],
            p01: quantile(&sorted, 0.01),
            p05: quantile(&sorted, 0.05),
            p50: quantile(&sorted, 0.50),
            p95: quantile(&sorted, 0.95),
            p99: quantile(&sorted, 0.99),
            max: sorted[n - 1],
            mean,
            std: variance.sqrt(),
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct Summary {
    n: usize,
    min: f64,
    p01: f64,
    p05: f64,
    p50: f64,
    p95: f64,
    p99: f64,
    max: f64,
    mean: f64,
    std: f64,
}

#[derive(Debug, Clone, Default)]
struct SourceStats {
    ccs: Values,
    mobility: Values,
    precursor_mz: Values,
    by_charge: BTreeMap<i32, ChargeStats>,
}

#[derive(Debug, Clone, Default)]
struct ChargeStats {
    ccs: Values,
    mobility: Values,
    precursor_mz: Values,
}

#[derive(Debug, Clone, Default)]
struct PairAccumulator {
    n: usize,
    sum_x: f64,
    sum_y: f64,
    sum_x2: f64,
    sum_y2: f64,
    sum_xy: f64,
    sum_error: f64,
    sum_abs_error: f64,
    sum_sq_error: f64,
    ratios: Values,
}

impl PairAccumulator {
    fn push(&mut self, x: f64, y: f64) {
        if !(x.is_finite() && y.is_finite()) {
            return;
        }
        let error = y - x;
        self.n += 1;
        self.sum_x += x;
        self.sum_y += y;
        self.sum_x2 += x * x;
        self.sum_y2 += y * y;
        self.sum_xy += x * y;
        self.sum_error += error;
        self.sum_abs_error += error.abs();
        self.sum_sq_error += error * error;
        if x != 0.0 {
            self.ratios.push(y / x);
        }
    }

    fn pearson(&self) -> Option<f64> {
        if self.n < 2 {
            return None;
        }
        let n = self.n as f64;
        let cov = self.sum_xy - self.sum_x * self.sum_y / n;
        let vx = self.sum_x2 - self.sum_x * self.sum_x / n;
        let vy = self.sum_y2 - self.sum_y * self.sum_y / n;
        let denom = (vx * vy).sqrt();
        (denom > 0.0).then_some(cov / denom)
    }
}

fn quantile(sorted: &[f64], q: f64) -> f64 {
    if sorted.len() == 1 {
        return sorted[0];
    }
    let position = q.clamp(0.0, 1.0) * (sorted.len() - 1) as f64;
    let lo = position.floor() as usize;
    let hi = position.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let fraction = position - lo as f64;
        sorted[lo] * (1.0 - fraction) + sorted[hi] * fraction
    }
}

fn print_summary(source: &str, charge: Option<i32>, field: &str, values: &Values) {
    let Some(summary) = values.summary() else {
        return;
    };
    let charge = charge.map_or_else(|| "all".to_string(), |value| value.to_string());
    println!(
        "distribution\t{source}\tcharge={charge}\tfield={field}\tn={}\tmin={}\tp01={}\tp05={}\tp50={}\tp95={}\tp99={}\tmax={}\tmean={}\tstd={}",
        summary.n,
        summary.min,
        summary.p01,
        summary.p05,
        summary.p50,
        summary.p95,
        summary.p99,
        summary.max,
        summary.mean,
        summary.std,
    );
}

fn peptidoform_charge_key(record: &FoundationTrainingRecord) -> Option<String> {
    let charge = record.context.charge?;
    let mut modifications = record.peptidoform.modifications.clone();
    modifications.sort_by(|left, right| {
        left.residue_index
            .cmp(&right.residue_index)
            .then_with(|| site_rank(left.site).cmp(&site_rank(right.site)))
            .then_with(|| left.identity_label().cmp(&right.identity_label()))
    });
    let mods = modifications
        .iter()
        .map(|modification| {
            format!(
                "{}:{}:{}",
                site_label(modification.site),
                modification.residue_index,
                modification.identity_label()
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    Some(format!(
        "{}|{}|z{}",
        record.peptidoform.sequence, mods, charge
    ))
}

fn site_rank(site: FoundationModificationSite) -> u8 {
    match site {
        FoundationModificationSite::NTerm => 0,
        FoundationModificationSite::Residue(_) => 1,
        FoundationModificationSite::CTerm => 2,
    }
}

fn site_label(site: FoundationModificationSite) -> String {
    match site {
        FoundationModificationSite::NTerm => "N".to_string(),
        FoundationModificationSite::Residue(index) => format!("R{index}"),
        FoundationModificationSite::CTerm => "C".to_string(),
    }
}

fn main() -> Result<()> {
    let mut args = env::args().skip(1);
    let Some(config_path) = args.next() else {
        bail!("usage: foundation_audit_ccs_corpus <corpus.yaml>");
    };
    if args.next().is_some() {
        bail!("usage: foundation_audit_ccs_corpus <corpus.yaml>");
    }

    let file = File::open(&config_path)
        .with_context(|| format!("failed to open corpus config '{config_path}'"))?;
    let config: FoundationCorpusConfig = serde_yaml::from_reader(file)
        .with_context(|| format!("failed to parse corpus config '{config_path}'"))?;
    let corpus = load_foundation_corpus(&config)?;

    println!("corpus_config\t{config_path}");
    println!(
        "corpus_fingerprint\tfnv1a64:{:016x}",
        corpus.corpus_fingerprint
    );
    println!("records\t{}", corpus.records.len());

    let mut stats = BTreeMap::<String, SourceStats>::new();
    let mut matched = HashMap::<String, BTreeMap<String, Values>>::new();

    for (record, provenance) in corpus.records.iter().zip(&corpus.provenance) {
        let source = stats.entry(provenance.source_id.clone()).or_default();
        if let Some(ccs) = record.ccs {
            source.ccs.push(f64::from(ccs));
        }
        if let Some(mobility) = record.context.ion_mobility {
            source.mobility.push(f64::from(mobility));
        }
        if let Some(mz) = record.context.precursor_mz {
            source.precursor_mz.push(f64::from(mz));
        }
        if let Some(charge) = record.context.charge {
            let charge_stats = source.by_charge.entry(charge).or_default();
            if let Some(ccs) = record.ccs {
                charge_stats.ccs.push(f64::from(ccs));
            }
            if let Some(mobility) = record.context.ion_mobility {
                charge_stats.mobility.push(f64::from(mobility));
            }
            if let Some(mz) = record.context.precursor_mz {
                charge_stats.precursor_mz.push(f64::from(mz));
            }
        }

        if let (Some(key), Some(ccs)) = (peptidoform_charge_key(record), record.ccs) {
            matched
                .entry(key)
                .or_default()
                .entry(provenance.source_id.clone())
                .or_default()
                .push(f64::from(ccs));
        }
    }

    for (source_id, source) in &stats {
        print_summary(source_id, None, "ccs", &source.ccs);
        print_summary(source_id, None, "ion_mobility", &source.mobility);
        print_summary(source_id, None, "precursor_mz", &source.precursor_mz);
        for (charge, charge_stats) in &source.by_charge {
            print_summary(source_id, Some(*charge), "ccs", &charge_stats.ccs);
            print_summary(
                source_id,
                Some(*charge),
                "ion_mobility",
                &charge_stats.mobility,
            );
            print_summary(
                source_id,
                Some(*charge),
                "precursor_mz",
                &charge_stats.precursor_mz,
            );
        }
    }

    let source_ids = stats.keys().cloned().collect::<Vec<_>>();
    for left_index in 0..source_ids.len() {
        for right_index in (left_index + 1)..source_ids.len() {
            let left = &source_ids[left_index];
            let right = &source_ids[right_index];
            let mut pair = PairAccumulator::default();
            for by_source in matched.values() {
                let (Some(left_values), Some(right_values)) =
                    (by_source.get(left), by_source.get(right))
                else {
                    continue;
                };
                let Some(left_summary) = left_values.summary() else {
                    continue;
                };
                let Some(right_summary) = right_values.summary() else {
                    continue;
                };
                pair.push(left_summary.mean, right_summary.mean);
            }
            if pair.n == 0 {
                println!("matched_ccs\t{left}\t{right}\tn=0");
                continue;
            }
            let ratio = pair.ratios.summary();
            println!(
                "matched_ccs\t{left}\t{right}\tn={}\tmean_delta_right_minus_left={}\tmae={}\trmse={}\tpearson={}\tmedian_ratio_right_over_left={}\tp05_ratio={}\tp95_ratio={}",
                pair.n,
                pair.sum_error / pair.n as f64,
                pair.sum_abs_error / pair.n as f64,
                (pair.sum_sq_error / pair.n as f64).sqrt(),
                pair.pearson().map_or_else(|| "NA".to_string(), |value| value.to_string()),
                ratio.map_or_else(|| "NA".to_string(), |summary| summary.p50.to_string()),
                ratio.map_or_else(|| "NA".to_string(), |summary| summary.p05.to_string()),
                ratio.map_or_else(|| "NA".to_string(), |summary| summary.p95.to_string()),
            );
        }
    }

    Ok(())
}
