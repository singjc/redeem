//! Train the frozen-backbone v0.14.0 spectrum-candidate setwise reranker.
//!
//! This example intentionally trains only a compact MLP ranking head over candidate-level
//! evidence exported by `foundation_generate_unified`. The peptide foundation model and the
//! accepted v0.13.23 proposal architecture remain frozen. Candidate identity is used only as
//! the post-generation supervised label. TEST input is forbidden by policy.

use anyhow::{Context, Result};
use serde::Serialize;
use std::collections::HashMap;
use std::env;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

const VERSION: &str = "v0.14.0";
const HIDDEN: usize = 16;
const EPOCHS: usize = 10;
const BATCH_GROUPS: usize = 16;
const LEARNING_RATE: f64 = 1.0e-3;
const WEIGHT_DECAY: f64 = 1.0e-4;
const ADAM_BETA1: f64 = 0.9;
const ADAM_BETA2: f64 = 0.999;
const ADAM_EPS: f64 = 1.0e-8;
const SEED: u64 = 20_260_913;

const FEATURE_NAMES: [&str; 12] = [
    "fragment_score",
    "matched_cleavages",
    "ar_total_log_probability",
    "ar_mean_log_probability",
    "neural_all_mask_log_probability",
    "neural_length_log_probability",
    "abs_mass_error_da",
    "candidate_length_error",
    "from_diffusion",
    "from_causal_beam",
    "from_reverse_causal_beam",
    "from_bidirectional_mitm",
];

#[derive(Debug, Clone)]
struct CandidateRow {
    raw_features: [f64; FEATURE_NAMES.len()],
    exact: bool,
    il_exact: bool,
    legacy_top1: bool,
}

#[derive(Debug, Clone)]
struct CandidateGroup {
    record_index: usize,
    rows: Vec<CandidateRow>,
}

#[derive(Debug, Clone, Serialize)]
struct Normalization {
    feature_names: Vec<String>,
    mean: Vec<f64>,
    std: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct RerankerCheckpoint {
    version: String,
    objective: String,
    architecture: String,
    proposal_policy: String,
    training_label: String,
    test_partition_consumed: bool,
    seed: u64,
    hidden_dim: usize,
    epochs: usize,
    batch_groups: usize,
    learning_rate: f64,
    weight_decay: f64,
    train_candidate_tsv: String,
    validation_candidate_tsv: String,
    train_groups: usize,
    train_supervised_groups: usize,
    validation_groups: usize,
    normalization: Normalization,
    w1: Vec<f64>,
    b1: Vec<f64>,
    w2: Vec<f64>,
    b2: f64,
}

#[derive(Debug, Clone)]
struct Model {
    w1: Vec<f64>,
    b1: Vec<f64>,
    w2: Vec<f64>,
    b2: f64,
}

impl Model {
    fn new(seed: u64) -> Self {
        let input = FEATURE_NAMES.len();
        let mut rng = Rng::new(seed);
        let scale1 = (6.0 / (input + HIDDEN) as f64).sqrt();
        let scale2 = (6.0 / (HIDDEN + 1) as f64).sqrt();
        let w1 = (0..input * HIDDEN)
            .map(|_| (rng.next_f64() * 2.0 - 1.0) * scale1)
            .collect();
        let w2 = (0..HIDDEN)
            .map(|_| (rng.next_f64() * 2.0 - 1.0) * scale2)
            .collect();
        Self {
            w1,
            b1: vec![0.0; HIDDEN],
            w2,
            b2: 0.0,
        }
    }

    fn score(&self, x: &[f64; FEATURE_NAMES.len()]) -> (f64, [f64; HIDDEN]) {
        let mut hidden = [0.0; HIDDEN];
        for (h, value) in hidden.iter_mut().enumerate() {
            let mut z = self.b1[h];
            for (j, &xj) in x.iter().enumerate() {
                z += self.w1[h * FEATURE_NAMES.len() + j] * xj;
            }
            *value = z.tanh();
        }
        let mut score = self.b2;
        for h in 0..HIDDEN {
            score += self.w2[h] * hidden[h];
        }
        (score, hidden)
    }
}

#[derive(Debug, Clone)]
struct Gradients {
    w1: Vec<f64>,
    b1: Vec<f64>,
    w2: Vec<f64>,
    b2: f64,
}

impl Gradients {
    fn zeros() -> Self {
        Self {
            w1: vec![0.0; FEATURE_NAMES.len() * HIDDEN],
            b1: vec![0.0; HIDDEN],
            w2: vec![0.0; HIDDEN],
            b2: 0.0,
        }
    }

    fn clear(&mut self) {
        self.w1.fill(0.0);
        self.b1.fill(0.0);
        self.w2.fill(0.0);
        self.b2 = 0.0;
    }

    fn scale(&mut self, factor: f64) {
        for v in &mut self.w1 {
            *v *= factor;
        }
        for v in &mut self.b1 {
            *v *= factor;
        }
        for v in &mut self.w2 {
            *v *= factor;
        }
        self.b2 *= factor;
    }
}

#[derive(Debug, Clone)]
struct AdamState {
    mw1: Vec<f64>,
    vw1: Vec<f64>,
    mb1: Vec<f64>,
    vb1: Vec<f64>,
    mw2: Vec<f64>,
    vw2: Vec<f64>,
    mb2: f64,
    vb2: f64,
    step: usize,
}

impl AdamState {
    fn new() -> Self {
        Self {
            mw1: vec![0.0; FEATURE_NAMES.len() * HIDDEN],
            vw1: vec![0.0; FEATURE_NAMES.len() * HIDDEN],
            mb1: vec![0.0; HIDDEN],
            vb1: vec![0.0; HIDDEN],
            mw2: vec![0.0; HIDDEN],
            vw2: vec![0.0; HIDDEN],
            mb2: 0.0,
            vb2: 0.0,
            step: 0,
        }
    }

    fn update(&mut self, model: &mut Model, grad: &Gradients) {
        self.step += 1;
        let t = self.step as i32;
        for i in 0..model.w1.len() {
            adam_scalar(
                &mut model.w1[i],
                grad.w1[i],
                &mut self.mw1[i],
                &mut self.vw1[i],
                t,
                true,
            );
        }
        for i in 0..model.b1.len() {
            adam_scalar(
                &mut model.b1[i],
                grad.b1[i],
                &mut self.mb1[i],
                &mut self.vb1[i],
                t,
                false,
            );
        }
        for i in 0..model.w2.len() {
            adam_scalar(
                &mut model.w2[i],
                grad.w2[i],
                &mut self.mw2[i],
                &mut self.vw2[i],
                t,
                true,
            );
        }
        adam_scalar(
            &mut model.b2,
            grad.b2,
            &mut self.mb2,
            &mut self.vb2,
            t,
            false,
        );
    }
}

fn adam_scalar(param: &mut f64, grad: f64, m: &mut f64, v: &mut f64, t: i32, decay: bool) {
    *m = ADAM_BETA1 * *m + (1.0 - ADAM_BETA1) * grad;
    *v = ADAM_BETA2 * *v + (1.0 - ADAM_BETA2) * grad * grad;
    let mhat = *m / (1.0 - ADAM_BETA1.powi(t));
    let vhat = *v / (1.0 - ADAM_BETA2.powi(t));
    let update = mhat / (vhat.sqrt() + ADAM_EPS) + if decay { WEIGHT_DECAY * *param } else { 0.0 };
    *param -= LEARNING_RATE * update;
}

#[derive(Debug, Clone, Copy)]
struct EvalMetrics {
    records: usize,
    oracle_exact: usize,
    oracle_il: usize,
    top1_exact: usize,
    top1_il: usize,
    legacy_top1_exact: usize,
    legacy_top1_il: usize,
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 4 {
        anyhow::bail!("usage: foundation_train_setwise_reranker TRAIN_CANDIDATES.tsv VALIDATION_CANDIDATES.tsv OUTPUT_DIR");
    }
    let train_path = PathBuf::from(&args[1]);
    let validation_path = PathBuf::from(&args[2]);
    let output_dir = PathBuf::from(&args[3]);

    reject_test_path(&train_path)?;
    reject_test_path(&validation_path)?;

    let train_groups = read_candidate_groups(&train_path)
        .with_context(|| format!("read train candidate TSV {train_path:?}"))?;
    let validation_groups = read_candidate_groups(&validation_path)
        .with_context(|| format!("read validation candidate TSV {validation_path:?}"))?;
    if train_groups.is_empty() || validation_groups.is_empty() {
        anyhow::bail!("train and validation candidate TSVs must both contain candidate groups");
    }

    let supervised_train: Vec<&CandidateGroup> = train_groups
        .iter()
        .filter(|group| group.rows.iter().any(|row| row.exact))
        .collect();
    if supervised_train.is_empty() {
        anyhow::bail!(
            "no TRAIN candidate group contains an exact target; cannot train listwise objective"
        );
    }

    let normalization = fit_normalization(&supervised_train);
    let mut model = Model::new(SEED);
    let mut adam = AdamState::new();
    let mut grad = Gradients::zeros();

    println!("setwise_reranker_version\t{VERSION}");
    println!("objective\tlistwise_softmax_cross_entropy_exact_target");
    println!("architecture\tfrozen_backbone_candidate_features_mlp_12x16x1");
    println!("proposal_policy\tv01323_final_two_view_fixed_budget_frozen");
    println!("test_partition_consumed\tNO");
    println!("train_groups\t{}", train_groups.len());
    println!("train_supervised_groups\t{}", supervised_train.len());
    println!("validation_groups\t{}", validation_groups.len());
    println!("epochs\t{EPOCHS}");
    println!("batch_groups\t{BATCH_GROUPS}");
    println!("learning_rate\t{LEARNING_RATE}");
    println!("weight_decay\t{WEIGHT_DECAY}");
    println!("seed\t{SEED}");

    let mut order: Vec<usize> = (0..supervised_train.len()).collect();
    for epoch in 0..EPOCHS {
        deterministic_shuffle(&mut order, SEED ^ (epoch as u64 + 1));
        let mut epoch_loss = 0.0;
        let mut epoch_groups = 0usize;
        let mut batch_count = 0usize;
        grad.clear();
        for &index in &order {
            let group = supervised_train[index];
            let loss = accumulate_group_gradient(group, &normalization, &model, &mut grad)?;
            epoch_loss += loss;
            epoch_groups += 1;
            batch_count += 1;
            if batch_count == BATCH_GROUPS {
                grad.scale(1.0 / batch_count as f64);
                adam.update(&mut model, &grad);
                grad.clear();
                batch_count = 0;
            }
        }
        if batch_count > 0 {
            grad.scale(1.0 / batch_count as f64);
            adam.update(&mut model, &grad);
        }
        println!(
            "training_epoch\tepoch={}\tmean_listwise_loss={:.8}\tsupervised_groups={}",
            epoch + 1,
            epoch_loss / epoch_groups as f64,
            epoch_groups
        );
    }

    let metrics = evaluate(&validation_groups, &normalization, &model);
    let literal_gate = metrics.top1_exact >= 28;
    let il_gate = metrics.top1_il >= 42;
    let oracle_gate = metrics.oracle_exact >= 44 && metrics.oracle_il >= 54;
    let gate = literal_gate && il_gate && oracle_gate;
    println!("validation_summary\trecords={}\toracle_literal={}\toracle_il={}\tlegacy_top1_literal={}\tlegacy_top1_il={}\tsetwise_top1_literal={}\tsetwise_top1_il={}", metrics.records, metrics.oracle_exact, metrics.oracle_il, metrics.legacy_top1_exact, metrics.legacy_top1_il, metrics.top1_exact, metrics.top1_il);
    println!("v0140_acceptance_gate\trequired_literal=28\trequired_il=42\trequired_oracle_literal=44\trequired_oracle_il=54\tobserved_literal={}\tobserved_il={}\tobserved_oracle_literal={}\tobserved_oracle_il={}\tgate={}", metrics.top1_exact, metrics.top1_il, metrics.oracle_exact, metrics.oracle_il, if gate { "PASS" } else { "FAIL" });

    fs::create_dir_all(&output_dir)?;
    let checkpoint = RerankerCheckpoint {
        version: VERSION.to_string(),
        objective: "listwise_softmax_cross_entropy_exact_target".to_string(),
        architecture: "frozen_backbone_candidate_features_mlp_12x16x1".to_string(),
        proposal_policy: "v01323_final_two_view_fixed_budget_frozen".to_string(),
        training_label: "post_generation_peptidoform_exact".to_string(),
        test_partition_consumed: false,
        seed: SEED,
        hidden_dim: HIDDEN,
        epochs: EPOCHS,
        batch_groups: BATCH_GROUPS,
        learning_rate: LEARNING_RATE,
        weight_decay: WEIGHT_DECAY,
        train_candidate_tsv: train_path.display().to_string(),
        validation_candidate_tsv: validation_path.display().to_string(),
        train_groups: train_groups.len(),
        train_supervised_groups: supervised_train.len(),
        validation_groups: validation_groups.len(),
        normalization,
        w1: model.w1,
        b1: model.b1,
        w2: model.w2,
        b2: model.b2,
    };
    let checkpoint_path = output_dir.join("reranker.yaml");
    serde_yaml::to_writer(BufWriter::new(File::create(&checkpoint_path)?), &checkpoint)?;
    let mut summary = BufWriter::new(File::create(output_dir.join("validation_summary.tsv"))?);
    writeln!(summary, "records\toracle_literal\toracle_il\tlegacy_top1_literal\tlegacy_top1_il\tsetwise_top1_literal\tsetwise_top1_il\tgate")?;
    writeln!(
        summary,
        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
        metrics.records,
        metrics.oracle_exact,
        metrics.oracle_il,
        metrics.legacy_top1_exact,
        metrics.legacy_top1_il,
        metrics.top1_exact,
        metrics.top1_il,
        if gate { "PASS" } else { "FAIL" }
    )?;
    println!("checkpoint\t{}", checkpoint_path.display());
    Ok(())
}

fn reject_test_path(path: &Path) -> Result<()> {
    let name = path.to_string_lossy().to_ascii_lowercase();
    let suspicious = name.contains("/test/")
        || name.contains("\\test\\")
        || name.contains("test_partition")
        || name.ends_with("/test.tsv")
        || name.ends_with("\\test.tsv");
    if suspicious {
        anyhow::bail!("v0.14.0 forbids TEST-partition inputs; suspicious path {path:?}");
    }
    Ok(())
}

fn read_candidate_groups(path: &Path) -> Result<Vec<CandidateGroup>> {
    let file = BufReader::new(File::open(path)?);
    let mut lines = file.lines();
    let header = lines.next().context("candidate TSV is empty")??;
    let columns: Vec<&str> = header.split('\t').collect();
    let mut index = HashMap::new();
    for (i, name) in columns.iter().enumerate() {
        index.insert(*name, i);
    }
    for required in [
        "record_index",
        "predicted_active_tokens",
        "candidate_sequence",
        "fragment_score",
        "matched_cleavages",
        "neural_all_mask_log_probability",
        "neural_length_log_probability",
        "ar_total_log_probability",
        "ar_mean_log_probability",
        "mass_error_da",
        "mass_valid",
        "from_diffusion",
        "from_causal_beam",
        "from_reverse_causal_beam",
        "from_bidirectional_mitm",
        "peptidoform_exact",
        "il_sequence_exact",
        "fragment_causal_mass_rank",
    ] {
        if !index.contains_key(required) {
            anyhow::bail!("candidate TSV missing required column '{required}'");
        }
    }

    let mut groups = Vec::new();
    let mut current_id: Option<usize> = None;
    let mut current_rows = Vec::new();
    for line in lines {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        let get = |name: &str| -> Result<&str> {
            let i = *index.get(name).context("internal missing TSV index")?;
            fields
                .get(i)
                .copied()
                .with_context(|| format!("row missing column {name}"))
        };
        if !parse_bool(get("mass_valid")?) {
            continue;
        }
        let record_index: usize = get("record_index")?.parse()?;
        if current_id != Some(record_index) {
            if let Some(id) = current_id.take() {
                groups.push(CandidateGroup {
                    record_index: id,
                    rows: std::mem::take(&mut current_rows),
                });
            }
            current_id = Some(record_index);
        }
        let predicted_len: f64 = parse_finite(get("predicted_active_tokens")?, 0.0);
        let candidate_len = get("candidate_sequence")?
            .chars()
            .filter(|c| c.is_ascii_alphabetic())
            .count() as f64;
        let mass_error = parse_finite(get("mass_error_da")?, 0.0).abs();
        current_rows.push(CandidateRow {
            raw_features: [
                parse_finite(get("fragment_score")?, 0.0),
                parse_finite(get("matched_cleavages")?, 0.0),
                parse_finite(get("ar_total_log_probability")?, -100.0),
                parse_finite(get("ar_mean_log_probability")?, -10.0),
                parse_finite(get("neural_all_mask_log_probability")?, -10.0),
                parse_finite(get("neural_length_log_probability")?, -10.0),
                mass_error,
                (candidate_len - predicted_len).abs(),
                if parse_bool(get("from_diffusion")?) {
                    1.0
                } else {
                    0.0
                },
                if parse_bool(get("from_causal_beam")?) {
                    1.0
                } else {
                    0.0
                },
                if parse_bool(get("from_reverse_causal_beam")?) {
                    1.0
                } else {
                    0.0
                },
                if parse_bool(get("from_bidirectional_mitm")?) {
                    1.0
                } else {
                    0.0
                },
            ],
            exact: parse_bool(get("peptidoform_exact")?),
            il_exact: parse_bool(get("il_sequence_exact")?),
            legacy_top1: get("fragment_causal_mass_rank")?.trim() == "1",
        });
    }
    if let Some(id) = current_id {
        groups.push(CandidateGroup {
            record_index: id,
            rows: current_rows,
        });
    }
    Ok(groups
        .into_iter()
        .filter(|group| !group.rows.is_empty())
        .collect())
}

fn fit_normalization(groups: &[&CandidateGroup]) -> Normalization {
    let mut mean = vec![0.0; FEATURE_NAMES.len()];
    let mut n: f64 = 0.0;
    for group in groups {
        for row in &group.rows {
            n += 1.0;
            for j in 0..FEATURE_NAMES.len() {
                mean[j] += row.raw_features[j];
            }
        }
    }
    for value in &mut mean {
        *value /= n.max(1.0);
    }
    let mut var = vec![0.0; FEATURE_NAMES.len()];
    for group in groups {
        for row in &group.rows {
            for j in 0..FEATURE_NAMES.len() {
                let d = row.raw_features[j] - mean[j];
                var[j] += d * d;
            }
        }
    }
    let std: Vec<f64> = var
        .into_iter()
        .map(|value| (value / n.max(1.0)).sqrt().max(1.0e-8))
        .collect();
    Normalization {
        feature_names: FEATURE_NAMES.iter().map(|s| s.to_string()).collect(),
        mean,
        std,
    }
}

fn normalized(row: &CandidateRow, norm: &Normalization) -> [f64; FEATURE_NAMES.len()] {
    let mut out = [0.0; FEATURE_NAMES.len()];
    for j in 0..FEATURE_NAMES.len() {
        out[j] = (row.raw_features[j] - norm.mean[j]) / norm.std[j];
    }
    out
}

fn accumulate_group_gradient(
    group: &CandidateGroup,
    norm: &Normalization,
    model: &Model,
    grad: &mut Gradients,
) -> Result<f64> {
    let positives: Vec<usize> = group
        .rows
        .iter()
        .enumerate()
        .filter_map(|(i, row)| row.exact.then_some(i))
        .collect();
    if positives.is_empty() {
        anyhow::bail!(
            "supervised group {} unexpectedly has no exact target",
            group.record_index
        );
    }
    let mut xs = Vec::with_capacity(group.rows.len());
    let mut hidden = Vec::with_capacity(group.rows.len());
    let mut scores = Vec::with_capacity(group.rows.len());
    for row in &group.rows {
        let x = normalized(row, norm);
        let (score, h) = model.score(&x);
        xs.push(x);
        hidden.push(h);
        scores.push(score);
    }
    let max_score = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exp_scores: Vec<f64> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let denom: f64 = exp_scores.iter().sum();
    let positive_mass: f64 = positives.iter().map(|&i| exp_scores[i]).sum();
    let loss = -((positive_mass / denom.max(f64::MIN_POSITIVE)).max(f64::MIN_POSITIVE)).ln();
    let pos_denom = positive_mass.max(f64::MIN_POSITIVE);

    for i in 0..group.rows.len() {
        let p_all = exp_scores[i] / denom;
        let p_pos = if group.rows[i].exact {
            exp_scores[i] / pos_denom
        } else {
            0.0
        };
        let ds = p_all - p_pos;
        grad.b2 += ds;
        for h in 0..HIDDEN {
            grad.w2[h] += ds * hidden[i][h];
            let dh = ds * model.w2[h] * (1.0 - hidden[i][h] * hidden[i][h]);
            grad.b1[h] += dh;
            for j in 0..FEATURE_NAMES.len() {
                grad.w1[h * FEATURE_NAMES.len() + j] += dh * xs[i][j];
            }
        }
    }
    Ok(loss)
}

fn evaluate(groups: &[CandidateGroup], norm: &Normalization, model: &Model) -> EvalMetrics {
    let mut m = EvalMetrics {
        records: groups.len(),
        oracle_exact: 0,
        oracle_il: 0,
        top1_exact: 0,
        top1_il: 0,
        legacy_top1_exact: 0,
        legacy_top1_il: 0,
    };
    for group in groups {
        m.oracle_exact += usize::from(group.rows.iter().any(|row| row.exact));
        m.oracle_il += usize::from(group.rows.iter().any(|row| row.il_exact));
        let mut best = None::<(f64, usize)>;
        for (i, row) in group.rows.iter().enumerate() {
            let x = normalized(row, norm);
            let score = model.score(&x).0;
            if best.map(|(s, _)| score > s).unwrap_or(true) {
                best = Some((score, i));
            }
        }
        if let Some((_, i)) = best {
            m.top1_exact += usize::from(group.rows[i].exact);
            m.top1_il += usize::from(group.rows[i].il_exact);
        }
        if let Some(row) = group.rows.iter().find(|row| row.legacy_top1) {
            m.legacy_top1_exact += usize::from(row.exact);
            m.legacy_top1_il += usize::from(row.il_exact);
        }
    }
    m
}

fn parse_bool(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "true" | "1" | "yes"
    )
}

fn parse_finite(value: &str, fallback: f64) -> f64 {
    value
        .trim()
        .parse::<f64>()
        .ok()
        .filter(|v| v.is_finite())
        .unwrap_or(fallback)
}

fn deterministic_shuffle(values: &mut [usize], seed: u64) {
    let mut rng = Rng::new(seed);
    for i in (1..values.len()).rev() {
        let j = (rng.next_u64() as usize) % (i + 1);
        values.swap(i, j);
    }
}

struct Rng {
    state: u64,
}
impl Rng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x9e3779b97f4a7c15,
        }
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(fragment: f64, exact: bool) -> CandidateRow {
        CandidateRow {
            raw_features: [
                fragment, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ],
            exact,
            il_exact: exact,
            legacy_top1: false,
        }
    }

    #[test]
    fn normalization_is_finite() {
        let group = CandidateGroup {
            record_index: 1,
            rows: vec![row(1.0, true), row(2.0, false)],
        };
        let norm = fit_normalization(&[&group]);
        assert_eq!(norm.mean.len(), FEATURE_NAMES.len());
        assert!(norm.std.iter().all(|v| v.is_finite() && *v > 0.0));
    }

    #[test]
    fn listwise_gradient_is_finite() {
        let group = CandidateGroup {
            record_index: 1,
            rows: vec![row(3.0, true), row(0.0, false)],
        };
        let norm = fit_normalization(&[&group]);
        let model = Model::new(SEED);
        let mut grad = Gradients::zeros();
        let loss = accumulate_group_gradient(&group, &norm, &model, &mut grad).unwrap();
        assert!(loss.is_finite());
        assert!(grad.w1.iter().all(|v| v.is_finite()));
        assert!(grad.w2.iter().all(|v| v.is_finite()));
    }
}
