//! Train the bounded v0.14.1 contextual hierarchical residual reranker.
//!
//! v0.14.1 keeps the accepted v0.13.23 candidate generator and final legacy score frozen.
//! It learns only a compact residual correction from target-independent candidate evidence,
//! within-spectrum relative/rank features, and proposal-source consensus. Supervision is
//! hierarchical: every TRAIN spectrum with an I/L-equivalent target contributes an I/L
//! listwise objective, and spectra that also contain the literal peptidoform contribute an
//! additional exact-target listwise objective with the same unit weight. TEST input is
//! forbidden by policy and validation is evaluated only after the fixed training schedule.

use anyhow::{Context, Result};
use serde::Serialize;
use std::collections::HashMap;
use std::env;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

const VERSION: &str = "v0.14.1";
const HIDDEN: usize = 16;
const EPOCHS: usize = 10;
const BATCH_GROUPS: usize = 16;
const LEARNING_RATE: f64 = 1.0e-3;
const WEIGHT_DECAY: f64 = 1.0e-4;
const ADAM_BETA1: f64 = 0.9;
const ADAM_BETA2: f64 = 0.999;
const ADAM_EPS: f64 = 1.0e-8;
const SEED: u64 = 20_260_914;

const REQUIRED_LITERAL_TOP1: usize = 28;
const REQUIRED_IL_TOP1: usize = 42;
const REQUIRED_LITERAL_ORACLE: usize = 44;
const REQUIRED_IL_ORACLE: usize = 54;

const FEATURE_NAMES: [&str; 27] = [
    "fragment_score",
    "matched_cleavages",
    "ar_total_log_probability",
    "ar_mean_log_probability",
    "neural_all_mask_log_probability",
    "neural_length_log_probability",
    "abs_mass_error_da",
    "candidate_length_error",
    "proposal_source_count",
    "from_diffusion",
    "from_causal_beam",
    "from_reverse_causal_beam",
    "from_bidirectional_mitm",
    "legacy_rank_percentile",
    "legacy_gap_to_best",
    "fragment_rank_percentile",
    "fragment_gap_to_best",
    "matched_cleavage_rank_percentile",
    "matched_cleavage_gap_to_best",
    "ar_total_rank_percentile",
    "ar_total_gap_to_best",
    "neural_rank_percentile",
    "neural_gap_to_best",
    "mass_error_rank_percentile",
    "mass_error_gap_to_best",
    "length_error_rank_percentile",
    "length_error_gap_to_best",
];

const IDX_FRAGMENT: usize = 0;
const IDX_MATCHED: usize = 1;
const IDX_AR_TOTAL: usize = 2;
const IDX_NEURAL: usize = 4;
const IDX_MASS_ERROR: usize = 6;
const IDX_LENGTH_ERROR: usize = 7;
const IDX_LEGACY_RANK_PCT: usize = 13;
const IDX_LEGACY_GAP: usize = 14;
const IDX_FRAGMENT_RANK_PCT: usize = 15;
const IDX_FRAGMENT_GAP: usize = 16;
const IDX_MATCHED_RANK_PCT: usize = 17;
const IDX_MATCHED_GAP: usize = 18;
const IDX_AR_TOTAL_RANK_PCT: usize = 19;
const IDX_AR_TOTAL_GAP: usize = 20;
const IDX_NEURAL_RANK_PCT: usize = 21;
const IDX_NEURAL_GAP: usize = 22;
const IDX_MASS_RANK_PCT: usize = 23;
const IDX_MASS_GAP: usize = 24;
const IDX_LENGTH_RANK_PCT: usize = 25;
const IDX_LENGTH_GAP: usize = 26;

#[derive(Debug, Clone)]
struct CandidateRow {
    raw_features: [f64; FEATURE_NAMES.len()],
    exact: bool,
    il_exact: bool,
    legacy_score: f64,
    legacy_rank: usize,
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
    base_score: String,
    supervision_policy: String,
    validation_selection_policy: String,
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
    train_il_supervised_groups: usize,
    train_exact_supervised_groups: usize,
    validation_groups: usize,
    feature_names: Vec<String>,
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
    fn new_zero_residual(seed: u64) -> Self {
        let input = FEATURE_NAMES.len();
        let mut rng = Rng::new(seed);
        let scale1 = (6.0 / (input + HIDDEN) as f64).sqrt();
        let w1 = (0..input * HIDDEN)
            .map(|_| (rng.next_f64() * 2.0 - 1.0) * scale1)
            .collect();
        Self {
            w1,
            b1: vec![0.0; HIDDEN],
            // Zero output weights make epoch-0 exactly the frozen legacy ranker.
            w2: vec![0.0; HIDDEN],
            b2: 0.0,
        }
    }

    fn residual_score(&self, x: &[f64; FEATURE_NAMES.len()]) -> (f64, [f64; HIDDEN]) {
        let mut hidden = [0.0; HIDDEN];
        for (h, value) in hidden.iter_mut().enumerate() {
            let mut z = self.b1[h];
            for (j, &xj) in x.iter().enumerate() {
                z += self.w1[h * FEATURE_NAMES.len() + j] * xj;
            }
            *value = z.tanh();
        }
        let mut residual = self.b2;
        for (h, &hidden_value) in hidden.iter().enumerate() {
            residual += self.w2[h] * hidden_value;
        }
        (residual, hidden)
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
        for value in &mut self.w1 {
            *value *= factor;
        }
        for value in &mut self.b1 {
            *value *= factor;
        }
        for value in &mut self.w2 {
            *value *= factor;
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

#[derive(Debug, Clone)]
struct GroupForward {
    xs: Vec<[f64; FEATURE_NAMES.len()]>,
    hidden: Vec<[f64; HIDDEN]>,
    scores: Vec<f64>,
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 4 {
        anyhow::bail!(
            "usage: foundation_train_contextual_reranker TRAIN_CANDIDATES.tsv VALIDATION_CANDIDATES.tsv OUTPUT_DIR"
        );
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
        .filter(|group| group.rows.iter().any(|row| row.il_exact))
        .collect();
    let exact_supervised_groups = supervised_train
        .iter()
        .filter(|group| group.rows.iter().any(|row| row.exact))
        .count();
    if supervised_train.is_empty() {
        anyhow::bail!("no TRAIN candidate group contains an I/L-equivalent target");
    }

    // Normalization is target-independent and uses all mass-valid TRAIN candidates,
    // including groups that do not contain a supervised target.
    let all_train_refs: Vec<&CandidateGroup> = train_groups.iter().collect();
    let normalization = fit_normalization(&all_train_refs);
    let mut model = Model::new_zero_residual(SEED);

    let initial = evaluate(&validation_groups, &normalization, &model);
    let initial_parity = initial.top1_exact == initial.legacy_top1_exact
        && initial.top1_il == initial.legacy_top1_il;
    if !initial_parity {
        anyhow::bail!(
            "zero-residual initialization failed frozen-ranker parity: residual={}/{} legacy={}/{}",
            initial.top1_exact,
            initial.top1_il,
            initial.legacy_top1_exact,
            initial.legacy_top1_il
        );
    }

    println!("contextual_reranker_version\t{VERSION}");
    println!("objective\thierarchical_il_then_exact_listwise_unit_weight");
    println!("architecture\tfrozen_v01323_base_plus_contextual_residual_mlp_27x16x1");
    println!("proposal_policy\tv01323_final_two_view_fixed_budget_frozen");
    println!("base_score\tfragment_score+0.1*n_to_c_ar_total_log_probability_frozen");
    println!("residual_initialization\texact_zero");
    println!("context_features\twithin_spectrum_rank_percentile+gap_to_best+proposal_consensus");
    println!("supervision_policy\til_equivalent_all_groups_plus_literal_exact_when_available");
    println!("validation_selection_policy\tnone_fixed_epoch10_evaluation_only");
    println!("test_partition_consumed\tNO");
    println!("train_groups\t{}", train_groups.len());
    println!("train_il_supervised_groups\t{}", supervised_train.len());
    println!("train_exact_supervised_groups\t{exact_supervised_groups}");
    println!("validation_groups\t{}", validation_groups.len());
    println!(
        "initial_residual_parity\tliteral={}\til={}\tlegacy_literal={}\tlegacy_il={}\tparity=YES",
        initial.top1_exact, initial.top1_il, initial.legacy_top1_exact, initial.legacy_top1_il
    );
    println!("epochs\t{EPOCHS}");
    println!("batch_groups\t{BATCH_GROUPS}");
    println!("learning_rate\t{LEARNING_RATE}");
    println!("weight_decay\t{WEIGHT_DECAY}");
    println!("seed\t{SEED}");

    let mut adam = AdamState::new();
    let mut grad = Gradients::zeros();
    let mut order: Vec<usize> = (0..supervised_train.len()).collect();

    for epoch in 0..EPOCHS {
        deterministic_shuffle(&mut order, SEED ^ (epoch as u64 + 1));
        let mut epoch_loss = 0.0;
        let mut epoch_terms = 0usize;
        let mut epoch_il_terms = 0usize;
        let mut epoch_exact_terms = 0usize;
        let mut batch_groups = 0usize;
        let mut batch_terms = 0usize;
        grad.clear();

        for &index in &order {
            let group = supervised_train[index];
            let forward = forward_group(group, &normalization, &model);

            let il_mask: Vec<bool> = group.rows.iter().map(|row| row.il_exact).collect();
            let il_loss =
                accumulate_listwise_gradient(group, &forward, &model, &il_mask, &mut grad)?;
            epoch_loss += il_loss;
            epoch_terms += 1;
            epoch_il_terms += 1;
            batch_terms += 1;

            if group.rows.iter().any(|row| row.exact) {
                let exact_mask: Vec<bool> = group.rows.iter().map(|row| row.exact).collect();
                let exact_loss =
                    accumulate_listwise_gradient(group, &forward, &model, &exact_mask, &mut grad)?;
                epoch_loss += exact_loss;
                epoch_terms += 1;
                epoch_exact_terms += 1;
                batch_terms += 1;
            }

            batch_groups += 1;
            if batch_groups == BATCH_GROUPS {
                grad.scale(1.0 / batch_terms.max(1) as f64);
                adam.update(&mut model, &grad);
                grad.clear();
                batch_groups = 0;
                batch_terms = 0;
            }
        }

        if batch_groups > 0 {
            grad.scale(1.0 / batch_terms.max(1) as f64);
            adam.update(&mut model, &grad);
        }

        println!(
            "training_epoch\tepoch={}\tmean_hierarchical_loss={:.8}\tobjective_terms={}\til_terms={}\texact_terms={}",
            epoch + 1,
            epoch_loss / epoch_terms.max(1) as f64,
            epoch_terms,
            epoch_il_terms,
            epoch_exact_terms
        );
    }

    let metrics = evaluate(&validation_groups, &normalization, &model);
    let literal_gate = metrics.top1_exact >= REQUIRED_LITERAL_TOP1;
    let il_gate = metrics.top1_il >= REQUIRED_IL_TOP1;
    let oracle_gate =
        metrics.oracle_exact >= REQUIRED_LITERAL_ORACLE && metrics.oracle_il >= REQUIRED_IL_ORACLE;
    let gate = literal_gate && il_gate && oracle_gate;

    println!(
        "validation_summary\trecords={}\toracle_literal={}\toracle_il={}\tlegacy_top1_literal={}\tlegacy_top1_il={}\tcontextual_top1_literal={}\tcontextual_top1_il={}",
        metrics.records,
        metrics.oracle_exact,
        metrics.oracle_il,
        metrics.legacy_top1_exact,
        metrics.legacy_top1_il,
        metrics.top1_exact,
        metrics.top1_il
    );
    println!(
        "v0141_acceptance_gate\trequired_literal={}\trequired_il={}\trequired_oracle_literal={}\trequired_oracle_il={}\tobserved_literal={}\tobserved_il={}\tobserved_oracle_literal={}\tobserved_oracle_il={}\tgate={}",
        REQUIRED_LITERAL_TOP1,
        REQUIRED_IL_TOP1,
        REQUIRED_LITERAL_ORACLE,
        REQUIRED_IL_ORACLE,
        metrics.top1_exact,
        metrics.top1_il,
        metrics.oracle_exact,
        metrics.oracle_il,
        if gate { "PASS" } else { "FAIL" }
    );
    println!(
        "v0141_stop_rule\t{}",
        if gate {
            "ACCEPT_CONTEXTUAL_RESIDUAL_AND_FREEZE"
        } else {
            "REJECT_AND_CLOSE_COMPACT_FEATURE_RERANKER_FAMILY"
        }
    );

    fs::create_dir_all(&output_dir)?;
    let checkpoint = RerankerCheckpoint {
        version: VERSION.to_string(),
        objective: "hierarchical_il_then_exact_listwise_unit_weight".to_string(),
        architecture: "frozen_v01323_base_plus_contextual_residual_mlp_27x16x1".to_string(),
        proposal_policy: "v01323_final_two_view_fixed_budget_frozen".to_string(),
        base_score: "fragment_score+0.1*n_to_c_ar_total_log_probability_frozen".to_string(),
        supervision_policy: "il_equivalent_all_groups_plus_literal_exact_when_available"
            .to_string(),
        validation_selection_policy: "none_fixed_epoch10_evaluation_only".to_string(),
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
        train_il_supervised_groups: supervised_train.len(),
        train_exact_supervised_groups: exact_supervised_groups,
        validation_groups: validation_groups.len(),
        feature_names: FEATURE_NAMES.iter().map(|name| name.to_string()).collect(),
        normalization: normalization.clone(),
        w1: model.w1.clone(),
        b1: model.b1.clone(),
        w2: model.w2.clone(),
        b2: model.b2,
    };

    let checkpoint_path = output_dir.join("reranker.yaml");
    serde_yaml::to_writer(BufWriter::new(File::create(&checkpoint_path)?), &checkpoint)?;

    let summary_path = output_dir.join("validation_summary.tsv");
    let mut summary = BufWriter::new(File::create(&summary_path)?);
    writeln!(
        summary,
        "records\toracle_literal\toracle_il\tlegacy_top1_literal\tlegacy_top1_il\tcontextual_top1_literal\tcontextual_top1_il\tgate"
    )?;
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

    let diagnostics_path = output_dir.join("validation_ranking_diagnostics.tsv");
    write_validation_diagnostics(
        &diagnostics_path,
        &validation_groups,
        &normalization,
        &model,
    )?;

    println!("checkpoint\t{}", checkpoint_path.display());
    println!("validation_summary_tsv\t{}", summary_path.display());
    println!(
        "validation_ranking_diagnostics_tsv\t{}",
        diagnostics_path.display()
    );
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
        anyhow::bail!("v0.14.1 forbids TEST-partition inputs; suspicious path {path:?}");
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
        "fragment_causal_score",
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
                contextualize_group(&mut current_rows);
                groups.push(CandidateGroup {
                    record_index: id,
                    rows: std::mem::take(&mut current_rows),
                });
            }
            current_id = Some(record_index);
        }

        let predicted_len = parse_finite(get("predicted_active_tokens")?, 0.0);
        let candidate_len = get("candidate_sequence")?
            .chars()
            .filter(|c| c.is_ascii_alphabetic())
            .count() as f64;
        let mass_error = parse_finite(get("mass_error_da")?, 0.0).abs();
        let from_diffusion = parse_bool(get("from_diffusion")?);
        let from_causal = parse_bool(get("from_causal_beam")?);
        let from_reverse = parse_bool(get("from_reverse_causal_beam")?);
        let from_mitm = parse_bool(get("from_bidirectional_mitm")?);
        let source_count = [from_diffusion, from_causal, from_reverse, from_mitm]
            .iter()
            .filter(|&&value| value)
            .count() as f64;

        let mut raw_features = [0.0; FEATURE_NAMES.len()];
        raw_features[0] = parse_finite(get("fragment_score")?, 0.0);
        raw_features[1] = parse_finite(get("matched_cleavages")?, 0.0);
        raw_features[2] = parse_finite(get("ar_total_log_probability")?, -100.0);
        raw_features[3] = parse_finite(get("ar_mean_log_probability")?, -10.0);
        raw_features[4] = parse_finite(get("neural_all_mask_log_probability")?, -10.0);
        raw_features[5] = parse_finite(get("neural_length_log_probability")?, -10.0);
        raw_features[6] = mass_error;
        raw_features[7] = (candidate_len - predicted_len).abs();
        raw_features[8] = source_count;
        raw_features[9] = if from_diffusion { 1.0 } else { 0.0 };
        raw_features[10] = if from_causal { 1.0 } else { 0.0 };
        raw_features[11] = if from_reverse { 1.0 } else { 0.0 };
        raw_features[12] = if from_mitm { 1.0 } else { 0.0 };

        current_rows.push(CandidateRow {
            raw_features,
            exact: parse_bool(get("peptidoform_exact")?),
            il_exact: parse_bool(get("il_sequence_exact")?),
            legacy_score: parse_finite(get("fragment_causal_score")?, f64::NEG_INFINITY),
            legacy_rank: parse_usize(get("fragment_causal_mass_rank")?, usize::MAX),
        });
    }

    if let Some(id) = current_id {
        contextualize_group(&mut current_rows);
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

fn contextualize_group(rows: &mut [CandidateRow]) {
    if rows.is_empty() {
        return;
    }

    let legacy: Vec<f64> = rows.iter().map(|row| row.legacy_score).collect();
    let fragment: Vec<f64> = rows
        .iter()
        .map(|row| row.raw_features[IDX_FRAGMENT])
        .collect();
    let matched: Vec<f64> = rows
        .iter()
        .map(|row| row.raw_features[IDX_MATCHED])
        .collect();
    let ar_total: Vec<f64> = rows
        .iter()
        .map(|row| row.raw_features[IDX_AR_TOTAL])
        .collect();
    let neural: Vec<f64> = rows
        .iter()
        .map(|row| row.raw_features[IDX_NEURAL])
        .collect();
    let mass_error: Vec<f64> = rows
        .iter()
        .map(|row| row.raw_features[IDX_MASS_ERROR])
        .collect();
    let length_error: Vec<f64> = rows
        .iter()
        .map(|row| row.raw_features[IDX_LENGTH_ERROR])
        .collect();

    write_rank_gap(rows, &legacy, true, IDX_LEGACY_RANK_PCT, IDX_LEGACY_GAP);
    write_rank_gap(
        rows,
        &fragment,
        true,
        IDX_FRAGMENT_RANK_PCT,
        IDX_FRAGMENT_GAP,
    );
    write_rank_gap(rows, &matched, true, IDX_MATCHED_RANK_PCT, IDX_MATCHED_GAP);
    write_rank_gap(
        rows,
        &ar_total,
        true,
        IDX_AR_TOTAL_RANK_PCT,
        IDX_AR_TOTAL_GAP,
    );
    write_rank_gap(rows, &neural, true, IDX_NEURAL_RANK_PCT, IDX_NEURAL_GAP);
    write_rank_gap(rows, &mass_error, false, IDX_MASS_RANK_PCT, IDX_MASS_GAP);
    write_rank_gap(
        rows,
        &length_error,
        false,
        IDX_LENGTH_RANK_PCT,
        IDX_LENGTH_GAP,
    );
}

fn write_rank_gap(
    rows: &mut [CandidateRow],
    values: &[f64],
    higher_is_better: bool,
    rank_index: usize,
    gap_index: usize,
) {
    let percentiles = competition_rank_percentiles(values, higher_is_better);
    let best = if higher_is_better {
        values.iter().copied().fold(f64::NEG_INFINITY, f64::max)
    } else {
        values.iter().copied().fold(f64::INFINITY, f64::min)
    };

    for i in 0..rows.len() {
        rows[i].raw_features[rank_index] = percentiles[i];
        rows[i].raw_features[gap_index] = if higher_is_better {
            (best - values[i]).max(0.0)
        } else {
            (values[i] - best).max(0.0)
        };
    }
}

fn competition_rank_percentiles(values: &[f64], higher_is_better: bool) -> Vec<f64> {
    if values.is_empty() {
        return Vec::new();
    }
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&a, &b| {
        let primary = if higher_is_better {
            values[b].total_cmp(&values[a])
        } else {
            values[a].total_cmp(&values[b])
        };
        primary.then_with(|| a.cmp(&b))
    });

    let denom = values.len().saturating_sub(1).max(1) as f64;
    let mut out = vec![0.0; values.len()];
    let mut pos = 0usize;
    while pos < order.len() {
        let value = values[order[pos]];
        let mut end = pos + 1;
        while end < order.len() && values[order[end]] == value {
            end += 1;
        }
        let percentile = pos as f64 / denom;
        for &index in &order[pos..end] {
            out[index] = percentile;
        }
        pos = end;
    }
    out
}

fn fit_normalization(groups: &[&CandidateGroup]) -> Normalization {
    let mut mean = vec![0.0; FEATURE_NAMES.len()];
    let mut n: f64 = 0.0;
    for group in groups {
        for row in &group.rows {
            n += 1.0;
            for (j, value) in mean.iter_mut().enumerate() {
                *value += row.raw_features[j];
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
                let delta = row.raw_features[j] - mean[j];
                var[j] += delta * delta;
            }
        }
    }
    let std = var
        .into_iter()
        .map(|value| (value / n.max(1.0)).sqrt().max(1.0e-8))
        .collect();

    Normalization {
        feature_names: FEATURE_NAMES.iter().map(|name| name.to_string()).collect(),
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

fn forward_group(group: &CandidateGroup, norm: &Normalization, model: &Model) -> GroupForward {
    let mut xs = Vec::with_capacity(group.rows.len());
    let mut hidden = Vec::with_capacity(group.rows.len());
    let mut scores = Vec::with_capacity(group.rows.len());

    for row in &group.rows {
        let x = normalized(row, norm);
        let (residual, h) = model.residual_score(&x);
        xs.push(x);
        hidden.push(h);
        scores.push(row.legacy_score + residual);
    }

    GroupForward { xs, hidden, scores }
}

fn accumulate_listwise_gradient(
    group: &CandidateGroup,
    forward: &GroupForward,
    model: &Model,
    positive_mask: &[bool],
    grad: &mut Gradients,
) -> Result<f64> {
    if positive_mask.len() != group.rows.len() {
        anyhow::bail!("internal positive-mask length mismatch");
    }
    if !positive_mask.iter().any(|&value| value) {
        anyhow::bail!(
            "supervised group {} unexpectedly has no positive candidate",
            group.record_index
        );
    }

    let max_score = forward
        .scores
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let exp_scores: Vec<f64> = forward
        .scores
        .iter()
        .map(|&score| (score - max_score).exp())
        .collect();
    let denom: f64 = exp_scores.iter().sum();
    let positive_mass: f64 = positive_mask
        .iter()
        .enumerate()
        .filter_map(|(i, &positive)| positive.then_some(exp_scores[i]))
        .sum();
    let probability = (positive_mass / denom.max(f64::MIN_POSITIVE)).max(f64::MIN_POSITIVE);
    let loss = -probability.ln();
    let positive_denom = positive_mass.max(f64::MIN_POSITIVE);

    for i in 0..group.rows.len() {
        let p_all = exp_scores[i] / denom.max(f64::MIN_POSITIVE);
        let p_positive = if positive_mask[i] {
            exp_scores[i] / positive_denom
        } else {
            0.0
        };
        let dscore = p_all - p_positive;
        grad.b2 += dscore;
        for h in 0..HIDDEN {
            grad.w2[h] += dscore * forward.hidden[i][h];
            let dh = dscore * model.w2[h] * (1.0 - forward.hidden[i][h] * forward.hidden[i][h]);
            grad.b1[h] += dh;
            for j in 0..FEATURE_NAMES.len() {
                grad.w1[h * FEATURE_NAMES.len() + j] += dh * forward.xs[i][j];
            }
        }
    }

    Ok(loss)
}

fn evaluate(groups: &[CandidateGroup], norm: &Normalization, model: &Model) -> EvalMetrics {
    let mut metrics = EvalMetrics {
        records: groups.len(),
        oracle_exact: 0,
        oracle_il: 0,
        top1_exact: 0,
        top1_il: 0,
        legacy_top1_exact: 0,
        legacy_top1_il: 0,
    };

    for group in groups {
        metrics.oracle_exact += usize::from(group.rows.iter().any(|row| row.exact));
        metrics.oracle_il += usize::from(group.rows.iter().any(|row| row.il_exact));

        if let Some(index) = best_contextual_index(group, norm, model) {
            metrics.top1_exact += usize::from(group.rows[index].exact);
            metrics.top1_il += usize::from(group.rows[index].il_exact);
        }

        if let Some(row) = group.rows.iter().min_by_key(|row| row.legacy_rank) {
            metrics.legacy_top1_exact += usize::from(row.exact);
            metrics.legacy_top1_il += usize::from(row.il_exact);
        }
    }

    metrics
}

fn best_contextual_index(
    group: &CandidateGroup,
    norm: &Normalization,
    model: &Model,
) -> Option<usize> {
    let mut best: Option<(f64, usize, usize)> = None;
    for (i, row) in group.rows.iter().enumerate() {
        let x = normalized(row, norm);
        let score = row.legacy_score + model.residual_score(&x).0;
        match best {
            None => best = Some((score, row.legacy_rank, i)),
            Some((best_score, best_legacy_rank, _)) => {
                if score > best_score || (score == best_score && row.legacy_rank < best_legacy_rank)
                {
                    best = Some((score, row.legacy_rank, i));
                }
            }
        }
    }
    best.map(|(_, _, index)| index)
}

fn contextual_order(group: &CandidateGroup, norm: &Normalization, model: &Model) -> Vec<usize> {
    let mut scored: Vec<(usize, f64, usize)> = group
        .rows
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let x = normalized(row, norm);
            let score = row.legacy_score + model.residual_score(&x).0;
            (i, score, row.legacy_rank)
        })
        .collect();
    scored.sort_by(|a, b| {
        b.1.total_cmp(&a.1)
            .then_with(|| a.2.cmp(&b.2))
            .then_with(|| a.0.cmp(&b.0))
    });
    scored.into_iter().map(|(index, _, _)| index).collect()
}

fn write_validation_diagnostics(
    path: &Path,
    groups: &[CandidateGroup],
    norm: &Normalization,
    model: &Model,
) -> Result<()> {
    let mut out = BufWriter::new(File::create(path)?);
    writeln!(
        out,
        "record_index\toracle_literal\toracle_il\tlegacy_literal_rank\tlegacy_il_rank\tcontextual_literal_rank\tcontextual_il_rank\tlegacy_top1_literal\tlegacy_top1_il\tcontextual_top1_literal\tcontextual_top1_il"
    )?;

    for group in groups {
        let legacy_literal_rank = group
            .rows
            .iter()
            .filter(|row| row.exact)
            .map(|row| row.legacy_rank)
            .min();
        let legacy_il_rank = group
            .rows
            .iter()
            .filter(|row| row.il_exact)
            .map(|row| row.legacy_rank)
            .min();
        let order = contextual_order(group, norm, model);
        let mut contextual_literal_rank = None;
        let mut contextual_il_rank = None;
        for (rank0, &index) in order.iter().enumerate() {
            let rank = rank0 + 1;
            if contextual_literal_rank.is_none() && group.rows[index].exact {
                contextual_literal_rank = Some(rank);
            }
            if contextual_il_rank.is_none() && group.rows[index].il_exact {
                contextual_il_rank = Some(rank);
            }
            if contextual_literal_rank.is_some() && contextual_il_rank.is_some() {
                break;
            }
        }

        let legacy_top = group.rows.iter().min_by_key(|row| row.legacy_rank);
        let contextual_top = order.first().map(|&index| &group.rows[index]);
        writeln!(
            out,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            group.record_index,
            yes_no(legacy_literal_rank.is_some()),
            yes_no(legacy_il_rank.is_some()),
            format_rank(legacy_literal_rank),
            format_rank(legacy_il_rank),
            format_rank(contextual_literal_rank),
            format_rank(contextual_il_rank),
            yes_no(legacy_top.map(|row| row.exact).unwrap_or(false)),
            yes_no(legacy_top.map(|row| row.il_exact).unwrap_or(false)),
            yes_no(contextual_top.map(|row| row.exact).unwrap_or(false)),
            yes_no(contextual_top.map(|row| row.il_exact).unwrap_or(false)),
        )?;
    }
    Ok(())
}

fn format_rank(rank: Option<usize>) -> String {
    rank.map(|value| value.to_string())
        .unwrap_or_else(|| "NA".to_string())
}

fn yes_no(value: bool) -> &'static str {
    if value {
        "YES"
    } else {
        "NO"
    }
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
        .filter(|parsed| parsed.is_finite())
        .unwrap_or(fallback)
}

fn parse_usize(value: &str, fallback: usize) -> usize {
    value.trim().parse::<usize>().unwrap_or(fallback)
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
            state: seed ^ 0x9e37_79b9_7f4a_7c15,
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1_u64 << 53) as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(
        fragment: f64,
        matched: f64,
        legacy: f64,
        legacy_rank: usize,
        exact: bool,
        il_exact: bool,
        source_count: usize,
    ) -> CandidateRow {
        let mut features = [0.0; FEATURE_NAMES.len()];
        features[IDX_FRAGMENT] = fragment;
        features[IDX_MATCHED] = matched;
        features[IDX_AR_TOTAL] = legacy - fragment;
        features[IDX_NEURAL] = -2.0;
        features[8] = source_count as f64;
        CandidateRow {
            raw_features: features,
            exact,
            il_exact,
            legacy_score: legacy,
            legacy_rank,
        }
    }

    #[test]
    fn contextual_features_are_tie_aware_and_finite() {
        let mut rows = vec![
            row(5.0, 4.0, 4.0, 1, true, true, 3),
            row(5.0, 4.0, 3.0, 2, false, true, 2),
            row(1.0, 1.0, 0.0, 3, false, false, 1),
        ];
        contextualize_group(&mut rows);
        assert_eq!(rows[0].raw_features[IDX_FRAGMENT_RANK_PCT], 0.0);
        assert_eq!(rows[1].raw_features[IDX_FRAGMENT_RANK_PCT], 0.0);
        assert!(rows
            .iter()
            .flat_map(|row| row.raw_features)
            .all(|value| value.is_finite()));
    }

    #[test]
    fn zero_residual_preserves_legacy_order() {
        let mut rows = vec![
            row(3.0, 3.0, 2.0, 2, true, true, 2),
            row(4.0, 4.0, 3.0, 1, false, false, 1),
        ];
        contextualize_group(&mut rows);
        let group = CandidateGroup {
            record_index: 1,
            rows,
        };
        let norm = fit_normalization(&[&group]);
        let model = Model::new_zero_residual(SEED);
        assert_eq!(best_contextual_index(&group, &norm, &model), Some(1));
    }

    #[test]
    fn hierarchical_gradient_uses_il_and_exact_positive_sets() {
        let mut rows = vec![
            row(4.0, 4.0, 3.0, 1, true, true, 3),
            row(3.5, 4.0, 2.5, 2, false, true, 2),
            row(1.0, 1.0, 0.0, 3, false, false, 1),
        ];
        contextualize_group(&mut rows);
        let group = CandidateGroup {
            record_index: 1,
            rows,
        };
        let norm = fit_normalization(&[&group]);
        let model = Model::new_zero_residual(SEED);
        let forward = forward_group(&group, &norm, &model);
        let mut grad = Gradients::zeros();
        let il_mask: Vec<bool> = group.rows.iter().map(|row| row.il_exact).collect();
        let exact_mask: Vec<bool> = group.rows.iter().map(|row| row.exact).collect();
        let il_loss =
            accumulate_listwise_gradient(&group, &forward, &model, &il_mask, &mut grad).unwrap();
        let exact_loss =
            accumulate_listwise_gradient(&group, &forward, &model, &exact_mask, &mut grad).unwrap();
        assert!(il_loss.is_finite());
        assert!(exact_loss.is_finite());
        assert!(grad.w2.iter().all(|value| value.is_finite()));
    }
}
