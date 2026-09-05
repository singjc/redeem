//! Train the bounded v0.15.0 spectrum-candidate interaction reranker.
//!
//! v0.15.0 is the first ranking architecture in this project that consumes learned
//! spectrum-conditioned candidate representations rather than only hand-engineered scalar
//! evidence. The accepted v0.13.23 proposal set and legacy score remain frozen. Candidate
//! representations come from the frozen N->C causal decoder after its existing cross-attention
//! to the observed spectrum/precursor context. A small residual MLP receives the pooled decoder
//! representation together with its elementwise product against the frozen pooled spectrum
//! embedding. Supervision remains the accepted hierarchical I/L + literal listwise objective.
//!
//! No candidate generation occurs here. TRAIN and VALIDATION candidate TSVs are verified against
//! the benchmark manifest, and TEST-partition records are rejected before any model execution.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use redeem_properties::foundation::{
    load_foundation_corpus, parse_modified_peptide, read_foundation_training_run_config,
    FoundationBenchmarkManifest, FoundationCausalCollator, FoundationDiffusionConfig,
    FoundationPartition, FoundationSpectrum, FoundationSpectrumCollator, FoundationTrainingRecord,
    PeptideSpectrumCausalModel, PeptidoformInput, PrecursorContextBatch,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::env;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

const VERSION: &str = "v0.15.0";
const EXPECTED_MODEL_DIM: usize = 192;
const INTERACTION_DIM: usize = EXPECTED_MODEL_DIM * 2;
const HIDDEN: usize = 8;
const TRAIN_HARD_WINDOW: usize = 32;
const VALIDATION_INTERACTION_WINDOW: usize = 128;
const ENCODE_BATCH: usize = 32;
const EPOCHS: usize = 10;
const BATCH_GROUPS: usize = 16;
const LEARNING_RATE: f64 = 1.0e-3;
const WEIGHT_DECAY: f64 = 1.0e-4;
const ADAM_BETA1: f64 = 0.9;
const ADAM_BETA2: f64 = 0.999;
const ADAM_EPS: f64 = 1.0e-8;
const SEED: u64 = 20_260_915;

const REQUIRED_LITERAL_TOP1: usize = 28;
const REQUIRED_IL_TOP1: usize = 42;
const REQUIRED_LITERAL_ORACLE: usize = 44;
const REQUIRED_IL_ORACLE: usize = 54;

#[derive(Debug, Deserialize)]
struct UnifiedCheckpointMetadata {
    inverse_config: FoundationDiffusionConfig,
}

#[derive(Debug, Clone)]
struct CandidateRow {
    sequence: String,
    modifications: String,
    exact: bool,
    il_exact: bool,
    legacy_score: f64,
    legacy_rank: usize,
    interaction: Option<Vec<f32>>,
}

#[derive(Debug, Clone)]
struct CandidateGroup {
    record_index: usize,
    rows: Vec<CandidateRow>,
    training_indices: Vec<usize>,
}

#[derive(Debug, Clone, Serialize)]
struct Normalization {
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
    representation: String,
    supervision_policy: String,
    validation_selection_policy: String,
    test_partition_consumed: bool,
    seed: u64,
    model_dim: usize,
    interaction_dim: usize,
    hidden_dim: usize,
    train_hard_window: usize,
    validation_interaction_window: usize,
    encode_batch: usize,
    epochs: usize,
    batch_groups: usize,
    learning_rate: f64,
    weight_decay: f64,
    training_yaml: String,
    unified_checkpoint: String,
    train_candidate_tsv: String,
    validation_candidate_tsv: String,
    train_groups: usize,
    train_il_supervised_groups: usize,
    train_exact_supervised_groups: usize,
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
    fn new_zero_residual(seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let scale = (6.0 / (INTERACTION_DIM + HIDDEN) as f64).sqrt();
        let w1 = (0..INTERACTION_DIM * HIDDEN)
            .map(|_| (rng.next_f64() * 2.0 - 1.0) * scale)
            .collect();
        Self {
            w1,
            b1: vec![0.0; HIDDEN],
            w2: vec![0.0; HIDDEN],
            b2: 0.0,
        }
    }

    fn residual_score(&self, raw: &[f32], norm: &Normalization) -> (f64, [f64; HIDDEN]) {
        let mut hidden = [0.0f64; HIDDEN];
        for (h, value) in hidden.iter_mut().enumerate() {
            let mut z = self.b1[h];
            let base = h * INTERACTION_DIM;
            for j in 0..INTERACTION_DIM {
                let x = (raw[j] as f64 - norm.mean[j]) / norm.std[j];
                z += self.w1[base + j] * x;
            }
            *value = z.tanh();
        }
        let mut residual = self.b2;
        for h in 0..HIDDEN {
            residual += self.w2[h] * hidden[h];
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
            w1: vec![0.0; INTERACTION_DIM * HIDDEN],
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
            mw1: vec![0.0; INTERACTION_DIM * HIDDEN],
            vw1: vec![0.0; INTERACTION_DIM * HIDDEN],
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
    exact_in_interaction_window: usize,
    il_in_interaction_window: usize,
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 6 {
        anyhow::bail!(
            "usage: foundation_train_interaction_reranker RUN.yaml UNIFIED_CHECKPOINT TRAIN_CANDIDATES.tsv VALIDATION_CANDIDATES.tsv OUTPUT_DIR"
        );
    }
    let training_yaml = PathBuf::from(&args[1]);
    let unified_checkpoint = PathBuf::from(&args[2]);
    let train_path = PathBuf::from(&args[3]);
    let validation_path = PathBuf::from(&args[4]);
    let output_dir = PathBuf::from(&args[5]);

    reject_test_path(&train_path)?;
    reject_test_path(&validation_path)?;

    let run = read_foundation_training_run_config(&training_yaml)
        .with_context(|| format!("read foundation run config {training_yaml:?}"))?;
    let corpus = load_foundation_corpus(&run.corpus).context("load foundation corpus")?;
    let benchmark = FoundationBenchmarkManifest::read_tsv(&run.benchmark_manifest)
        .with_context(|| format!("read benchmark manifest {:?}", run.benchmark_manifest))?;
    benchmark.validate_against_records(&corpus.records)?;

    let mut train_groups = read_candidate_groups(&train_path)
        .with_context(|| format!("read train candidate TSV {train_path:?}"))?;
    let mut validation_groups = read_candidate_groups(&validation_path)
        .with_context(|| format!("read validation candidate TSV {validation_path:?}"))?;
    if train_groups.is_empty() || validation_groups.is_empty() {
        anyhow::bail!("TRAIN and VALIDATION candidate TSVs must both contain mass-valid groups");
    }

    verify_partition(
        &train_groups,
        &benchmark,
        FoundationPartition::Train,
        "TRAIN",
    )?;
    verify_partition(
        &validation_groups,
        &benchmark,
        FoundationPartition::Validation,
        "VALIDATION",
    )?;

    let train_il_supervised = train_groups
        .iter()
        .filter(|group| group.rows.iter().any(|row| row.il_exact))
        .count();
    let train_exact_supervised = train_groups
        .iter()
        .filter(|group| group.rows.iter().any(|row| row.exact))
        .count();
    if train_il_supervised == 0 {
        anyhow::bail!("no TRAIN group contains an I/L-equivalent target");
    }

    for group in &mut train_groups {
        if group.rows.iter().any(|row| row.il_exact) {
            group.training_indices = fixed_training_indices(group);
        }
    }

    let metadata_path = unified_checkpoint.join("metadata.yaml");
    let metadata: UnifiedCheckpointMetadata = serde_yaml::from_str(
        &fs::read_to_string(&metadata_path)
            .with_context(|| format!("read unified metadata {metadata_path:?}"))?,
    )?;
    let config = metadata.inverse_config;
    config.validate().map_err(anyhow::Error::msg)?;
    if config.model_dim != EXPECTED_MODEL_DIM {
        anyhow::bail!(
            "v0.15.0 fixed interaction head expects causal model_dim {}, checkpoint has {}",
            EXPECTED_MODEL_DIM,
            config.model_dim
        );
    }

    let device = Device::Cpu;
    let causal_varmap = VarMap::new();
    let causal_vb = VarBuilder::from_varmap(&causal_varmap, DType::F32, &device);
    let causal_model = PeptideSpectrumCausalModel::new(config.clone(), causal_vb)?;
    load_matching_variables(
        &causal_varmap,
        &unified_checkpoint.join("model.safetensors"),
        &device,
    )?;
    let causal_collator = FoundationCausalCollator::new(config.clone())?;
    let spectrum_collator = FoundationSpectrumCollator::new(config.spectrum.clone())?;

    println!("interaction_reranker_version\t{VERSION}");
    println!("objective\thierarchical_il_then_exact_listwise_unit_weight_fixed_hard_negatives");
    println!("architecture\tfrozen_causal_cross_attention_hidden_plus_bilinear_product_residual_mlp_384x8x1");
    println!("proposal_policy\tv01323_final_two_view_fixed_budget_frozen");
    println!("base_score\tfragment_score+0.1*n_to_c_ar_total_log_probability_frozen");
    println!("representation\tmean_pooled_frozen_causal_decoder_hidden+elementwise_product_with_frozen_spectrum_embedding");
    println!("representation_target_visibility\tshifted_prefix_only_clean_next_token_targets_excluded_from_model_input");
    println!("residual_initialization\texact_zero");
    println!("supervision_policy\til_equivalent_all_groups_plus_literal_exact_when_available");
    println!("train_hard_negative_window\t{TRAIN_HARD_WINDOW}");
    println!("validation_interaction_window\t{VALIDATION_INTERACTION_WINDOW}");
    println!("encode_batch\t{ENCODE_BATCH}");
    println!("validation_selection_policy\tnone_fixed_epoch10_evaluation_only");
    println!("test_partition_consumed\tNO");
    println!("train_groups\t{}", train_groups.len());
    println!("train_il_supervised_groups\t{train_il_supervised}");
    println!("train_exact_supervised_groups\t{train_exact_supervised}");
    println!("validation_groups\t{}", validation_groups.len());
    println!("unified_checkpoint\t{}", unified_checkpoint.display());

    let encoded_train = precompute_training_interactions(
        &mut train_groups,
        &corpus.records,
        &causal_model,
        &causal_collator,
        &spectrum_collator,
        &device,
    )?;
    let encoded_validation = precompute_validation_interactions(
        &mut validation_groups,
        &corpus.records,
        &causal_model,
        &causal_collator,
        &spectrum_collator,
        &device,
    )?;
    println!("train_interaction_candidates_encoded\t{encoded_train}");
    println!("validation_interaction_candidates_encoded\t{encoded_validation}");

    let normalization = fit_normalization(&train_groups)?;
    let mut model = Model::new_zero_residual(SEED);
    let initial = evaluate(&validation_groups, &normalization, &model);
    let initial_parity = initial.top1_exact == initial.legacy_top1_exact
        && initial.top1_il == initial.legacy_top1_il;
    if !initial_parity {
        anyhow::bail!(
            "zero-residual interaction initialization failed frozen-ranker parity: residual={}/{} legacy={}/{}",
            initial.top1_exact,
            initial.top1_il,
            initial.legacy_top1_exact,
            initial.legacy_top1_il
        );
    }
    if initial.oracle_exact != REQUIRED_LITERAL_ORACLE || initial.oracle_il != REQUIRED_IL_ORACLE {
        anyhow::bail!(
            "v0.15.0 validation candidate oracle parity failed: expected {}/{} observed {}/{}",
            REQUIRED_LITERAL_ORACLE,
            REQUIRED_IL_ORACLE,
            initial.oracle_exact,
            initial.oracle_il
        );
    }
    println!(
        "initial_residual_parity\tliteral={}\til={}\tlegacy_literal={}\tlegacy_il={}\toracle_literal={}\toracle_il={}\tparity=YES",
        initial.top1_exact,
        initial.top1_il,
        initial.legacy_top1_exact,
        initial.legacy_top1_il,
        initial.oracle_exact,
        initial.oracle_il
    );
    println!(
        "validation_target_window_coverage\tliteral={}\til={}\toracle_literal={}\toracle_il={}",
        initial.exact_in_interaction_window,
        initial.il_in_interaction_window,
        initial.oracle_exact,
        initial.oracle_il
    );
    println!("epochs\t{EPOCHS}");
    println!("batch_groups\t{BATCH_GROUPS}");
    println!("learning_rate\t{LEARNING_RATE}");
    println!("weight_decay\t{WEIGHT_DECAY}");
    println!("seed\t{SEED}");

    let supervised_indices: Vec<usize> = train_groups
        .iter()
        .enumerate()
        .filter_map(|(index, group)| group.rows.iter().any(|row| row.il_exact).then_some(index))
        .collect();
    let mut order = supervised_indices;
    let mut adam = AdamState::new();
    let mut grad = Gradients::zeros();

    for epoch in 0..EPOCHS {
        deterministic_shuffle(&mut order, SEED ^ (epoch as u64 + 1));
        let mut epoch_loss = 0.0;
        let mut epoch_terms = 0usize;
        let mut epoch_il_terms = 0usize;
        let mut epoch_exact_terms = 0usize;
        let mut batch_groups = 0usize;
        let mut batch_terms = 0usize;
        grad.clear();

        for &group_index in &order {
            let group = &train_groups[group_index];
            let il_loss = accumulate_listwise_gradient(
                group,
                &group.training_indices,
                &normalization,
                &model,
                |row| row.il_exact,
                &mut grad,
            )?;
            epoch_loss += il_loss;
            epoch_terms += 1;
            epoch_il_terms += 1;
            batch_terms += 1;

            if group.rows.iter().any(|row| row.exact) {
                let exact_loss = accumulate_listwise_gradient(
                    group,
                    &group.training_indices,
                    &normalization,
                    &model,
                    |row| row.exact,
                    &mut grad,
                )?;
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
    let gate = metrics.oracle_exact >= REQUIRED_LITERAL_ORACLE
        && metrics.oracle_il >= REQUIRED_IL_ORACLE
        && metrics.top1_exact >= REQUIRED_LITERAL_TOP1
        && metrics.top1_il >= REQUIRED_IL_TOP1;

    println!(
        "validation_summary\trecords={}\toracle_literal={}\toracle_il={}\tlegacy_top1_literal={}\tlegacy_top1_il={}\tinteraction_top1_literal={}\tinteraction_top1_il={}\tliteral_in_window={}\til_in_window={}",
        metrics.records,
        metrics.oracle_exact,
        metrics.oracle_il,
        metrics.legacy_top1_exact,
        metrics.legacy_top1_il,
        metrics.top1_exact,
        metrics.top1_il,
        metrics.exact_in_interaction_window,
        metrics.il_in_interaction_window
    );
    println!(
        "v0150_acceptance_gate\trequired_literal={}\trequired_il={}\trequired_oracle_literal={}\trequired_oracle_il={}\tobserved_literal={}\tobserved_il={}\tobserved_oracle_literal={}\tobserved_oracle_il={}\tgate={}",
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
        "v0150_stop_rule\t{}",
        if gate {
            "ACCEPT_INTERACTION_RERANKER_AND_FREEZE"
        } else {
            "REJECT_FROZEN_INTERACTION_HEAD_AND_ESCALATE_TO_TRAINABLE_INTERACTION_BLOCK"
        }
    );

    fs::create_dir_all(&output_dir)?;
    let checkpoint = RerankerCheckpoint {
        version: VERSION.to_string(),
        objective: "hierarchical_il_then_exact_listwise_unit_weight_fixed_hard_negatives".into(),
        architecture:
            "frozen_causal_cross_attention_hidden_plus_bilinear_product_residual_mlp_384x8x1"
                .into(),
        proposal_policy: "v01323_final_two_view_fixed_budget_frozen".into(),
        base_score: "fragment_score+0.1*n_to_c_ar_total_log_probability_frozen".into(),
        representation: "mean_pooled_frozen_causal_decoder_hidden+elementwise_product_with_frozen_spectrum_embedding".into(),
        supervision_policy:
            "il_equivalent_all_groups_plus_literal_exact_when_available".into(),
        validation_selection_policy: "none_fixed_epoch10_evaluation_only".into(),
        test_partition_consumed: false,
        seed: SEED,
        model_dim: EXPECTED_MODEL_DIM,
        interaction_dim: INTERACTION_DIM,
        hidden_dim: HIDDEN,
        train_hard_window: TRAIN_HARD_WINDOW,
        validation_interaction_window: VALIDATION_INTERACTION_WINDOW,
        encode_batch: ENCODE_BATCH,
        epochs: EPOCHS,
        batch_groups: BATCH_GROUPS,
        learning_rate: LEARNING_RATE,
        weight_decay: WEIGHT_DECAY,
        training_yaml: training_yaml.display().to_string(),
        unified_checkpoint: unified_checkpoint.display().to_string(),
        train_candidate_tsv: train_path.display().to_string(),
        validation_candidate_tsv: validation_path.display().to_string(),
        train_groups: train_groups.len(),
        train_il_supervised_groups: train_il_supervised,
        train_exact_supervised_groups: train_exact_supervised,
        validation_groups: validation_groups.len(),
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
        "records\toracle_literal\toracle_il\tlegacy_top1_literal\tlegacy_top1_il\tinteraction_top1_literal\tinteraction_top1_il\tliteral_in_window\til_in_window\tgate"
    )?;
    writeln!(
        summary,
        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
        metrics.records,
        metrics.oracle_exact,
        metrics.oracle_il,
        metrics.legacy_top1_exact,
        metrics.legacy_top1_il,
        metrics.top1_exact,
        metrics.top1_il,
        metrics.exact_in_interaction_window,
        metrics.il_in_interaction_window,
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
        anyhow::bail!("v0.15.0 forbids TEST-partition inputs; suspicious path {path:?}");
    }
    Ok(())
}

fn verify_partition(
    groups: &[CandidateGroup],
    benchmark: &FoundationBenchmarkManifest,
    expected: FoundationPartition,
    label: &str,
) -> Result<()> {
    let allowed: HashSet<usize> = benchmark.partition_indices(expected).into_iter().collect();
    for group in groups {
        if !allowed.contains(&group.record_index) {
            anyhow::bail!(
                "v0.15.0 {label} candidate group {} is not assigned to expected benchmark partition {}",
                group.record_index,
                label
            );
        }
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
        "candidate_sequence",
        "candidate_modifications",
        "fragment_causal_score",
        "fragment_causal_mass_rank",
        "mass_valid",
        "peptidoform_exact",
        "il_sequence_exact",
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
                    training_indices: Vec::new(),
                });
            }
            current_id = Some(record_index);
        }
        current_rows.push(CandidateRow {
            sequence: get("candidate_sequence")?.to_string(),
            modifications: get("candidate_modifications")?.to_string(),
            exact: parse_bool(get("peptidoform_exact")?),
            il_exact: parse_bool(get("il_sequence_exact")?),
            legacy_score: parse_finite(get("fragment_causal_score")?, f64::NEG_INFINITY),
            legacy_rank: parse_usize(get("fragment_causal_mass_rank")?, usize::MAX),
            interaction: None,
        });
    }
    if let Some(id) = current_id {
        groups.push(CandidateGroup {
            record_index: id,
            rows: current_rows,
            training_indices: Vec::new(),
        });
    }
    Ok(groups
        .into_iter()
        .filter(|group| !group.rows.is_empty())
        .collect())
}

fn fixed_training_indices(group: &CandidateGroup) -> Vec<usize> {
    let mut ranked: Vec<usize> = (0..group.rows.len()).collect();
    ranked.sort_by_key(|&index| (group.rows[index].legacy_rank, index));
    ranked.truncate(TRAIN_HARD_WINDOW.min(ranked.len()));
    for (index, row) in group.rows.iter().enumerate() {
        if (row.il_exact || row.exact) && !ranked.contains(&index) {
            ranked.push(index);
        }
    }
    ranked.sort_unstable();
    ranked
}

fn fixed_validation_indices(group: &CandidateGroup) -> Vec<usize> {
    let mut ranked: Vec<usize> = (0..group.rows.len()).collect();
    ranked.sort_by_key(|&index| (group.rows[index].legacy_rank, index));
    ranked.truncate(VALIDATION_INTERACTION_WINDOW.min(ranked.len()));
    ranked.sort_unstable();
    ranked
}

fn precompute_training_interactions(
    groups: &mut [CandidateGroup],
    records: &[FoundationTrainingRecord],
    model: &PeptideSpectrumCausalModel,
    causal_collator: &FoundationCausalCollator,
    spectrum_collator: &FoundationSpectrumCollator,
    device: &Device,
) -> Result<usize> {
    let mut encoded = 0usize;
    let total = groups
        .iter()
        .filter(|group| !group.training_indices.is_empty())
        .count();
    let mut done = 0usize;
    for group in groups.iter_mut() {
        if group.training_indices.is_empty() {
            continue;
        }
        let record = records
            .get(group.record_index)
            .with_context(|| format!("TRAIN record index {} exceeds corpus", group.record_index))?;
        let indices = group.training_indices.clone();
        encoded += encode_group_indices(
            group,
            &indices,
            record,
            model,
            causal_collator,
            spectrum_collator,
            device,
        )?;
        done += 1;
        if done == 1 || done % 50 == 0 || done == total {
            println!(
                "interaction_precompute\tpartition=TRAIN\tgroups_done={}\tgroups_total={}\tcandidates_encoded={}",
                done, total, encoded
            );
        }
    }
    Ok(encoded)
}

fn precompute_validation_interactions(
    groups: &mut [CandidateGroup],
    records: &[FoundationTrainingRecord],
    model: &PeptideSpectrumCausalModel,
    causal_collator: &FoundationCausalCollator,
    spectrum_collator: &FoundationSpectrumCollator,
    device: &Device,
) -> Result<usize> {
    let total = groups.len();
    let mut encoded = 0usize;
    for (done0, group) in groups.iter_mut().enumerate() {
        let record = records.get(group.record_index).with_context(|| {
            format!(
                "VALIDATION record index {} exceeds corpus",
                group.record_index
            )
        })?;
        let indices = fixed_validation_indices(group);
        encoded += encode_group_indices(
            group,
            &indices,
            record,
            model,
            causal_collator,
            spectrum_collator,
            device,
        )?;
        let done = done0 + 1;
        if done == 1 || done % 25 == 0 || done == total {
            println!(
                "interaction_precompute\tpartition=VALIDATION\tgroups_done={}\tgroups_total={}\tcandidates_encoded={}",
                done, total, encoded
            );
        }
    }
    Ok(encoded)
}

fn encode_group_indices(
    group: &mut CandidateGroup,
    indices: &[usize],
    record: &FoundationTrainingRecord,
    model: &PeptideSpectrumCausalModel,
    causal_collator: &FoundationCausalCollator,
    spectrum_collator: &FoundationSpectrumCollator,
    device: &Device,
) -> Result<usize> {
    if indices.is_empty() {
        return Ok(0);
    }
    let spectrum = FoundationSpectrum::from_training_record(record).with_context(|| {
        format!(
            "record {} has no observed spectrum for v0.15.0 interaction encoding",
            group.record_index
        )
    })?;
    let spectrum_batch = spectrum_collator.collate(&[spectrum], device)?;
    let precursor = precursor_context(record, device)?;
    let context = model.prepare_context(&spectrum_batch, &precursor, false)?;

    let mut encoded = 0usize;
    for chunk in indices.chunks(ENCODE_BATCH) {
        let peptides: Vec<PeptidoformInput> = chunk
            .iter()
            .map(|&row_index| exported_peptidoform(&group.rows[row_index]))
            .collect::<Result<Vec<_>>>()?;
        let causal = causal_collator.collate(&peptides, device)?;
        let output = model.forward_t_with_context(&causal.input, &context, false)?;
        let (batch, token_len, model_dim) = output.decoder_hidden.dims3()?;
        if batch != chunk.len() || model_dim != EXPECTED_MODEL_DIM {
            anyhow::bail!(
                "interaction hidden shape mismatch: batch={} expected={} dim={} expected_dim={}",
                batch,
                chunk.len(),
                model_dim,
                EXPECTED_MODEL_DIM
            );
        }
        let mask = causal
            .input
            .token_mask
            .unsqueeze(2)?
            .broadcast_as((batch, token_len, model_dim))?;
        let pooled = output
            .decoder_hidden
            .broadcast_mul(&mask)?
            .sum(1)?
            .broadcast_div(
                &causal
                    .input
                    .token_mask
                    .sum(1)?
                    .clamp(1.0, f64::INFINITY)?
                    .unsqueeze(1)?,
            )?;
        let pooled_rows = pooled.to_vec2::<f32>()?;
        let spectrum_rows = output.spectrum_embedding.to_vec2::<f32>()?;
        for local in 0..chunk.len() {
            if pooled_rows[local].len() != EXPECTED_MODEL_DIM
                || spectrum_rows[local].len() != EXPECTED_MODEL_DIM
            {
                anyhow::bail!("interaction representation dimension mismatch");
            }
            let mut representation = Vec::with_capacity(INTERACTION_DIM);
            representation.extend_from_slice(&pooled_rows[local]);
            for j in 0..EXPECTED_MODEL_DIM {
                representation.push(pooled_rows[local][j] * spectrum_rows[local][j]);
            }
            group.rows[chunk[local]].interaction = Some(representation);
            encoded += 1;
        }
    }
    Ok(encoded)
}

fn precursor_context(
    record: &FoundationTrainingRecord,
    device: &Device,
) -> Result<PrecursorContextBatch> {
    let charge = record.context.charge.unwrap_or(0) as f32;
    let charge_present = if record.context.charge.is_some() {
        1.0f32
    } else {
        0.0
    };
    let precursor_mz = record.context.precursor_mz.unwrap_or(0.0);
    let precursor_mz_present = if record.context.precursor_mz.is_some() {
        1.0f32
    } else {
        0.0
    };
    Ok(PrecursorContextBatch {
        charge: Tensor::from_vec(vec![charge], 1, device)?,
        charge_present: Tensor::from_vec(vec![charge_present], 1, device)?,
        precursor_mz: Tensor::from_vec(vec![precursor_mz], 1, device)?,
        precursor_mz_present: Tensor::from_vec(vec![precursor_mz_present], 1, device)?,
        nce: Tensor::zeros(1, DType::F32, device)?,
        nce_present: Tensor::zeros(1, DType::F32, device)?,
        instrument_ids: Tensor::zeros(1, DType::U32, device)?,
        instrument_present: Tensor::zeros(1, DType::F32, device)?,
    })
}

fn exported_peptidoform(row: &CandidateRow) -> Result<PeptidoformInput> {
    if row.modifications.trim().is_empty() {
        return Ok(PeptidoformInput::unmodified(row.sequence.clone()));
    }

    let residues: Vec<char> = row.sequence.chars().collect();
    let mut nterm = Vec::<u32>::new();
    let mut residue_mods: HashMap<usize, Vec<u32>> = HashMap::new();
    for part in row
        .modifications
        .split(';')
        .filter(|part| !part.trim().is_empty())
    {
        let (identity, site) = part
            .split_once('@')
            .with_context(|| format!("invalid exported modification '{part}'"))?;
        let id: u32 = identity
            .strip_prefix("UniMod:")
            .with_context(|| format!("unsupported exported modification identity '{identity}'"))?
            .parse()?;
        if site == "NTerm" {
            nterm.push(id);
        } else if let Some(inner) = site
            .strip_prefix("Residue(")
            .and_then(|value| value.strip_suffix(')'))
        {
            let residue_index: usize = inner.parse()?;
            if residue_index >= residues.len() {
                anyhow::bail!(
                    "exported modification residue index {} exceeds sequence '{}'",
                    residue_index,
                    row.sequence
                );
            }
            residue_mods.entry(residue_index).or_default().push(id);
        } else {
            anyhow::bail!(
                "v0.15.0 cannot reconstruct unsupported exported modification site '{site}'"
            );
        }
    }
    nterm.sort_unstable();
    for values in residue_mods.values_mut() {
        values.sort_unstable();
    }

    let mut encoded = String::new();
    for id in nterm {
        encoded.push_str(&format!("[UniMod:{id}]"));
    }
    for (index, residue) in residues.into_iter().enumerate() {
        encoded.push(residue);
        if let Some(ids) = residue_mods.get(&index) {
            for id in ids {
                encoded.push_str(&format!("[UniMod:{id}]"));
            }
        }
    }
    parse_modified_peptide(&encoded)
        .with_context(|| format!("reconstruct exported candidate '{}'", row.sequence))
}

fn fit_normalization(groups: &[CandidateGroup]) -> Result<Normalization> {
    let mut mean = vec![0.0f64; INTERACTION_DIM];
    let mut n = 0.0f64;
    for group in groups {
        for &index in &group.training_indices {
            let representation = group.rows[index].interaction.as_ref().with_context(|| {
                format!(
                    "missing TRAIN interaction for record {}",
                    group.record_index
                )
            })?;
            n += 1.0;
            for j in 0..INTERACTION_DIM {
                mean[j] += representation[j] as f64;
            }
        }
    }
    if n == 0.0 {
        anyhow::bail!("no TRAIN interaction representations were encoded");
    }
    for value in &mut mean {
        *value /= n;
    }

    let mut var = vec![0.0f64; INTERACTION_DIM];
    for group in groups {
        for &index in &group.training_indices {
            let representation = group.rows[index]
                .interaction
                .as_ref()
                .context("missing TRAIN interaction during variance fit")?;
            for j in 0..INTERACTION_DIM {
                let delta = representation[j] as f64 - mean[j];
                var[j] += delta * delta;
            }
        }
    }
    let std = var
        .into_iter()
        .map(|value| (value / n).sqrt().max(1.0e-6))
        .collect();
    Ok(Normalization { mean, std })
}

fn accumulate_listwise_gradient<F>(
    group: &CandidateGroup,
    subset: &[usize],
    norm: &Normalization,
    model: &Model,
    positive: F,
    grad: &mut Gradients,
) -> Result<f64>
where
    F: Fn(&CandidateRow) -> bool,
{
    if subset.is_empty() {
        anyhow::bail!(
            "supervised group {} has empty hard-negative subset",
            group.record_index
        );
    }
    if !subset.iter().any(|&index| positive(&group.rows[index])) {
        anyhow::bail!(
            "supervised group {} hard-negative subset does not contain a positive",
            group.record_index
        );
    }

    let mut scores = Vec::with_capacity(subset.len());
    let mut hidden = Vec::with_capacity(subset.len());
    for &index in subset {
        let row = &group.rows[index];
        let representation = row.interaction.as_ref().with_context(|| {
            format!(
                "missing TRAIN interaction for record {}",
                group.record_index
            )
        })?;
        let (residual, h) = model.residual_score(representation, norm);
        scores.push(row.legacy_score + residual);
        hidden.push(h);
    }

    let max_score = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exp_scores: Vec<f64> = scores
        .iter()
        .map(|&score| (score - max_score).exp())
        .collect();
    let denom: f64 = exp_scores.iter().sum();
    let positive_mass: f64 = subset
        .iter()
        .enumerate()
        .filter_map(|(local, &index)| positive(&group.rows[index]).then_some(exp_scores[local]))
        .sum();
    let probability = (positive_mass / denom.max(f64::MIN_POSITIVE)).max(f64::MIN_POSITIVE);
    let loss = -probability.ln();
    let positive_denom = positive_mass.max(f64::MIN_POSITIVE);

    for (local, &index) in subset.iter().enumerate() {
        let row = &group.rows[index];
        let representation = row.interaction.as_ref().context("missing interaction")?;
        let p_all = exp_scores[local] / denom.max(f64::MIN_POSITIVE);
        let p_positive = if positive(row) {
            exp_scores[local] / positive_denom
        } else {
            0.0
        };
        let dscore = p_all - p_positive;
        grad.b2 += dscore;
        for h in 0..HIDDEN {
            grad.w2[h] += dscore * hidden[local][h];
            let dh = dscore * model.w2[h] * (1.0 - hidden[local][h] * hidden[local][h]);
            grad.b1[h] += dh;
            let base = h * INTERACTION_DIM;
            for j in 0..INTERACTION_DIM {
                let x = (representation[j] as f64 - norm.mean[j]) / norm.std[j];
                grad.w1[base + j] += dh * x;
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
        exact_in_interaction_window: 0,
        il_in_interaction_window: 0,
    };
    for group in groups {
        metrics.oracle_exact += usize::from(group.rows.iter().any(|row| row.exact));
        metrics.oracle_il += usize::from(group.rows.iter().any(|row| row.il_exact));
        metrics.exact_in_interaction_window += usize::from(
            group
                .rows
                .iter()
                .any(|row| row.exact && row.interaction.is_some()),
        );
        metrics.il_in_interaction_window += usize::from(
            group
                .rows
                .iter()
                .any(|row| row.il_exact && row.interaction.is_some()),
        );

        if let Some(index) = best_interaction_index(group, norm, model) {
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

fn best_interaction_index(
    group: &CandidateGroup,
    norm: &Normalization,
    model: &Model,
) -> Option<usize> {
    let mut best: Option<(f64, usize, usize)> = None;
    for (index, row) in group.rows.iter().enumerate() {
        let residual = row
            .interaction
            .as_ref()
            .map(|representation| model.residual_score(representation, norm).0)
            .unwrap_or(0.0);
        let score = row.legacy_score + residual;
        match best {
            None => best = Some((score, row.legacy_rank, index)),
            Some((best_score, best_legacy_rank, _)) => {
                if score > best_score || (score == best_score && row.legacy_rank < best_legacy_rank)
                {
                    best = Some((score, row.legacy_rank, index));
                }
            }
        }
    }
    best.map(|(_, _, index)| index)
}

fn interaction_order(group: &CandidateGroup, norm: &Normalization, model: &Model) -> Vec<usize> {
    let mut scored: Vec<(usize, f64, usize)> = group
        .rows
        .iter()
        .enumerate()
        .map(|(index, row)| {
            let residual = row
                .interaction
                .as_ref()
                .map(|representation| model.residual_score(representation, norm).0)
                .unwrap_or(0.0);
            (index, row.legacy_score + residual, row.legacy_rank)
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
        "record_index\toracle_literal\toracle_il\tliteral_in_interaction_window\til_in_interaction_window\tlegacy_literal_rank\tlegacy_il_rank\tinteraction_literal_rank\tinteraction_il_rank\tlegacy_top1_literal\tlegacy_top1_il\tinteraction_top1_literal\tinteraction_top1_il"
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
        let literal_in_window = group
            .rows
            .iter()
            .any(|row| row.exact && row.interaction.is_some());
        let il_in_window = group
            .rows
            .iter()
            .any(|row| row.il_exact && row.interaction.is_some());
        let order = interaction_order(group, norm, model);
        let mut interaction_literal_rank = None;
        let mut interaction_il_rank = None;
        for (rank0, &index) in order.iter().enumerate() {
            let rank = rank0 + 1;
            if interaction_literal_rank.is_none() && group.rows[index].exact {
                interaction_literal_rank = Some(rank);
            }
            if interaction_il_rank.is_none() && group.rows[index].il_exact {
                interaction_il_rank = Some(rank);
            }
            if interaction_literal_rank.is_some() && interaction_il_rank.is_some() {
                break;
            }
        }
        let legacy_top = group.rows.iter().min_by_key(|row| row.legacy_rank);
        let interaction_top = order.first().map(|&index| &group.rows[index]);
        writeln!(
            out,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            group.record_index,
            yes_no(legacy_literal_rank.is_some()),
            yes_no(legacy_il_rank.is_some()),
            yes_no(literal_in_window),
            yes_no(il_in_window),
            format_rank(legacy_literal_rank),
            format_rank(legacy_il_rank),
            format_rank(interaction_literal_rank),
            format_rank(interaction_il_rank),
            yes_no(legacy_top.map(|row| row.exact).unwrap_or(false)),
            yes_no(legacy_top.map(|row| row.il_exact).unwrap_or(false)),
            yes_no(interaction_top.map(|row| row.exact).unwrap_or(false)),
            yes_no(interaction_top.map(|row| row.il_exact).unwrap_or(false)),
        )?;
    }
    Ok(())
}

fn load_matching_variables(varmap: &VarMap, checkpoint: &Path, device: &Device) -> Result<()> {
    let tensors = candle_core::safetensors::load(checkpoint, device)?;
    let data = varmap
        .data()
        .lock()
        .map_err(|_| anyhow::anyhow!("v0.15.0 causal VarMap lock poisoned"))?;
    let mut missing = Vec::new();
    for (name, variable) in data.iter() {
        match tensors.get(name) {
            Some(tensor) => {
                if tensor.dims() != variable.as_tensor().dims() {
                    anyhow::bail!(
                        "shape mismatch for '{name}': checkpoint {:?}, model {:?}",
                        tensor.dims(),
                        variable.as_tensor().dims()
                    );
                }
                variable.set(tensor)?;
            }
            None => missing.push(name.clone()),
        }
    }
    drop(data);
    if !missing.is_empty() {
        anyhow::bail!(
            "unified checkpoint is missing causal variables: {}",
            missing.join(", ")
        );
    }
    Ok(())
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
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    fn next_f64(&mut self) -> f64 {
        let value = self.next_u64() >> 11;
        value as f64 * (1.0 / ((1u64 << 53) as f64))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(rank: usize, exact: bool, il_exact: bool) -> CandidateRow {
        CandidateRow {
            sequence: "PEPTIDE".into(),
            modifications: String::new(),
            exact,
            il_exact,
            legacy_score: 100.0 - rank as f64,
            legacy_rank: rank,
            interaction: None,
        }
    }

    #[test]
    fn hard_window_always_keeps_supervised_target() {
        let mut rows: Vec<CandidateRow> = (1..=40)
            .map(|rank| row(rank, rank == 40, rank == 40))
            .collect();
        rows[39].legacy_rank = 40;
        let group = CandidateGroup {
            record_index: 7,
            rows,
            training_indices: Vec::new(),
        };
        let selected = fixed_training_indices(&group);
        assert_eq!(selected.len(), TRAIN_HARD_WINDOW + 1);
        assert!(selected.contains(&39));
    }

    #[test]
    fn zero_residual_head_is_exactly_zero() {
        let model = Model::new_zero_residual(SEED);
        let norm = Normalization {
            mean: vec![0.0; INTERACTION_DIM],
            std: vec![1.0; INTERACTION_DIM],
        };
        let raw = vec![0.5f32; INTERACTION_DIM];
        assert_eq!(model.residual_score(&raw, &norm).0, 0.0);
    }

    #[test]
    fn exported_modifications_round_trip_supported_unimod_sites() {
        let row = CandidateRow {
            sequence: "ACDM".into(),
            modifications: "UniMod:1@NTerm;UniMod:4@Residue(1);UniMod:35@Residue(3)".into(),
            exact: false,
            il_exact: false,
            legacy_score: 0.0,
            legacy_rank: 1,
            interaction: None,
        };
        let peptide = exported_peptidoform(&row).unwrap();
        assert_eq!(peptide.sequence, "ACDM");
        assert_eq!(peptide.modifications.len(), 3);
        assert!(peptide.modifications.iter().any(|m| m.unimod_id == Some(1)));
        assert!(peptide.modifications.iter().any(|m| m.unimod_id == Some(4)));
        assert!(peptide
            .modifications
            .iter()
            .any(|m| m.unimod_id == Some(35)));
    }
}
