//! Train the bounded v0.16.0 trainable spectrum-candidate interaction adapter.
//!
//! The accepted v0.13.23 candidate proposal system and final legacy score remain frozen.
//! Unlike v0.15.0, the spectrum-candidate compatibility transform itself is trainable:
//! frozen causal token states query frozen spectrum memory through one new cross-attention
//! layer followed by a narrow feed-forward adapter.  A zero-initialized scalar residual head
//! preserves exact legacy ranking before the first update.
//!
//! No candidate generation occurs here. TRAIN and VALIDATION candidate TSVs are verified
//! against the benchmark manifest and TEST records are rejected before model execution.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{ops, VarBuilder, VarMap};
use redeem_properties::foundation::{
    load_foundation_corpus, parse_modified_peptide, read_foundation_training_run_config,
    FoundationAdamW, FoundationAdamWConfig, FoundationBenchmarkManifest, FoundationCausalCollator,
    FoundationDiffusionConfig, FoundationPartition, FoundationSpectrum,
    FoundationSpectrumCandidateInteractionAdapter, FoundationSpectrumCollator,
    FoundationTrainingRecord, PeptideSpectrumCausalModel, PeptidoformInput, PrecursorContextBatch,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::env;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

const VERSION: &str = "v0.16.0";
const ADAPTER_BOTTLENECK: usize = 48;
const TRAIN_HARD_WINDOW: usize = 32;
const VALIDATION_INTERACTION_WINDOW: usize = 128;
const ENCODE_BATCH: usize = 32;
const EPOCHS: usize = 10;
const BATCH_GROUPS: usize = 16;
const LEARNING_RATE: f64 = 2.0e-4;
const WEIGHT_DECAY: f64 = 1.0e-4;
const MAX_GRADIENT_NORM: f64 = 5.0;
const SEED: u64 = 20_260_916;

const REQUIRED_LITERAL_TOP1: usize = 28;
const REQUIRED_IL_TOP1: usize = 42;
const REQUIRED_LITERAL_ORACLE: usize = 44;
const REQUIRED_IL_ORACLE: usize = 54;

#[derive(Debug, Deserialize)]
struct UnifiedCheckpointMetadata {
    inverse_config: FoundationDiffusionConfig,
}

#[derive(Debug, Clone)]
struct FrozenCandidateState {
    token_len: usize,
    hidden: Vec<f32>,
}

#[derive(Debug, Clone)]
struct FrozenSpectrumState {
    memory_len: usize,
    memory: Vec<f32>,
    mask: Vec<f32>,
}

#[derive(Debug, Clone)]
struct CandidateRow {
    sequence: String,
    modifications: String,
    exact: bool,
    il_exact: bool,
    legacy_score: f64,
    legacy_rank: usize,
    frozen: Option<FrozenCandidateState>,
}

#[derive(Debug, Clone)]
struct CandidateGroup {
    record_index: usize,
    rows: Vec<CandidateRow>,
    training_indices: Vec<usize>,
    spectrum: Option<FrozenSpectrumState>,
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

#[derive(Debug, Serialize)]
struct AdapterMetadata {
    version: String,
    objective: String,
    architecture: String,
    proposal_policy: String,
    base_score: String,
    representation: String,
    frozen_backbone: String,
    supervision_policy: String,
    validation_selection_policy: String,
    test_partition_consumed: bool,
    seed: u64,
    model_dim: usize,
    attention_heads: usize,
    adapter_bottleneck: usize,
    train_hard_window: usize,
    validation_interaction_window: usize,
    encode_batch: usize,
    epochs: usize,
    batch_groups: usize,
    learning_rate: f64,
    weight_decay: f64,
    max_gradient_norm: f64,
    training_yaml: String,
    unified_checkpoint: String,
    train_candidate_tsv: String,
    validation_candidate_tsv: String,
    train_groups: usize,
    train_il_supervised_groups: usize,
    train_exact_supervised_groups: usize,
    validation_groups: usize,
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 6 {
        anyhow::bail!(
            "usage: foundation_train_trainable_interaction_reranker RUN.yaml UNIFIED_CHECKPOINT TRAIN_CANDIDATES.tsv VALIDATION_CANDIDATES.tsv OUTPUT_DIR"
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
    if config.model_dim != 96 {
        anyhow::bail!(
            "v0.16.0 is frozen to the accepted 96-d unified checkpoint; checkpoint has model_dim {}",
            config.model_dim
        );
    }

    // The deployment image is built with Candle CUDA support. Use CUDA when that
    // feature is present (the Slurm path supplies the GPU through Singularity --nv),
    // while retaining CPU behavior for ordinary non-CUDA local builds.
    let device = Device::cuda_if_available(0)?;
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
    println!(
        "compute_device\t{}",
        if device.is_cuda() { "cuda:0" } else { "cpu" }
    );
    println!("objective\thierarchical_il_then_exact_listwise_unit_weight_fixed_hard_negatives");
    println!("architecture\tfrozen_causal_states_plus_trainable_cross_attention_adapter_96x48x96_zero_residual");
    println!("proposal_policy\tv01323_final_two_view_fixed_budget_frozen");
    println!("base_score\tfragment_score+0.1*n_to_c_ar_total_log_probability_frozen");
    println!("representation\tfrozen_shifted_prefix_causal_hidden_queries_frozen_spectrum_memory_through_trainable_cross_attention");
    println!("trainable_parameters\tinteraction_query_norm+cross_attention+ff_adapter+output_norm+zero_initialized_residual_head");
    println!("frozen_parameters\tall_unified_foundation_and_causal_backbone_parameters");
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

    let encoded_train = precompute_training_states(
        &mut train_groups,
        &corpus.records,
        &causal_model,
        &causal_collator,
        &spectrum_collator,
        config.model_dim,
        &device,
    )?;
    let encoded_validation = precompute_validation_states(
        &mut validation_groups,
        &corpus.records,
        &causal_model,
        &causal_collator,
        &spectrum_collator,
        config.model_dim,
        &device,
    )?;
    println!("train_interaction_candidates_encoded\t{encoded_train}");
    println!("validation_interaction_candidates_encoded\t{encoded_validation}");

    let adapter_varmap = VarMap::new();
    let adapter_vb = VarBuilder::from_varmap(&adapter_varmap, DType::F32, &device);
    let adapter = FoundationSpectrumCandidateInteractionAdapter::new(
        config.model_dim,
        config.num_attention_heads,
        ADAPTER_BOTTLENECK,
        adapter_vb.pp("interaction_adapter"),
    )?;

    let initial = evaluate(&validation_groups, &adapter, config.model_dim, &device)?;
    let parity = initial.top1_exact == initial.legacy_top1_exact
        && initial.top1_il == initial.legacy_top1_il;
    if !parity {
        anyhow::bail!(
            "zero-residual v0.16.0 initialization failed legacy parity: residual={}/{} legacy={}/{}",
            initial.top1_exact,
            initial.top1_il,
            initial.legacy_top1_exact,
            initial.legacy_top1_il
        );
    }
    if initial.oracle_exact != REQUIRED_LITERAL_ORACLE || initial.oracle_il != REQUIRED_IL_ORACLE {
        anyhow::bail!(
            "v0.16.0 validation oracle parity failed: expected {}/{} observed {}/{}",
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
    println!("max_gradient_norm\t{MAX_GRADIENT_NORM}");
    println!("seed\t{SEED}");

    let mut optimizer = FoundationAdamW::new(
        &adapter_varmap,
        FoundationAdamWConfig {
            learning_rate: LEARNING_RATE,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1.0e-8,
            weight_decay: WEIGHT_DECAY,
        },
    )?;

    let supervised: Vec<usize> = train_groups
        .iter()
        .enumerate()
        .filter_map(|(index, group)| group.rows.iter().any(|row| row.il_exact).then_some(index))
        .collect();
    let mut order = supervised;

    for epoch in 0..EPOCHS {
        deterministic_shuffle(&mut order, SEED ^ (epoch as u64 + 1));
        let mut epoch_loss = 0.0f64;
        let mut epoch_terms = 0usize;
        let mut epoch_il_terms = 0usize;
        let mut epoch_exact_terms = 0usize;
        let mut gradient_norm_sum = 0.0f64;
        let mut optimizer_steps = 0usize;

        for chunk in order.chunks(BATCH_GROUPS) {
            let mut losses = Vec::<Tensor>::new();
            for &group_index in chunk {
                let group = &train_groups[group_index];
                let scores = score_subset(
                    group,
                    &group.training_indices,
                    &adapter,
                    config.model_dim,
                    &device,
                )?;
                let il_mask = group
                    .training_indices
                    .iter()
                    .map(|&index| group.rows[index].il_exact)
                    .collect::<Vec<_>>();
                let il_loss = listwise_positive_set_loss(&scores, &il_mask, &device)?;
                epoch_loss += f64::from(il_loss.to_scalar::<f32>()?);
                epoch_terms += 1;
                epoch_il_terms += 1;
                losses.push(il_loss);

                if group.rows.iter().any(|row| row.exact) {
                    let exact_mask = group
                        .training_indices
                        .iter()
                        .map(|&index| group.rows[index].exact)
                        .collect::<Vec<_>>();
                    let exact_loss = listwise_positive_set_loss(&scores, &exact_mask, &device)?;
                    epoch_loss += f64::from(exact_loss.to_scalar::<f32>()?);
                    epoch_terms += 1;
                    epoch_exact_terms += 1;
                    losses.push(exact_loss);
                }
            }
            let batch_loss = Tensor::stack(&losses, 0)?.mean_all()?;
            let step = optimizer.backward_step(&batch_loss, Some(MAX_GRADIENT_NORM))?;
            gradient_norm_sum += step.gradient_norm;
            optimizer_steps += 1;
        }

        println!(
            "training_epoch\tepoch={}\tmean_hierarchical_loss={:.8}\tobjective_terms={}\til_terms={}\texact_terms={}\toptimizer_steps={}\tmean_gradient_norm={:.6}",
            epoch + 1,
            epoch_loss / epoch_terms.max(1) as f64,
            epoch_terms,
            epoch_il_terms,
            epoch_exact_terms,
            optimizer_steps,
            gradient_norm_sum / optimizer_steps.max(1) as f64,
        );
    }

    let metrics = evaluate(&validation_groups, &adapter, config.model_dim, &device)?;
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
        metrics.il_in_interaction_window,
    );
    println!(
        "v0160_acceptance_gate\trequired_literal={}\trequired_il={}\trequired_oracle_literal={}\trequired_oracle_il={}\tobserved_literal={}\tobserved_il={}\tobserved_oracle_literal={}\tobserved_oracle_il={}\tgate={}",
        REQUIRED_LITERAL_TOP1,
        REQUIRED_IL_TOP1,
        REQUIRED_LITERAL_ORACLE,
        REQUIRED_IL_ORACLE,
        metrics.top1_exact,
        metrics.top1_il,
        metrics.oracle_exact,
        metrics.oracle_il,
        if gate { "PASS" } else { "FAIL" },
    );
    println!(
        "v0160_stop_rule\t{}",
        if gate {
            "ACCEPT_TRAINABLE_INTERACTION_ADAPTER_AND_FREEZE"
        } else {
            "REJECT_SHALLOW_ADAPTER_AND_REASSESS_FOUNDATION_REPRESENTATION_PRETRAINING"
        }
    );

    fs::create_dir_all(&output_dir)?;
    let model_path = output_dir.join("interaction_adapter.safetensors");
    adapter_varmap.save(&model_path)?;
    let metadata_out = AdapterMetadata {
        version: VERSION.to_string(),
        objective: "hierarchical_il_then_exact_listwise_unit_weight_fixed_hard_negatives".into(),
        architecture:
            "frozen_causal_states_plus_trainable_cross_attention_adapter_96x48x96_zero_residual"
                .into(),
        proposal_policy: "v01323_final_two_view_fixed_budget_frozen".into(),
        base_score: "fragment_score+0.1*n_to_c_ar_total_log_probability_frozen".into(),
        representation:
            "frozen_causal_hidden_queries_frozen_spectrum_memory_through_trainable_cross_attention"
                .into(),
        frozen_backbone: "unified_v01310+accepted_v01323_proposals".into(),
        supervision_policy: "il_equivalent_all_groups_plus_literal_exact_when_available".into(),
        validation_selection_policy: "none_fixed_epoch10_evaluation_only".into(),
        test_partition_consumed: false,
        seed: SEED,
        model_dim: config.model_dim,
        attention_heads: config.num_attention_heads,
        adapter_bottleneck: ADAPTER_BOTTLENECK,
        train_hard_window: TRAIN_HARD_WINDOW,
        validation_interaction_window: VALIDATION_INTERACTION_WINDOW,
        encode_batch: ENCODE_BATCH,
        epochs: EPOCHS,
        batch_groups: BATCH_GROUPS,
        learning_rate: LEARNING_RATE,
        weight_decay: WEIGHT_DECAY,
        max_gradient_norm: MAX_GRADIENT_NORM,
        training_yaml: training_yaml.display().to_string(),
        unified_checkpoint: unified_checkpoint.display().to_string(),
        train_candidate_tsv: train_path.display().to_string(),
        validation_candidate_tsv: validation_path.display().to_string(),
        train_groups: train_groups.len(),
        train_il_supervised_groups: train_il_supervised,
        train_exact_supervised_groups: train_exact_supervised,
        validation_groups: validation_groups.len(),
    };
    serde_yaml::to_writer(
        BufWriter::new(File::create(output_dir.join("metadata.yaml"))?),
        &metadata_out,
    )?;

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
        if gate { "PASS" } else { "FAIL" },
    )?;
    summary.flush()?;

    let diagnostics_path = output_dir.join("validation_ranking_diagnostics.tsv");
    write_validation_diagnostics(
        &diagnostics_path,
        &validation_groups,
        &adapter,
        config.model_dim,
        &device,
    )?;

    println!("checkpoint\t{}", model_path.display());
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
        anyhow::bail!("v0.16.0 forbids TEST-partition inputs; suspicious path {path:?}");
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
                "v0.16.0 {label} candidate group {} is not assigned to expected benchmark partition {label}",
                group.record_index
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
                    spectrum: None,
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
            frozen: None,
        });
    }
    if let Some(id) = current_id {
        groups.push(CandidateGroup {
            record_index: id,
            rows: current_rows,
            training_indices: Vec::new(),
            spectrum: None,
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

fn precompute_training_states(
    groups: &mut [CandidateGroup],
    records: &[FoundationTrainingRecord],
    model: &PeptideSpectrumCausalModel,
    causal_collator: &FoundationCausalCollator,
    spectrum_collator: &FoundationSpectrumCollator,
    model_dim: usize,
    device: &Device,
) -> Result<usize> {
    let total = groups
        .iter()
        .filter(|group| !group.training_indices.is_empty())
        .count();
    let mut done = 0usize;
    let mut encoded = 0usize;
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
            model_dim,
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

fn precompute_validation_states(
    groups: &mut [CandidateGroup],
    records: &[FoundationTrainingRecord],
    model: &PeptideSpectrumCausalModel,
    causal_collator: &FoundationCausalCollator,
    spectrum_collator: &FoundationSpectrumCollator,
    model_dim: usize,
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
            model_dim,
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

#[allow(clippy::too_many_arguments)]
fn encode_group_indices(
    group: &mut CandidateGroup,
    indices: &[usize],
    record: &FoundationTrainingRecord,
    model: &PeptideSpectrumCausalModel,
    causal_collator: &FoundationCausalCollator,
    spectrum_collator: &FoundationSpectrumCollator,
    model_dim: usize,
    device: &Device,
) -> Result<usize> {
    if indices.is_empty() {
        return Ok(0);
    }
    let spectrum = FoundationSpectrum::from_training_record(record).with_context(|| {
        format!(
            "record {} has no observed spectrum for v0.16.0 interaction encoding",
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
        let (batch, token_len, hidden_dim) = output.decoder_hidden.dims3()?;
        if batch != chunk.len() || hidden_dim != model_dim {
            anyhow::bail!(
                "frozen interaction hidden shape mismatch: batch={} expected={} dim={} expected_dim={}",
                batch,
                chunk.len(),
                hidden_dim,
                model_dim
            );
        }
        let hidden = output.decoder_hidden.to_vec3::<f32>()?;
        let token_masks = causal.input.token_mask.to_vec2::<f32>()?;
        for local in 0..batch {
            let active = token_masks[local]
                .iter()
                .take(token_len)
                .take_while(|&&value| value > 0.5)
                .count();
            if active == 0 {
                anyhow::bail!("candidate has no active causal positions");
            }
            let mut flattened = Vec::with_capacity(active * model_dim);
            for position in 0..active {
                flattened.extend_from_slice(&hidden[local][position]);
            }
            group.rows[chunk[local]].frozen = Some(FrozenCandidateState {
                token_len: active,
                hidden: flattened,
            });
            encoded += 1;
        }

        if group.spectrum.is_none() {
            let memory = output.spectrum_memory.to_vec3::<f32>()?;
            let memory_mask = output.spectrum_memory_mask.to_vec2::<f32>()?;
            let memory_len = memory_mask[0].len();
            let mut flattened = Vec::with_capacity(memory_len * model_dim);
            for position in 0..memory_len {
                flattened.extend_from_slice(&memory[0][position]);
            }
            group.spectrum = Some(FrozenSpectrumState {
                memory_len,
                memory: flattened,
                mask: memory_mask[0].clone(),
            });
        }
    }
    Ok(encoded)
}

fn precursor_context(
    record: &FoundationTrainingRecord,
    device: &Device,
) -> Result<PrecursorContextBatch> {
    let charge = record.context.charge.unwrap_or(0) as f32;
    // Keep every continuous precursor-context tensor in F32. Without an explicit
    // type here, Rust defaults these standalone floating literals to f64, which
    // produces F64 presence masks and fails inside the frozen causal context path
    // when they are multiplied by the F32 precursor features.
    let charge_present: f32 = if record.context.charge.is_some() {
        1.0
    } else {
        0.0
    };
    let precursor_mz: f32 = record.context.precursor_mz.unwrap_or(0.0);
    let precursor_mz_present: f32 = if record.context.precursor_mz.is_some() {
        1.0
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
                "v0.16.0 cannot reconstruct unsupported exported modification site '{site}'"
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

fn score_subset(
    group: &CandidateGroup,
    subset: &[usize],
    adapter: &FoundationSpectrumCandidateInteractionAdapter,
    model_dim: usize,
    device: &Device,
) -> Result<Tensor> {
    if subset.is_empty() {
        anyhow::bail!("cannot score empty candidate subset");
    }
    let spectrum = group.spectrum.as_ref().with_context(|| {
        format!(
            "record {} missing frozen spectrum state",
            group.record_index
        )
    })?;
    let mut max_tokens = 0usize;
    for &index in subset {
        let frozen = group.rows[index].frozen.as_ref().with_context(|| {
            format!(
                "record {} candidate {} lacks frozen state",
                group.record_index, index
            )
        })?;
        max_tokens = max_tokens.max(frozen.token_len);
    }
    let batch = subset.len();
    let mut hidden = vec![0.0f32; batch * max_tokens * model_dim];
    let mut mask = vec![0.0f32; batch * max_tokens];
    let mut legacy = Vec::with_capacity(batch);
    for (local, &index) in subset.iter().enumerate() {
        let row = &group.rows[index];
        let frozen = row.frozen.as_ref().context("missing frozen candidate")?;
        for position in 0..frozen.token_len {
            let src = position * model_dim;
            let dst = (local * max_tokens + position) * model_dim;
            hidden[dst..dst + model_dim].copy_from_slice(&frozen.hidden[src..src + model_dim]);
            mask[local * max_tokens + position] = 1.0;
        }
        legacy.push(row.legacy_score as f32);
    }
    let hidden = Tensor::from_vec(hidden, (batch, max_tokens, model_dim), device)?;
    let mask = Tensor::from_vec(mask, (batch, max_tokens), device)?;
    let memory = Tensor::from_vec(
        spectrum.memory.clone(),
        (1, spectrum.memory_len, model_dim),
        device,
    )?
    .broadcast_as((batch, spectrum.memory_len, model_dim))?;
    let memory_mask = Tensor::from_vec(spectrum.mask.clone(), (1, spectrum.memory_len), device)?
        .broadcast_as((batch, spectrum.memory_len))?;
    let residual = adapter.forward(&hidden, &mask, &memory, &memory_mask)?;
    let legacy = Tensor::from_vec(legacy, batch, device)?;
    Ok((legacy + residual)?)
}

fn listwise_positive_set_loss(
    scores: &Tensor,
    positive: &[bool],
    device: &Device,
) -> Result<Tensor> {
    let n = scores.dims1()?;
    if positive.len() != n || !positive.iter().any(|&value| value) {
        anyhow::bail!("listwise loss requires a positive mask aligned to scores");
    }
    let log_prob = ops::log_softmax(scores, 0)?;
    let prob = log_prob.exp()?;
    let mask_values = positive
        .iter()
        .map(|&value| if value { 1.0f32 } else { 0.0f32 })
        .collect::<Vec<_>>();
    let positive_mass = prob
        .broadcast_mul(&Tensor::from_vec(mask_values, n, device)?)?
        .sum_all()?
        .clamp(1.0e-12, 1.0)?;
    Ok(positive_mass.log()?.affine(-1.0, 0.0)?)
}

fn evaluate(
    groups: &[CandidateGroup],
    adapter: &FoundationSpectrumCandidateInteractionAdapter,
    model_dim: usize,
    device: &Device,
) -> Result<EvalMetrics> {
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
                .any(|row| row.exact && row.frozen.is_some()),
        );
        metrics.il_in_interaction_window += usize::from(
            group
                .rows
                .iter()
                .any(|row| row.il_exact && row.frozen.is_some()),
        );
        let order = interaction_order(group, adapter, model_dim, device)?;
        if let Some(&index) = order.first() {
            metrics.top1_exact += usize::from(group.rows[index].exact);
            metrics.top1_il += usize::from(group.rows[index].il_exact);
        }
        if let Some(row) = group.rows.iter().min_by_key(|row| row.legacy_rank) {
            metrics.legacy_top1_exact += usize::from(row.exact);
            metrics.legacy_top1_il += usize::from(row.il_exact);
        }
    }
    Ok(metrics)
}

fn interaction_order(
    group: &CandidateGroup,
    adapter: &FoundationSpectrumCandidateInteractionAdapter,
    model_dim: usize,
    device: &Device,
) -> Result<Vec<usize>> {
    let encoded = group
        .rows
        .iter()
        .enumerate()
        .filter_map(|(index, row)| row.frozen.is_some().then_some(index))
        .collect::<Vec<_>>();
    let mut residual_by_index = HashMap::<usize, f64>::new();
    if !encoded.is_empty() {
        let scores = score_subset(group, &encoded, adapter, model_dim, device)?.to_vec1::<f32>()?;
        for (local, &index) in encoded.iter().enumerate() {
            residual_by_index.insert(
                index,
                f64::from(scores[local]) - group.rows[index].legacy_score,
            );
        }
    }
    let mut scored = group
        .rows
        .iter()
        .enumerate()
        .map(|(index, row)| {
            let score = row.legacy_score + residual_by_index.get(&index).copied().unwrap_or(0.0);
            (index, score, row.legacy_rank)
        })
        .collect::<Vec<_>>();
    scored.sort_by(|a, b| {
        b.1.total_cmp(&a.1)
            .then_with(|| a.2.cmp(&b.2))
            .then_with(|| a.0.cmp(&b.0))
    });
    Ok(scored.into_iter().map(|(index, _, _)| index).collect())
}

fn write_validation_diagnostics(
    path: &Path,
    groups: &[CandidateGroup],
    adapter: &FoundationSpectrumCandidateInteractionAdapter,
    model_dim: usize,
    device: &Device,
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
            .any(|row| row.exact && row.frozen.is_some());
        let il_in_window = group
            .rows
            .iter()
            .any(|row| row.il_exact && row.frozen.is_some());
        let order = interaction_order(group, adapter, model_dim, device)?;
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
        .map_err(|_| anyhow::anyhow!("foundation VarMap lock poisoned"))?;
    let mut loaded = 0usize;
    for (name, variable) in data.iter() {
        let Some(value) = tensors.get(name) else {
            continue;
        };
        if variable.shape() != value.shape() {
            continue;
        }
        variable.set(value)?;
        loaded += 1;
    }
    if loaded == 0 {
        anyhow::bail!("no causal variables matched unified checkpoint {checkpoint:?}");
    }
    Ok(())
}

fn parse_bool(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "y"
    )
}

fn parse_finite(value: &str, fallback: f64) -> f64 {
    value
        .parse::<f64>()
        .ok()
        .filter(|value| value.is_finite())
        .unwrap_or(fallback)
}

fn parse_usize(value: &str, fallback: usize) -> usize {
    value.parse::<usize>().unwrap_or(fallback)
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

#[derive(Debug, Clone, Copy)]
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_window_keeps_supervised_positive() {
        let mut rows = (0..40)
            .map(|index| CandidateRow {
                sequence: format!("PEPTIDE{index}"),
                modifications: String::new(),
                exact: false,
                il_exact: false,
                legacy_score: -(index as f64),
                legacy_rank: index + 1,
                frozen: None,
            })
            .collect::<Vec<_>>();
        rows[39].il_exact = true;
        let group = CandidateGroup {
            record_index: 1,
            rows,
            training_indices: Vec::new(),
            spectrum: None,
        };
        let selected = fixed_training_indices(&group);
        assert!(selected.contains(&39));
        assert_eq!(selected.len(), 33);
    }

    #[test]
    fn modification_reconstruction_accepts_supported_export_syntax() -> Result<()> {
        let row = CandidateRow {
            sequence: "ACDMK".into(),
            modifications: "UniMod:1@NTerm;UniMod:4@Residue(1);UniMod:35@Residue(3)".into(),
            exact: false,
            il_exact: false,
            legacy_score: 0.0,
            legacy_rank: 1,
            frozen: None,
        };
        let peptide = exported_peptidoform(&row)?;
        assert_eq!(peptide.sequence, "ACDMK");
        Ok(())
    }
}
