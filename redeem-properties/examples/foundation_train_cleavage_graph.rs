//! Train the isolated v0.13.16 spectrum-conditioned cleavage-mass graph edge scorer.
//!
//! The accepted v0.13.10 unified model and v0.13.13 reverse-causal checkpoint
//! are read only for provenance/compatibility checks. The optimizer owns only
//! `cleavage_graph.*` variables.

use anyhow::{Context, Result};
use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use redeem_properties::foundation::{
    foundation_build_cleavage_graph, foundation_cleavage_graph_outgoing_edge_loss,
    foundation_cleavage_graph_training_batch, foundation_cleavage_graph_true_path_audit,
    foundation_diffusion_dataset_fingerprint, load_foundation_corpus,
    read_foundation_training_run_config, validate_cleavage_graph_namespace, FoundationAdamW,
    FoundationAdamWConfig, FoundationBenchmarkManifest, FoundationDiffusionConfig,
    FoundationDiffusionVocabulary, FoundationPartition, FoundationSpectrum,
    FoundationTrainingRecord, PeptideSpectrumCleavageGraphScorer,
    FOUNDATION_CLEAVAGE_GRAPH_HIDDEN_DIM_V01316, FOUNDATION_CLEAVAGE_GRAPH_K_BEST_PATHS_V01316,
    FOUNDATION_CLEAVAGE_GRAPH_MASS_TOLERANCE_DA_V01316,
    FOUNDATION_CLEAVAGE_GRAPH_MAX_FRAGMENT_CHARGE_V01316,
    FOUNDATION_CLEAVAGE_GRAPH_MAX_OUTGOING_EDGES_V01316,
    FOUNDATION_CLEAVAGE_GRAPH_NAMESPACE_V01316, FOUNDATION_CLEAVAGE_GRAPH_OBJECTIVE_V01316,
    FOUNDATION_REVERSE_CAUSAL_DIRECTION_V01313, FOUNDATION_REVERSE_CAUSAL_NAMESPACE_V01313,
};
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const TRAIN_STEPS: usize = 1_000;
const BATCH_SIZE: usize = 8;
const VALIDATION_RECORDS: usize = 128;
const VALIDATION_BATCHES: usize = VALIDATION_RECORDS / BATCH_SIZE;
const SEED: u64 = 20_260_912;
const LEARNING_RATE: f64 = 1.0e-5;
const MAX_GRADIENT_NORM: f64 = 1.0;
const VALIDATION_INTERVAL: usize = 100;

#[derive(Debug, Deserialize)]
struct UnifiedParentMetadata {
    corpus_fingerprint: String,
    benchmark_manifest_fingerprint: String,
    completed_steps: usize,
    inverse_config: FoundationDiffusionConfig,
}

#[derive(Debug, Deserialize)]
struct ReverseCausalCheckpointMetadata {
    objective: String,
    direction: String,
    parameter_namespace: String,
    corpus_fingerprint: String,
    benchmark_manifest_fingerprint: String,
    parent_unified_checkpoint: String,
    inverse_config: FoundationDiffusionConfig,
}

#[derive(Debug, Clone, Serialize)]
struct CleavageGraphCheckpointMetadata {
    version: u32,
    objective: String,
    parameter_namespace: String,
    corpus_fingerprint: String,
    benchmark_manifest_fingerprint: String,
    train_dataset_fingerprint: String,
    validation_dataset_fingerprint: String,
    usable_train_pairs: usize,
    usable_validation_pairs: usize,
    train_steps: usize,
    batch_size: usize,
    validation_batches: usize,
    seed: u64,
    learning_rate: f64,
    max_gradient_norm: f64,
    graph_mass_tolerance_da: f64,
    maximum_outgoing_edges: usize,
    k_best_graph_paths: usize,
    maximum_fragment_charge: usize,
    hidden_dim: usize,
    parent_unified_checkpoint: String,
    parent_unified_completed_steps: usize,
    reverse_causal_checkpoint: String,
    global_step: usize,
    best_validation_loss: f64,
    best_validation_structural_coverage: f64,
    inverse_config: FoundationDiffusionConfig,
}

#[derive(Debug, Clone, Copy, Default)]
struct GraphMetrics {
    mean_loss: f64,
    outgoing_edge_accuracy: f64,
    structural_coverage: f64,
    true_node_coverage: f64,
    true_edge_coverage: f64,
    graph_records: usize,
    structural_records: usize,
    classification_groups: usize,
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 5 {
        anyhow::bail!(
            "usage: foundation_train_cleavage_graph FOUNDATION_TRAINING.yaml OUTPUT_DIR PARENT_UNIFIED_CHECKPOINT REVERSE_CAUSAL_CHECKPOINT"
        );
    }
    let training_yaml = &args[1];
    let output_root = PathBuf::from(&args[2]);
    let parent_checkpoint = PathBuf::from(&args[3]);
    let reverse_checkpoint = PathBuf::from(&args[4]);
    let device = Device::Cpu;

    let run = read_foundation_training_run_config(training_yaml)?;
    let corpus = load_foundation_corpus(&run.corpus)?;
    let benchmark = FoundationBenchmarkManifest::read_tsv(&run.benchmark_manifest)
        .with_context(|| format!("failed to read {:?}", run.benchmark_manifest))?;
    benchmark.validate_against_records(&corpus.records)?;

    let parent_metadata_path = metadata_path(&parent_checkpoint);
    let parent_metadata: UnifiedParentMetadata = serde_yaml::from_str(
        &fs::read_to_string(&parent_metadata_path)
            .with_context(|| format!("failed to read {parent_metadata_path:?}"))?,
    )?;
    parent_metadata
        .inverse_config
        .validate()
        .map_err(anyhow::Error::msg)?;
    let parent_model_path = resolve_model_safetensors(&parent_checkpoint);

    let reverse_metadata_path = metadata_path(&reverse_checkpoint);
    let reverse_metadata: ReverseCausalCheckpointMetadata = serde_yaml::from_str(
        &fs::read_to_string(&reverse_metadata_path)
            .with_context(|| format!("failed to read {reverse_metadata_path:?}"))?,
    )?;
    if reverse_metadata.objective != "reverse_causal_next_token_ce_v01313"
        || reverse_metadata.direction != FOUNDATION_REVERSE_CAUSAL_DIRECTION_V01313
        || reverse_metadata.parameter_namespace != FOUNDATION_REVERSE_CAUSAL_NAMESPACE_V01313
    {
        anyhow::bail!("reverse checkpoint is not the accepted v0.13.13 isolated C->N branch");
    }
    if reverse_metadata.inverse_config != parent_metadata.inverse_config {
        anyhow::bail!("reverse checkpoint inverse config does not match parent unified config");
    }

    let expected_corpus = format!("fnv1a64:{:016x}", corpus.corpus_fingerprint);
    let expected_benchmark = format!("fnv1a64:{:016x}", benchmark.manifest_fingerprint());
    if parent_metadata.corpus_fingerprint != expected_corpus
        || reverse_metadata.corpus_fingerprint != expected_corpus
    {
        anyhow::bail!("parent/reverse corpus fingerprint does not match current corpus");
    }
    if parent_metadata.benchmark_manifest_fingerprint != expected_benchmark
        || reverse_metadata.benchmark_manifest_fingerprint != expected_benchmark
    {
        anyhow::bail!("parent/reverse benchmark fingerprint does not match current benchmark");
    }
    if PathBuf::from(&reverse_metadata.parent_unified_checkpoint) != parent_model_path {
        anyhow::bail!(
            "reverse checkpoint parent {:?} does not match requested parent {:?}",
            reverse_metadata.parent_unified_checkpoint,
            parent_model_path
        );
    }

    let vocabulary = FoundationDiffusionVocabulary;
    let train_indices = usable_indices(
        &corpus.records,
        &benchmark,
        FoundationPartition::Train,
        &parent_metadata.inverse_config,
        vocabulary,
    );
    let validation_indices = usable_indices(
        &corpus.records,
        &benchmark,
        FoundationPartition::Validation,
        &parent_metadata.inverse_config,
        vocabulary,
    );
    if train_indices.len() < BATCH_SIZE || validation_indices.len() < BATCH_SIZE {
        anyhow::bail!(
            "insufficient cleavage-graph pairs: train={} validation={}",
            train_indices.len(),
            validation_indices.len()
        );
    }
    let train_fingerprint =
        foundation_diffusion_dataset_fingerprint(&corpus.records, &train_indices)?;
    let validation_fingerprint =
        foundation_diffusion_dataset_fingerprint(&corpus.records, &validation_indices)?;
    let validation_selection = deterministic_subset(&validation_indices, VALIDATION_RECORDS, SEED);

    fs::create_dir_all(&output_root)?;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = PeptideSpectrumCleavageGraphScorer::new(vb)?;
    validate_cleavage_graph_namespace(&varmap)?;
    let mut optimizer = FoundationAdamW::new(
        &varmap,
        FoundationAdamWConfig {
            learning_rate: LEARNING_RATE,
            ..FoundationAdamWConfig::default()
        },
    )?;

    println!("objective\t{FOUNDATION_CLEAVAGE_GRAPH_OBJECTIVE_V01316}");
    println!("parameter_namespace\t{FOUNDATION_CLEAVAGE_GRAPH_NAMESPACE_V01316}");
    println!("optimizer_scope\tcleavage_graph_only");
    println!("parent_unified_frozen\tYES");
    println!("diffusion_parameters_in_optimizer\tNO");
    println!("n_to_c_causal_parameters_in_optimizer\tNO");
    println!("reverse_causal_parameters_in_optimizer\tNO");
    println!("forward_parameters_in_optimizer\tNO");
    println!("test_partition_consumed\tNO");
    println!("corpus_fingerprint\t{expected_corpus}");
    println!("benchmark_manifest_fingerprint\t{expected_benchmark}");
    println!("train_dataset_fingerprint\tfnv1a64:{train_fingerprint:016x}");
    println!("validation_dataset_fingerprint\tfnv1a64:{validation_fingerprint:016x}");
    println!("usable_train_pairs\t{}", train_indices.len());
    println!("usable_validation_pairs\t{}", validation_indices.len());
    println!("train_steps\t{TRAIN_STEPS}");
    println!("batch_size\t{BATCH_SIZE}");
    println!("validation_records\t{VALIDATION_RECORDS}");
    println!("validation_batches\t{VALIDATION_BATCHES}");
    println!("seed\t{SEED}");
    println!("learning_rate\t{LEARNING_RATE}");
    println!("max_gradient_norm\t{MAX_GRADIENT_NORM}");
    println!(
        "graph_mass_tolerance_da\t{}",
        FOUNDATION_CLEAVAGE_GRAPH_MASS_TOLERANCE_DA_V01316
    );
    println!(
        "maximum_outgoing_edges\t{}",
        FOUNDATION_CLEAVAGE_GRAPH_MAX_OUTGOING_EDGES_V01316
    );
    println!(
        "k_best_graph_paths\t{}",
        FOUNDATION_CLEAVAGE_GRAPH_K_BEST_PATHS_V01316
    );
    println!(
        "maximum_fragment_charge\t{}",
        FOUNDATION_CLEAVAGE_GRAPH_MAX_FRAGMENT_CHARGE_V01316
    );
    println!("edge_scorer_hidden_dim\t{FOUNDATION_CLEAVAGE_GRAPH_HIDDEN_DIM_V01316}");
    println!("parent_unified_checkpoint\t{}", parent_model_path.display());
    println!(
        "parent_unified_completed_steps\t{}",
        parent_metadata.completed_steps
    );
    println!(
        "reverse_causal_checkpoint\t{}",
        resolve_model_safetensors(&reverse_checkpoint).display()
    );
    println!("graph_construction\tobserved_b_like+precursor_complementary_y_like_v01316");
    println!("training_target\tper_true_source_node_outgoing_edge_ce_v01316");

    let (structurally_present, structural_records) =
        report_true_path_structural_presence(&corpus.records, &validation_selection)?;
    println!(
        "true_path_structural_coverage\tpresent={}\trecords={}\tcoverage={:.6}",
        structurally_present,
        structural_records,
        structurally_present as f64 / structural_records.max(1) as f64
    );

    let initial = evaluate(
        &model,
        &corpus.records,
        &validation_selection,
        BATCH_SIZE,
        &device,
    )?;
    println_metrics("initial_validation", 0, initial);

    let mut best_validation_loss = initial.mean_loss;
    let mut best_validation_structural_coverage = initial.structural_coverage;
    let initial_metadata = checkpoint_metadata(
        &parent_metadata,
        &corpus,
        &benchmark,
        train_fingerprint,
        validation_fingerprint,
        train_indices.len(),
        validation_indices.len(),
        &parent_model_path,
        &reverse_checkpoint,
        0,
        best_validation_loss,
        best_validation_structural_coverage,
    );
    save_checkpoint(&output_root.join("best"), &varmap, &initial_metadata)?;

    let mut training_rng = GraphRng::new(SEED ^ 0x8af2_771e_190d_3bc5);
    let mut global_step = 0usize;
    let mut attempts = 0usize;
    while global_step < TRAIN_STEPS {
        attempts += 1;
        if attempts > TRAIN_STEPS * 50 {
            anyhow::bail!(
                "unable to obtain enough structurally trainable cleavage-graph batches after {attempts} attempts"
            );
        }
        let selected_indices = (0..BATCH_SIZE)
            .map(|_| train_indices[training_rng.next_u64() as usize % train_indices.len()])
            .collect::<Vec<_>>();
        let packed = build_examples(&corpus.records, &selected_indices)?;
        let refs = packed
            .iter()
            .map(|(graph, audit)| (graph, audit))
            .collect::<Vec<_>>();
        let Some(batch) = foundation_cleavage_graph_training_batch(&refs, &device)? else {
            continue;
        };
        let loss = foundation_cleavage_graph_outgoing_edge_loss(&model, &batch)?;
        let optimizer_step = optimizer.backward_step(&loss, Some(MAX_GRADIENT_NORM))?;
        global_step += 1;

        if global_step == 1 || global_step % 25 == 0 {
            println!(
                "train_step\tstep={global_step}\tloss={:.6}\tgroups={}\tpre_clip_gradient_norm={:.6}\tgradient_scale={:.6}",
                f64::from(loss.to_scalar::<f32>()?),
                batch.groups,
                optimizer_step.gradient_norm,
                optimizer_step.gradient_scale
            );
        }

        if global_step % VALIDATION_INTERVAL == 0 || global_step == TRAIN_STEPS {
            let validation = evaluate(
                &model,
                &corpus.records,
                &validation_selection,
                BATCH_SIZE,
                &device,
            )?;
            println_metrics("validation", global_step, validation);
            if validation.mean_loss < best_validation_loss {
                best_validation_loss = validation.mean_loss;
                best_validation_structural_coverage = validation.structural_coverage;
                let metadata = checkpoint_metadata(
                    &parent_metadata,
                    &corpus,
                    &benchmark,
                    train_fingerprint,
                    validation_fingerprint,
                    train_indices.len(),
                    validation_indices.len(),
                    &parent_model_path,
                    &reverse_checkpoint,
                    global_step,
                    best_validation_loss,
                    best_validation_structural_coverage,
                );
                save_checkpoint(&output_root.join("best"), &varmap, &metadata)?;
            }
        }
    }

    let final_metrics = evaluate(
        &model,
        &corpus.records,
        &validation_selection,
        BATCH_SIZE,
        &device,
    )?;
    let final_metadata = checkpoint_metadata(
        &parent_metadata,
        &corpus,
        &benchmark,
        train_fingerprint,
        validation_fingerprint,
        train_indices.len(),
        validation_indices.len(),
        &parent_model_path,
        &reverse_checkpoint,
        global_step,
        best_validation_loss,
        best_validation_structural_coverage,
    );
    save_checkpoint(&output_root.join("final"), &varmap, &final_metadata)?;
    println_metrics("final_validation", global_step, final_metrics);
    println!("best_validation_loss\t{best_validation_loss:.8}");
    println!("best_validation_structural_coverage\t{best_validation_structural_coverage:.8}");
    println!("final_checkpoint\t{}", output_root.join("final").display());
    println!("best_checkpoint\t{}", output_root.join("best").display());
    Ok(())
}

fn report_true_path_structural_presence(
    records: &[FoundationTrainingRecord],
    indices: &[usize],
) -> Result<(usize, usize)> {
    let mut structurally_present = 0usize;
    for &index in indices {
        let record = &records[index];
        let expected_edges = record.peptidoform.sequence.chars().count();
        let Some(spectrum) = FoundationSpectrum::from_training_record(record) else {
            println!(
                "cleavage_graph_structural\trecord_index={index}\ttrue_path_structurally_present=NO\tpresent_nodes=0\ttotal_nodes={}\tpresent_edges=0\ttotal_edges={}\treason=spectrum_unavailable",
                expected_edges + 1,
                expected_edges
            );
            continue;
        };
        let graph = match foundation_build_cleavage_graph(record, &spectrum) {
            Ok(Some(graph)) => graph,
            Ok(None) => {
                println!(
                    "cleavage_graph_structural\trecord_index={index}\ttrue_path_structurally_present=NO\tpresent_nodes=0\ttotal_nodes={}\tpresent_edges=0\ttotal_edges={}\treason=graph_unavailable",
                    expected_edges + 1,
                    expected_edges
                );
                continue;
            }
            Err(error) => {
                println!(
                    "cleavage_graph_structural\trecord_index={index}\ttrue_path_structurally_present=NO\tpresent_nodes=0\ttotal_nodes={}\tpresent_edges=0\ttotal_edges={}\treason=graph_construction_error:{}",
                    expected_edges + 1,
                    expected_edges,
                    sanitize_diagnostic_text(&error)
                );
                continue;
            }
        };
        match foundation_cleavage_graph_true_path_audit(&graph, &record.peptidoform) {
            Ok(audit) => {
                structurally_present += usize::from(audit.structurally_present);
                println!(
                    "cleavage_graph_structural\trecord_index={index}\ttrue_path_structurally_present={}\tpresent_nodes={}\ttotal_nodes={}\tpresent_edges={}\ttotal_edges={}\tgraph_nodes={}\tgraph_edges={}",
                    if audit.structurally_present { "YES" } else { "NO" },
                    audit.present_nodes,
                    audit.total_nodes,
                    audit.present_edges,
                    audit.total_edges,
                    graph.nodes.len(),
                    graph.edge_count()
                );
            }
            Err(error) => {
                println!(
                    "cleavage_graph_structural\trecord_index={index}\ttrue_path_structurally_present=NO\tpresent_nodes=0\ttotal_nodes={}\tpresent_edges=0\ttotal_edges={}\tgraph_nodes={}\tgraph_edges={}\treason=true_path_audit_error:{}",
                    expected_edges + 1,
                    expected_edges,
                    graph.nodes.len(),
                    graph.edge_count(),
                    sanitize_diagnostic_text(&error)
                );
            }
        }
    }
    Ok((structurally_present, indices.len()))
}

fn sanitize_diagnostic_text(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch == '\t' || ch == '\n' || ch == '\r' {
                ' '
            } else {
                ch
            }
        })
        .collect()
}

fn build_examples(
    records: &[FoundationTrainingRecord],
    indices: &[usize],
) -> Result<
    Vec<(
        redeem_properties::foundation::FoundationCleavageGraph,
        redeem_properties::foundation::FoundationCleavageGraphTruePathAudit,
    )>,
> {
    let mut examples = Vec::new();
    for &index in indices {
        let record = &records[index];
        let Some(spectrum) = FoundationSpectrum::from_training_record(record) else {
            continue;
        };
        let Some(graph) =
            foundation_build_cleavage_graph(record, &spectrum).map_err(anyhow::Error::msg)?
        else {
            continue;
        };
        let audit = match foundation_cleavage_graph_true_path_audit(&graph, &record.peptidoform) {
            Ok(audit) => audit,
            Err(_) => continue,
        };
        examples.push((graph, audit));
    }
    Ok(examples)
}

fn evaluate(
    model: &PeptideSpectrumCleavageGraphScorer,
    records: &[FoundationTrainingRecord],
    indices: &[usize],
    batch_size: usize,
    device: &Device,
) -> Result<GraphMetrics> {
    let mut loss_weighted_sum = 0.0f64;
    let mut correct = 0usize;
    let mut classification_groups = 0usize;
    let mut graph_records = 0usize;
    let mut structural_records = 0usize;
    let mut present_nodes = 0usize;
    let mut total_nodes = 0usize;
    let mut present_edges = 0usize;
    let mut total_edges = 0usize;

    for chunk in indices.chunks(batch_size) {
        let mut packed = Vec::new();
        for &index in chunk {
            let record = &records[index];
            let true_edges = record.peptidoform.sequence.chars().count();
            graph_records += 1;
            total_edges += true_edges;
            total_nodes += true_edges + 1;

            let Some(spectrum) = FoundationSpectrum::from_training_record(record) else {
                continue;
            };
            let Some(graph) =
                foundation_build_cleavage_graph(record, &spectrum).map_err(anyhow::Error::msg)?
            else {
                continue;
            };
            let Ok(audit) = foundation_cleavage_graph_true_path_audit(&graph, &record.peptidoform)
            else {
                continue;
            };
            structural_records += usize::from(audit.structurally_present);
            present_nodes += audit.present_nodes;
            present_edges += audit.present_edges;
            packed.push((graph, audit));
        }

        let refs = packed
            .iter()
            .map(|(graph, audit)| (graph, audit))
            .collect::<Vec<_>>();
        let Some(batch) = foundation_cleavage_graph_training_batch(&refs, device)? else {
            continue;
        };
        let loss = foundation_cleavage_graph_outgoing_edge_loss(model, &batch)?;
        loss_weighted_sum += f64::from(loss.to_scalar::<f32>()?) * batch.groups as f64;
        let logits = model.forward_batch(&batch)?.to_vec2::<f32>()?;
        let targets = batch.target_indices.to_vec1::<u32>()?;
        for (row, target) in logits.iter().zip(targets) {
            let predicted = row
                .iter()
                .enumerate()
                .max_by(|left, right| left.1.total_cmp(right.1))
                .map(|(index, _)| index)
                .unwrap_or(0);
            correct += usize::from(predicted == target as usize);
        }
        classification_groups += batch.groups;
    }

    if graph_records == 0 || classification_groups == 0 {
        anyhow::bail!(
            "cleavage-graph validation produced no requested records/classification groups"
        );
    }
    Ok(GraphMetrics {
        mean_loss: loss_weighted_sum / classification_groups as f64,
        outgoing_edge_accuracy: correct as f64 / classification_groups as f64,
        structural_coverage: structural_records as f64 / graph_records as f64,
        true_node_coverage: present_nodes as f64 / total_nodes.max(1) as f64,
        true_edge_coverage: present_edges as f64 / total_edges.max(1) as f64,
        graph_records,
        structural_records,
        classification_groups,
    })
}

fn println_metrics(label: &str, step: usize, metrics: GraphMetrics) {
    println!(
        "{label}\tstep={step}\tloss={:.6}\toutgoing_edge_accuracy={:.6}\ttrue_path_structural_coverage={:.6}\ttrue_node_coverage={:.6}\ttrue_edge_coverage={:.6}\tgraph_records={}\tstructural_records={}\tclassification_groups={}",
        metrics.mean_loss,
        metrics.outgoing_edge_accuracy,
        metrics.structural_coverage,
        metrics.true_node_coverage,
        metrics.true_edge_coverage,
        metrics.graph_records,
        metrics.structural_records,
        metrics.classification_groups
    );
}

#[allow(clippy::too_many_arguments)]
fn checkpoint_metadata(
    parent_metadata: &UnifiedParentMetadata,
    corpus: &redeem_properties::foundation::FoundationCorpus,
    benchmark: &FoundationBenchmarkManifest,
    train_fingerprint: u64,
    validation_fingerprint: u64,
    usable_train_pairs: usize,
    usable_validation_pairs: usize,
    parent_model_path: &Path,
    reverse_checkpoint: &Path,
    global_step: usize,
    best_validation_loss: f64,
    best_validation_structural_coverage: f64,
) -> CleavageGraphCheckpointMetadata {
    CleavageGraphCheckpointMetadata {
        version: 1,
        objective: FOUNDATION_CLEAVAGE_GRAPH_OBJECTIVE_V01316.into(),
        parameter_namespace: FOUNDATION_CLEAVAGE_GRAPH_NAMESPACE_V01316.into(),
        corpus_fingerprint: format!("fnv1a64:{:016x}", corpus.corpus_fingerprint),
        benchmark_manifest_fingerprint: format!(
            "fnv1a64:{:016x}",
            benchmark.manifest_fingerprint()
        ),
        train_dataset_fingerprint: format!("fnv1a64:{train_fingerprint:016x}"),
        validation_dataset_fingerprint: format!("fnv1a64:{validation_fingerprint:016x}"),
        usable_train_pairs,
        usable_validation_pairs,
        train_steps: TRAIN_STEPS,
        batch_size: BATCH_SIZE,
        validation_batches: VALIDATION_BATCHES,
        seed: SEED,
        learning_rate: LEARNING_RATE,
        max_gradient_norm: MAX_GRADIENT_NORM,
        graph_mass_tolerance_da: FOUNDATION_CLEAVAGE_GRAPH_MASS_TOLERANCE_DA_V01316,
        maximum_outgoing_edges: FOUNDATION_CLEAVAGE_GRAPH_MAX_OUTGOING_EDGES_V01316,
        k_best_graph_paths: FOUNDATION_CLEAVAGE_GRAPH_K_BEST_PATHS_V01316,
        maximum_fragment_charge: FOUNDATION_CLEAVAGE_GRAPH_MAX_FRAGMENT_CHARGE_V01316,
        hidden_dim: FOUNDATION_CLEAVAGE_GRAPH_HIDDEN_DIM_V01316,
        parent_unified_checkpoint: parent_model_path.display().to_string(),
        parent_unified_completed_steps: parent_metadata.completed_steps,
        reverse_causal_checkpoint: resolve_model_safetensors(reverse_checkpoint)
            .display()
            .to_string(),
        global_step,
        best_validation_loss,
        best_validation_structural_coverage,
        inverse_config: parent_metadata.inverse_config.clone(),
    }
}

fn save_checkpoint(
    directory: &Path,
    varmap: &VarMap,
    metadata: &CleavageGraphCheckpointMetadata,
) -> Result<()> {
    fs::create_dir_all(directory)?;
    varmap.save(directory.join("model.safetensors"))?;
    fs::write(
        directory.join("metadata.yaml"),
        serde_yaml::to_string(metadata)?,
    )?;
    Ok(())
}

fn usable_indices(
    records: &[FoundationTrainingRecord],
    benchmark: &FoundationBenchmarkManifest,
    partition: FoundationPartition,
    config: &FoundationDiffusionConfig,
    vocabulary: FoundationDiffusionVocabulary,
) -> Vec<usize> {
    benchmark
        .entries
        .iter()
        .filter(|entry| entry.partition == partition)
        .filter_map(|entry| {
            let record = &records[entry.record_index];
            let usable = FoundationSpectrum::from_training_record(record).is_some()
                && vocabulary
                    .encode(&record.peptidoform, config.max_tokens)
                    .is_ok();
            usable.then_some(entry.record_index)
        })
        .collect()
}

fn deterministic_subset(indices: &[usize], requested: usize, seed: u64) -> Vec<usize> {
    let mut ranked = indices
        .iter()
        .copied()
        .map(|index| (mix64(seed ^ index as u64), index))
        .collect::<Vec<_>>();
    ranked.sort_unstable();
    ranked
        .into_iter()
        .take(requested.min(indices.len()))
        .map(|(_, index)| index)
        .collect()
}

fn metadata_path(path: &Path) -> PathBuf {
    if path.is_dir() {
        path.join("metadata.yaml")
    } else {
        path.parent()
            .unwrap_or_else(|| Path::new("."))
            .join("metadata.yaml")
    }
}

fn resolve_model_safetensors(path: &Path) -> PathBuf {
    if path.is_dir() {
        path.join("model.safetensors")
    } else {
        path.to_path_buf()
    }
}

#[derive(Debug, Clone, Copy)]
struct GraphRng {
    state: u64,
}

impl GraphRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0xa076_1d64_78bd_642f,
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

fn mix64(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}
