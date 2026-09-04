//! Isolated spectrum-conditioned cleavage-mass graph proposal branch.
//!
//! v0.13.16 deliberately avoids peptide-seed initialization and left-to-right
//! prefix commitment. The graph is constructed directly from observed fragment
//! evidence plus the measured precursor mass, connected by chemically valid
//! residue/PTM units, scored by a small isolated MLP, and decoded with a
//! deterministic precursor-mass-constrained k-best DAG search.

use super::chemistry::common_unimod_definition;
use super::data::FoundationTrainingRecord;
use super::diffusion::{
    foundation_diffusion_residue_ptm_valid, foundation_diffusion_token_mass_da,
    foundation_diffusion_token_residue, foundation_precursor_neutral_mass,
    FOUNDATION_DIFFUSION_CARBAMIDOMETHYL, FOUNDATION_DIFFUSION_DEAMIDATED,
    FOUNDATION_DIFFUSION_FIRST_RESIDUE, FOUNDATION_DIFFUSION_OXIDATION,
    FOUNDATION_DIFFUSION_RESIDUE_ACETYL, FOUNDATION_PEPTIDE_WATER_MASS_DA,
};
use super::featurize::{FoundationModification, FoundationModificationSite, PeptidoformInput};
use super::spectrum::FoundationSpectrum;
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{self as nn, loss, Linear, VarBuilder, VarMap};
use std::collections::HashSet;

/// Isolated parameter namespace for v0.13.16.
pub const FOUNDATION_CLEAVAGE_GRAPH_NAMESPACE_V01316: &str = "cleavage_graph";
/// Fixed training objective identifier.
pub const FOUNDATION_CLEAVAGE_GRAPH_OBJECTIVE_V01316: &str = "outgoing_edge_cross_entropy_v01316";
/// Fixed node/edge mass compatibility tolerance for the first pilot.
pub const FOUNDATION_CLEAVAGE_GRAPH_MASS_TOLERANCE_DA_V01316: f64 = 0.05;
/// Fixed outgoing-edge cap for every graph node.
pub const FOUNDATION_CLEAVAGE_GRAPH_MAX_OUTGOING_EDGES_V01316: usize = 64;
/// Fixed number of globally decoded graph paths retained per record.
pub const FOUNDATION_CLEAVAGE_GRAPH_K_BEST_PATHS_V01316: usize = 64;
/// Fixed fragment-charge hypotheses used to create cleavage nodes.
pub const FOUNDATION_CLEAVAGE_GRAPH_MAX_FRAGMENT_CHARGE_V01316: usize = 2;
/// Fixed hidden width of the deliberately small first edge scorer.
pub const FOUNDATION_CLEAVAGE_GRAPH_HIDDEN_DIM_V01316: usize = 64;
/// Fixed edge-feature width.
pub const FOUNDATION_CLEAVAGE_GRAPH_FEATURE_DIM_V01316: usize = 42;

const PROTON_MASS_DA: f64 = 1.007_276_466_77;
const RESIDUES: [char; 20] = [
    'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W',
    'Y',
];
const LOCAL_UNIMOD_IDS: [u32; 4] = [1, 4, 7, 35];

/// One plausible cumulative N-terminal residue mass in the graph.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FoundationCleavageGraphNode {
    /// Cumulative neutral residue/PTM mass from the peptide N terminus.
    pub mass_da: f64,
    /// Best normalized intensity supporting this mass as an N-terminal fragment.
    pub n_terminal_support: f32,
    /// Best normalized intensity supporting this mass via a complementary C-terminal fragment.
    pub c_terminal_support: f32,
    /// Whether this is the explicit 0-Da source node.
    pub is_source: bool,
    /// Whether this is the explicit precursor-residue-mass sink node.
    pub is_sink: bool,
}

impl FoundationCleavageGraphNode {
    fn total_support(self) -> f32 {
        self.n_terminal_support.max(self.c_terminal_support)
    }
}

/// One chemically valid residue or residue+PTM transition unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FoundationCleavageGraphUnit {
    /// Amino-acid residue added by this edge.
    pub residue: char,
    /// Optional residue-local canonical UniMod family.
    pub residue_unimod_id: Option<u32>,
    /// Whether the first edge also carries peptide N-terminal UniMod:1 acetylation.
    pub n_terminal_acetyl: bool,
}

impl FoundationCleavageGraphUnit {
    /// Monoisotopic residue/PTM mass added by this unit.
    pub fn mass_da(self) -> std::result::Result<f64, String> {
        let mut mass = residue_mass_da(self.residue)
            .ok_or_else(|| format!("unsupported cleavage-graph residue '{}'", self.residue))?;
        if let Some(unimod_id) = self.residue_unimod_id {
            let definition = common_unimod_definition(unimod_id)
                .ok_or_else(|| format!("missing UniMod:{unimod_id} definition"))?;
            mass += definition.mass_delta as f64;
        }
        if self.n_terminal_acetyl {
            let definition = common_unimod_definition(1)
                .ok_or_else(|| "missing UniMod:1 definition".to_string())?;
            mass += definition.mass_delta as f64;
        }
        Ok(mass)
    }
}

/// One directed residue/PTM transition between cleavage-mass nodes.
#[derive(Debug, Clone, PartialEq)]
pub struct FoundationCleavageGraphEdge {
    /// Source node index.
    pub source_node: usize,
    /// Target node index.
    pub target_node: usize,
    /// Chemical transition identity.
    pub unit: FoundationCleavageGraphUnit,
    /// Observed target-node mass minus the exact chemical transition endpoint.
    pub mass_residual_da: f64,
}

/// One precursor-mass-constrained cleavage DAG.
#[derive(Debug, Clone, PartialEq)]
pub struct FoundationCleavageGraph {
    /// Sorted cumulative-mass nodes. Source is index 0 and sink is last.
    pub nodes: Vec<FoundationCleavageGraphNode>,
    /// Deterministic outgoing edge lists aligned with `nodes`.
    pub outgoing: Vec<Vec<FoundationCleavageGraphEdge>>,
    /// Measured neutral precursor mass including water.
    pub precursor_neutral_mass: f64,
    /// Precursor residue/PTM mass total excluding water.
    pub residue_mass_total: f64,
    /// Positive precursor charge.
    pub precursor_charge: usize,
    /// Measured precursor m/z.
    pub precursor_mz: f64,
    /// Number of finite positive observed spectrum peaks.
    pub spectrum_peak_count: usize,
    /// Intensity-weighted observed product m/z mean.
    pub spectrum_weighted_mean_mz: f64,
}

impl FoundationCleavageGraph {
    /// Total directed edge count.
    pub fn edge_count(&self) -> usize {
        self.outgoing.iter().map(Vec::len).sum()
    }

    /// Explicit sink node index.
    pub fn sink_index(&self) -> usize {
        self.nodes.len().saturating_sub(1)
    }
}

/// True-path structural audit before any neural edge scoring.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FoundationCleavageGraphTruePathAudit {
    /// Whether every true cleavage node and true transition is present.
    pub structurally_present: bool,
    /// Number of true cumulative nodes represented, including source and sink.
    pub present_nodes: usize,
    /// Number of true cumulative nodes required, including source and sink.
    pub total_nodes: usize,
    /// Number of true residue/PTM transitions represented.
    pub present_edges: usize,
    /// Number of true residue/PTM transitions required.
    pub total_edges: usize,
    /// Training targets for true outgoing edges that are structurally available.
    pub training_groups: Vec<FoundationCleavageGraphTrainingGroup>,
}

/// One source-node outgoing-edge classification target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FoundationCleavageGraphTrainingGroup {
    /// Graph node whose outgoing alternatives form one CE classification row.
    pub source_node: usize,
    /// Index of the true transition within `graph.outgoing[source_node]`.
    pub target_outgoing_index: usize,
}

/// Padded outgoing-edge CE batch.
#[derive(Debug, Clone)]
pub struct FoundationCleavageGraphBatch {
    /// Edge features `[groups, max_outgoing, feature_dim]`.
    pub edge_features: Tensor,
    /// Additive score mask `[groups, max_outgoing]`, zero for valid and large negative for padding.
    pub additive_mask: Tensor,
    /// True outgoing-edge class `[groups]`.
    pub target_indices: Tensor,
    /// Number of source-node classification groups.
    pub groups: usize,
}

/// One globally decoded cleavage-graph peptide candidate.
#[derive(Debug, Clone, PartialEq)]
pub struct FoundationCleavageGraphCandidate {
    /// Canonical chemistry-aware peptide.
    pub peptide: PeptidoformInput,
    /// Sum of local outgoing-edge log probabilities along the source-to-sink path.
    pub path_log_probability: f64,
    /// Number of residue edges in the path.
    pub edge_count: usize,
}

/// Deliberately small v0.13.16 neural edge scorer.
pub struct PeptideSpectrumCleavageGraphScorer {
    input: Linear,
    output: Linear,
}

impl PeptideSpectrumCleavageGraphScorer {
    /// Construct the scorer under the fixed `cleavage_graph.*` namespace.
    pub fn new(vb: VarBuilder<'_>) -> Result<Self> {
        let vb = vb.pp(FOUNDATION_CLEAVAGE_GRAPH_NAMESPACE_V01316);
        let input = nn::linear(
            FOUNDATION_CLEAVAGE_GRAPH_FEATURE_DIM_V01316,
            FOUNDATION_CLEAVAGE_GRAPH_HIDDEN_DIM_V01316,
            vb.pp("edge_mlp.input"),
        )?;
        let output = nn::linear(
            FOUNDATION_CLEAVAGE_GRAPH_HIDDEN_DIM_V01316,
            1,
            vb.pp("edge_mlp.output"),
        )?;
        Ok(Self { input, output })
    }

    /// Score a flat `[edge, feature]` tensor and return `[edge]` logits.
    pub fn forward(&self, features: &Tensor) -> Result<Tensor> {
        let (_, feature_dim) = features.dims2()?;
        if feature_dim != FOUNDATION_CLEAVAGE_GRAPH_FEATURE_DIM_V01316 {
            candle_core::bail!(
                "cleavage-graph feature width {feature_dim} does not match fixed {}",
                FOUNDATION_CLEAVAGE_GRAPH_FEATURE_DIM_V01316
            );
        }
        self.output
            .forward(&self.input.forward(features)?.relu()?)?
            .squeeze(1)
    }

    /// Score one padded outgoing-edge training batch.
    pub fn forward_batch(&self, batch: &FoundationCleavageGraphBatch) -> Result<Tensor> {
        let (groups, max_outgoing, feature_dim) = batch.edge_features.dims3()?;
        let flat = batch
            .edge_features
            .reshape((groups * max_outgoing, feature_dim))?;
        let logits = self.forward(&flat)?.reshape((groups, max_outgoing))?;
        logits + &batch.additive_mask
    }
}

/// Validate that a scorer VarMap is isolated to `cleavage_graph.*`.
pub fn validate_cleavage_graph_namespace(varmap: &VarMap) -> Result<()> {
    let data = varmap
        .data()
        .lock()
        .map_err(|_| candle_core::Error::Msg("cleavage-graph VarMap lock poisoned".into()))?;
    if data.is_empty() {
        candle_core::bail!("cleavage-graph VarMap contains no variables");
    }
    let prefix = format!("{FOUNDATION_CLEAVAGE_GRAPH_NAMESPACE_V01316}.");
    if let Some(name) = data.keys().find(|name| !name.starts_with(&prefix)) {
        candle_core::bail!(
            "cleavage-graph variable '{name}' is outside expected namespace '{prefix}'"
        );
    }
    Ok(())
}

/// Build the deterministic v0.13.16 graph from observed spectrum + precursor only.
pub fn foundation_build_cleavage_graph(
    record: &FoundationTrainingRecord,
    spectrum: &FoundationSpectrum,
) -> std::result::Result<Option<FoundationCleavageGraph>, String> {
    let (precursor_mz, precursor_charge) =
        match (record.context.precursor_mz, record.context.charge) {
            (Some(mz), Some(charge)) if mz.is_finite() && mz > 0.0 && charge > 0 => {
                (mz as f64, charge as usize)
            }
            _ => return Ok(None),
        };
    let precursor_neutral_mass =
        foundation_precursor_neutral_mass(precursor_mz, precursor_charge as i32)?;
    let residue_mass_total = precursor_neutral_mass - FOUNDATION_PEPTIDE_WATER_MASS_DA;
    if !(residue_mass_total > 0.0 && residue_mass_total.is_finite()) {
        return Ok(None);
    }

    let max_intensity = spectrum
        .peaks
        .iter()
        .filter_map(|peak| {
            (peak.intensity.is_finite() && peak.intensity > 0.0).then_some(peak.intensity as f64)
        })
        .fold(0.0f64, f64::max);
    if !(max_intensity > 0.0 && max_intensity.is_finite()) {
        return Ok(None);
    }

    #[derive(Debug, Clone, Copy)]
    struct CandidateNode {
        mass_da: f64,
        n_support: f32,
        c_support: f32,
    }

    let mut candidates = Vec::<CandidateNode>::new();
    let mut intensity_sum = 0.0f64;
    let mut intensity_mz_sum = 0.0f64;
    let mut spectrum_peak_count = 0usize;
    let max_fragment_charge = precursor_charge
        .min(FOUNDATION_CLEAVAGE_GRAPH_MAX_FRAGMENT_CHARGE_V01316)
        .max(1);

    for peak in &spectrum.peaks {
        if !(peak.mz.is_finite()
            && peak.mz > 0.0
            && peak.intensity.is_finite()
            && peak.intensity > 0.0)
        {
            continue;
        }
        let mz = peak.mz as f64;
        let normalized = (peak.intensity as f64 / max_intensity).clamp(0.0, 1.0) as f32;
        intensity_sum += peak.intensity as f64;
        intensity_mz_sum += peak.intensity as f64 * mz;
        spectrum_peak_count += 1;

        for fragment_charge in 1..=max_fragment_charge {
            let z = fragment_charge as f64;
            // b-like interpretation: b_z = (prefix residue mass + zH+)/z.
            let n_prefix_mass = z * mz - z * PROTON_MASS_DA;
            if n_prefix_mass > FOUNDATION_CLEAVAGE_GRAPH_MASS_TOLERANCE_DA_V01316
                && n_prefix_mass
                    < residue_mass_total - FOUNDATION_CLEAVAGE_GRAPH_MASS_TOLERANCE_DA_V01316
            {
                candidates.push(CandidateNode {
                    mass_da: n_prefix_mass,
                    n_support: normalized,
                    c_support: 0.0,
                });
            }

            // y-like interpretation: y_z contains suffix residue mass + water.
            // Convert it into the complementary N-terminal cumulative residue mass.
            let suffix_residue_mass =
                z * mz - z * PROTON_MASS_DA - FOUNDATION_PEPTIDE_WATER_MASS_DA;
            let complementary_prefix_mass = residue_mass_total - suffix_residue_mass;
            if complementary_prefix_mass > FOUNDATION_CLEAVAGE_GRAPH_MASS_TOLERANCE_DA_V01316
                && complementary_prefix_mass
                    < residue_mass_total - FOUNDATION_CLEAVAGE_GRAPH_MASS_TOLERANCE_DA_V01316
            {
                candidates.push(CandidateNode {
                    mass_da: complementary_prefix_mass,
                    n_support: 0.0,
                    c_support: normalized,
                });
            }
        }
    }
    if spectrum_peak_count == 0 {
        return Ok(None);
    }

    candidates.sort_by(|left, right| left.mass_da.total_cmp(&right.mass_da));
    let mut clustered = Vec::<FoundationCleavageGraphNode>::new();
    for candidate in candidates {
        if let Some(last) = clustered.last_mut() {
            if (candidate.mass_da - last.mass_da).abs()
                <= FOUNDATION_CLEAVAGE_GRAPH_MASS_TOLERANCE_DA_V01316
            {
                let candidate_support = candidate.n_support.max(candidate.c_support);
                if candidate_support > last.total_support() {
                    last.mass_da = candidate.mass_da;
                }
                last.n_terminal_support = last.n_terminal_support.max(candidate.n_support);
                last.c_terminal_support = last.c_terminal_support.max(candidate.c_support);
                continue;
            }
        }
        clustered.push(FoundationCleavageGraphNode {
            mass_da: candidate.mass_da,
            n_terminal_support: candidate.n_support,
            c_terminal_support: candidate.c_support,
            is_source: false,
            is_sink: false,
        });
    }

    clustered.retain(|node| {
        node.mass_da > FOUNDATION_CLEAVAGE_GRAPH_MASS_TOLERANCE_DA_V01316
            && (residue_mass_total - node.mass_da).abs()
                > FOUNDATION_CLEAVAGE_GRAPH_MASS_TOLERANCE_DA_V01316
    });
    clustered.sort_by(|left, right| left.mass_da.total_cmp(&right.mass_da));

    let mut nodes = Vec::with_capacity(clustered.len() + 2);
    nodes.push(FoundationCleavageGraphNode {
        mass_da: 0.0,
        n_terminal_support: 1.0,
        c_terminal_support: 0.0,
        is_source: true,
        is_sink: false,
    });
    nodes.extend(clustered);
    nodes.push(FoundationCleavageGraphNode {
        mass_da: residue_mass_total,
        n_terminal_support: 0.0,
        c_terminal_support: 1.0,
        is_source: false,
        is_sink: true,
    });

    let mut outgoing = vec![Vec::<FoundationCleavageGraphEdge>::new(); nodes.len()];
    for source_node in 0..nodes.len().saturating_sub(1) {
        let source_mass = nodes[source_node].mass_da;
        let units = allowed_units(source_node == 0)?;
        let mut edges = Vec::<FoundationCleavageGraphEdge>::new();
        for unit in units {
            let unit_mass = unit.mass_da()?;
            let expected = source_mass + unit_mass;
            if expected > residue_mass_total + FOUNDATION_CLEAVAGE_GRAPH_MASS_TOLERANCE_DA_V01316 {
                continue;
            }
            let low = expected - FOUNDATION_CLEAVAGE_GRAPH_MASS_TOLERANCE_DA_V01316;
            let high = expected + FOUNDATION_CLEAVAGE_GRAPH_MASS_TOLERANCE_DA_V01316;
            let first_target = lower_bound_node_mass(&nodes, source_node + 1, low);
            for target_node in first_target..nodes.len() {
                let target_mass = nodes[target_node].mass_da;
                if target_mass > high {
                    break;
                }
                edges.push(FoundationCleavageGraphEdge {
                    source_node,
                    target_node,
                    unit,
                    mass_residual_da: target_mass - expected,
                });
            }
        }
        edges.sort_by(|left, right| {
            left.mass_residual_da
                .abs()
                .total_cmp(&right.mass_residual_da.abs())
                .then_with(|| {
                    nodes[right.target_node]
                        .total_support()
                        .total_cmp(&nodes[left.target_node].total_support())
                })
                .then_with(|| left.target_node.cmp(&right.target_node))
                .then_with(|| left.unit.cmp(&right.unit))
        });
        edges.dedup_by(|left, right| {
            left.target_node == right.target_node && left.unit == right.unit
        });
        edges.truncate(FOUNDATION_CLEAVAGE_GRAPH_MAX_OUTGOING_EDGES_V01316);
        outgoing[source_node] = edges;
    }

    let spectrum_weighted_mean_mz = if intensity_sum > 0.0 {
        intensity_mz_sum / intensity_sum
    } else {
        0.0
    };
    Ok(Some(FoundationCleavageGraph {
        nodes,
        outgoing,
        precursor_neutral_mass,
        residue_mass_total,
        precursor_charge,
        precursor_mz,
        spectrum_peak_count,
        spectrum_weighted_mean_mz,
    }))
}

/// Audit whether the exact known peptide path exists before neural scoring.
pub fn foundation_cleavage_graph_true_path_audit(
    graph: &FoundationCleavageGraph,
    peptide: &PeptidoformInput,
) -> std::result::Result<FoundationCleavageGraphTruePathAudit, String> {
    let units = true_units(peptide)?;
    let total_edges = units.len();
    let total_nodes = total_edges + 1;
    let mut mapped_nodes = Vec::<Option<usize>>::with_capacity(total_nodes);
    mapped_nodes.push(Some(0));
    let mut cumulative = 0.0f64;
    for (unit_index, unit) in units.iter().copied().enumerate() {
        cumulative += unit.mass_da()?;
        if unit_index + 1 == units.len() {
            mapped_nodes.push(Some(graph.sink_index()));
        } else {
            mapped_nodes.push(closest_node_within_tolerance(graph, cumulative));
        }
    }

    let present_nodes = mapped_nodes.iter().filter(|node| node.is_some()).count();
    let mut present_edges = 0usize;
    let mut training_groups = Vec::<FoundationCleavageGraphTrainingGroup>::new();
    for edge_index in 0..units.len() {
        let (Some(source_node), Some(target_node)) =
            (mapped_nodes[edge_index], mapped_nodes[edge_index + 1])
        else {
            continue;
        };
        if let Some(target_outgoing_index) = graph.outgoing[source_node]
            .iter()
            .position(|edge| edge.target_node == target_node && edge.unit == units[edge_index])
        {
            present_edges += 1;
            training_groups.push(FoundationCleavageGraphTrainingGroup {
                source_node,
                target_outgoing_index,
            });
        }
    }

    Ok(FoundationCleavageGraphTruePathAudit {
        structurally_present: present_nodes == total_nodes && present_edges == total_edges,
        present_nodes,
        total_nodes,
        present_edges,
        total_edges,
        training_groups,
    })
}

/// Collate available true outgoing-edge classification groups from one record batch.
pub fn foundation_cleavage_graph_training_batch(
    examples: &[(
        &FoundationCleavageGraph,
        &FoundationCleavageGraphTruePathAudit,
    )],
    device: &Device,
) -> Result<Option<FoundationCleavageGraphBatch>> {
    let groups = examples
        .iter()
        .flat_map(|(_, audit)| audit.training_groups.iter())
        .count();
    if groups == 0 {
        return Ok(None);
    }
    let max_outgoing = FOUNDATION_CLEAVAGE_GRAPH_MAX_OUTGOING_EDGES_V01316;
    let feature_dim = FOUNDATION_CLEAVAGE_GRAPH_FEATURE_DIM_V01316;
    let mut features = vec![0.0f32; groups * max_outgoing * feature_dim];
    let mut mask = vec![-1.0e9f32; groups * max_outgoing];
    let mut targets = Vec::<u32>::with_capacity(groups);
    let mut group_index = 0usize;

    for (graph, audit) in examples {
        for group in &audit.training_groups {
            let outgoing = &graph.outgoing[group.source_node];
            if outgoing.is_empty() || group.target_outgoing_index >= outgoing.len() {
                candle_core::bail!("invalid cleavage-graph training group");
            }
            for (edge_index, edge) in outgoing.iter().enumerate() {
                let edge_features = foundation_cleavage_graph_edge_features(graph, edge);
                let offset = (group_index * max_outgoing + edge_index) * feature_dim;
                features[offset..offset + feature_dim].copy_from_slice(&edge_features);
                mask[group_index * max_outgoing + edge_index] = 0.0;
            }
            targets.push(group.target_outgoing_index as u32);
            group_index += 1;
        }
    }

    Ok(Some(FoundationCleavageGraphBatch {
        edge_features: Tensor::from_vec(features, (groups, max_outgoing, feature_dim), device)?,
        additive_mask: Tensor::from_vec(mask, (groups, max_outgoing), device)?,
        target_indices: Tensor::from_vec(targets, groups, device)?.to_dtype(DType::U32)?,
        groups,
    }))
}

/// Fixed per-source-node outgoing-edge cross entropy.
pub fn foundation_cleavage_graph_outgoing_edge_loss(
    model: &PeptideSpectrumCleavageGraphScorer,
    batch: &FoundationCleavageGraphBatch,
) -> Result<Tensor> {
    let logits = model.forward_batch(batch)?;
    loss::cross_entropy(&logits, &batch.target_indices)
}

/// Decode deterministic k-best source-to-sink paths with the learned edge scorer.
pub fn foundation_cleavage_graph_k_best_candidates(
    model: &PeptideSpectrumCleavageGraphScorer,
    graph: &FoundationCleavageGraph,
    device: &Device,
) -> Result<Vec<FoundationCleavageGraphCandidate>> {
    let edge_scores = foundation_cleavage_graph_edge_scores(model, graph, device)?;
    foundation_cleavage_graph_k_best_from_edge_scores(graph, &edge_scores)
        .map_err(candle_core::Error::Msg)
}

/// Decode k-best paths from externally supplied raw edge logits.
///
/// This public helper makes the deterministic graph search independently testable
/// without depending on random scorer initialization.
pub fn foundation_cleavage_graph_k_best_from_edge_scores(
    graph: &FoundationCleavageGraph,
    edge_scores: &[Vec<f64>],
) -> std::result::Result<Vec<FoundationCleavageGraphCandidate>, String> {
    if edge_scores.len() != graph.outgoing.len() {
        return Err("cleavage-graph edge-score/source-node arity mismatch".into());
    }
    let mut local_log_probabilities = Vec::<Vec<f64>>::with_capacity(edge_scores.len());
    for (scores, outgoing) in edge_scores.iter().zip(&graph.outgoing) {
        if scores.len() != outgoing.len() {
            return Err("cleavage-graph edge-score/outgoing arity mismatch".into());
        }
        local_log_probabilities.push(log_softmax(scores));
    }

    #[derive(Debug, Clone, PartialEq)]
    struct PathState {
        score: f64,
        units: Vec<FoundationCleavageGraphUnit>,
    }

    let mut paths = vec![Vec::<PathState>::new(); graph.nodes.len()];
    paths[0].push(PathState {
        score: 0.0,
        units: Vec::new(),
    });

    for source_node in 0..graph.nodes.len().saturating_sub(1) {
        if paths[source_node].is_empty() {
            continue;
        }
        let source_paths = paths[source_node].clone();
        for state in source_paths {
            for (edge_index, edge) in graph.outgoing[source_node].iter().enumerate() {
                let mut units = state.units.clone();
                units.push(edge.unit);
                paths[edge.target_node].push(PathState {
                    score: state.score + local_log_probabilities[source_node][edge_index],
                    units,
                });
                paths[edge.target_node].sort_by(|left, right| {
                    right
                        .score
                        .total_cmp(&left.score)
                        .then_with(|| left.units.cmp(&right.units))
                });
                paths[edge.target_node].truncate(FOUNDATION_CLEAVAGE_GRAPH_K_BEST_PATHS_V01316);
            }
        }
    }

    let mut candidates = Vec::<FoundationCleavageGraphCandidate>::new();
    let mut seen = HashSet::<String>::new();
    let mut sink_paths = paths[graph.sink_index()].clone();
    sink_paths.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.units.cmp(&right.units))
    });
    for path in sink_paths {
        // A path reaches the measured precursor sink through node-local tolerances, but
        // individual edge residuals can accumulate. Enforce one final hard precursor
        // residue-mass constraint before a decoded peptide enters the proposal pool.
        let theoretical_residue_mass = path
            .units
            .iter()
            .copied()
            .map(FoundationCleavageGraphUnit::mass_da)
            .collect::<std::result::Result<Vec<_>, _>>()?
            .into_iter()
            .sum::<f64>();
        if (theoretical_residue_mass - graph.residue_mass_total).abs()
            > FOUNDATION_CLEAVAGE_GRAPH_MASS_TOLERANCE_DA_V01316
        {
            continue;
        }
        let peptide = peptide_from_units(&path.units)?;
        let key = canonical_candidate_key(&peptide);
        if seen.insert(key) {
            candidates.push(FoundationCleavageGraphCandidate {
                peptide,
                path_log_probability: path.score,
                edge_count: path.units.len(),
            });
        }
        if candidates.len() >= FOUNDATION_CLEAVAGE_GRAPH_K_BEST_PATHS_V01316 {
            break;
        }
    }
    Ok(candidates)
}

/// Fixed edge feature vector used by the v0.13.16 MLP.
pub fn foundation_cleavage_graph_edge_features(
    graph: &FoundationCleavageGraph,
    edge: &FoundationCleavageGraphEdge,
) -> [f32; FOUNDATION_CLEAVAGE_GRAPH_FEATURE_DIM_V01316] {
    let mut features = [0.0f32; FOUNDATION_CLEAVAGE_GRAPH_FEATURE_DIM_V01316];
    let left = graph.nodes[edge.source_node];
    let right = graph.nodes[edge.target_node];
    let total = graph.residue_mass_total.max(f64::EPSILON);
    features[0] = (left.mass_da / total) as f32;
    features[1] = (right.mass_da / total) as f32;
    features[2] = ((right.mass_da - left.mass_da) / 200.0) as f32;
    features[3] = (edge.mass_residual_da / FOUNDATION_CLEAVAGE_GRAPH_MASS_TOLERANCE_DA_V01316)
        .clamp(-1.0, 1.0) as f32;
    features[4] = left.n_terminal_support;
    features[5] = left.c_terminal_support;
    features[6] = right.n_terminal_support;
    features[7] = right.c_terminal_support;
    features[8] = left.total_support();
    features[9] = right.total_support();
    features[10] = (graph.precursor_charge as f32 / 6.0).clamp(0.0, 1.5);
    features[11] = (graph.precursor_mz / 2000.0) as f32;
    features[12] = (graph.spectrum_peak_count as f32 / 256.0).min(4.0);
    features[13] = (graph.spectrum_weighted_mean_mz / 2000.0) as f32;
    features[14] = if left.is_source { 1.0 } else { 0.0 };
    features[15] = if right.is_sink { 1.0 } else { 0.0 };
    features[16] = if edge.unit.n_terminal_acetyl {
        1.0
    } else {
        0.0
    };

    if let Some(residue_index) = RESIDUES
        .iter()
        .position(|&residue| residue == edge.unit.residue)
    {
        features[17 + residue_index] = 1.0;
    }
    let modification_index = match edge.unit.residue_unimod_id {
        None => 0,
        Some(1) => 1,
        Some(4) => 2,
        Some(7) => 3,
        Some(35) => 4,
        Some(_) => 0,
    };
    features[37 + modification_index] = 1.0;
    features
}

fn foundation_cleavage_graph_edge_scores(
    model: &PeptideSpectrumCleavageGraphScorer,
    graph: &FoundationCleavageGraph,
    device: &Device,
) -> Result<Vec<Vec<f64>>> {
    let edge_count = graph.edge_count();
    if edge_count == 0 {
        return Ok(vec![Vec::new(); graph.nodes.len()]);
    }
    let mut features =
        Vec::<f32>::with_capacity(edge_count * FOUNDATION_CLEAVAGE_GRAPH_FEATURE_DIM_V01316);
    let mut lengths = Vec::<usize>::with_capacity(graph.outgoing.len());
    for outgoing in &graph.outgoing {
        lengths.push(outgoing.len());
        for edge in outgoing {
            features.extend_from_slice(&foundation_cleavage_graph_edge_features(graph, edge));
        }
    }
    let tensor = Tensor::from_vec(
        features,
        (edge_count, FOUNDATION_CLEAVAGE_GRAPH_FEATURE_DIM_V01316),
        device,
    )?;
    let flat_scores = model.forward(&tensor)?.to_vec1::<f32>()?;
    let mut cursor = 0usize;
    let mut scores = Vec::<Vec<f64>>::with_capacity(lengths.len());
    for length in lengths {
        scores.push(
            flat_scores[cursor..cursor + length]
                .iter()
                .map(|&value| value as f64)
                .collect(),
        );
        cursor += length;
    }
    Ok(scores)
}

fn allowed_units(
    include_n_terminal_acetyl: bool,
) -> std::result::Result<Vec<FoundationCleavageGraphUnit>, String> {
    let mut units = Vec::<FoundationCleavageGraphUnit>::new();
    for residue in RESIDUES {
        let base = FoundationCleavageGraphUnit {
            residue,
            residue_unimod_id: None,
            n_terminal_acetyl: false,
        };
        units.push(base);
        for unimod_id in LOCAL_UNIMOD_IDS {
            let token = unimod_token(unimod_id)?;
            if foundation_diffusion_residue_ptm_valid(token, residue) {
                units.push(FoundationCleavageGraphUnit {
                    residue,
                    residue_unimod_id: Some(unimod_id),
                    n_terminal_acetyl: false,
                });
            }
        }
    }
    if include_n_terminal_acetyl {
        let with_nterm = units
            .iter()
            .copied()
            .map(|mut unit| {
                unit.n_terminal_acetyl = true;
                unit
            })
            .collect::<Vec<_>>();
        units.extend(with_nterm);
    }
    units.sort_by(|left, right| {
        left.mass_da()
            .unwrap_or(f64::INFINITY)
            .total_cmp(&right.mass_da().unwrap_or(f64::INFINITY))
            .then_with(|| left.cmp(right))
    });
    Ok(units)
}

fn true_units(
    peptide: &PeptidoformInput,
) -> std::result::Result<Vec<FoundationCleavageGraphUnit>, String> {
    let residues = peptide.sequence.chars().collect::<Vec<_>>();
    if residues.is_empty() {
        return Err("cleavage-graph true peptide cannot be empty".into());
    }
    if peptide
        .modifications
        .iter()
        .any(|modification| modification.site == FoundationModificationSite::CTerm)
    {
        return Err("cleavage graph does not support C-terminal modification units".into());
    }
    let nterm = peptide
        .modifications
        .iter()
        .filter(|modification| modification.site == FoundationModificationSite::NTerm)
        .collect::<Vec<_>>();
    if nterm.len() > 1
        || nterm
            .iter()
            .any(|modification| modification.unimod_id != Some(1))
    {
        return Err("cleavage graph supports at most one N-terminal UniMod:1".into());
    }

    let mut units = Vec::<FoundationCleavageGraphUnit>::with_capacity(residues.len());
    for (residue_index, residue) in residues.iter().copied().enumerate() {
        let local = peptide
            .modifications
            .iter()
            .filter(|modification| {
                modification.site == FoundationModificationSite::Residue(residue_index)
            })
            .collect::<Vec<_>>();
        if local.len() > 1 {
            return Err(format!(
                "cleavage graph supports at most one residue-local PTM at residue {residue_index}"
            ));
        }
        let residue_unimod_id = local
            .first()
            .and_then(|modification| modification.unimod_id);
        if let Some(unimod_id) = residue_unimod_id {
            if !LOCAL_UNIMOD_IDS.contains(&unimod_id) {
                return Err(format!(
                    "cleavage graph does not support UniMod:{unimod_id}"
                ));
            }
            if !foundation_diffusion_residue_ptm_valid(unimod_token(unimod_id)?, residue) {
                return Err(format!(
                    "cleavage graph UniMod:{unimod_id} is invalid on residue {residue}"
                ));
            }
        } else if local.first().is_some() {
            return Err("cleavage graph does not support unresolved/open PTMs".into());
        }
        units.push(FoundationCleavageGraphUnit {
            residue,
            residue_unimod_id,
            n_terminal_acetyl: residue_index == 0 && !nterm.is_empty(),
        });
    }
    Ok(units)
}

fn peptide_from_units(
    units: &[FoundationCleavageGraphUnit],
) -> std::result::Result<PeptidoformInput, String> {
    if units.is_empty() {
        return Err("cleavage-graph path contains no residue units".into());
    }
    let sequence = units.iter().map(|unit| unit.residue).collect::<String>();
    let mut modifications = Vec::<FoundationModification>::new();
    if units[0].n_terminal_acetyl {
        let definition =
            common_unimod_definition(1).ok_or_else(|| "missing UniMod:1 definition".to_string())?;
        modifications.push(FoundationModification::unimod(
            FoundationModificationSite::NTerm,
            0,
            1,
            definition.mass_delta,
        ));
    }
    for (residue_index, unit) in units.iter().enumerate() {
        if residue_index > 0 && unit.n_terminal_acetyl {
            return Err("N-terminal acetyl marker appeared after the first graph edge".into());
        }
        if let Some(unimod_id) = unit.residue_unimod_id {
            let definition = common_unimod_definition(unimod_id)
                .ok_or_else(|| format!("missing UniMod:{unimod_id} definition"))?;
            modifications.push(FoundationModification::unimod(
                FoundationModificationSite::Residue(residue_index),
                residue_index,
                unimod_id,
                definition.mass_delta,
            ));
        }
    }
    Ok(PeptidoformInput {
        sequence,
        modifications,
    })
}

fn lower_bound_node_mass(
    nodes: &[FoundationCleavageGraphNode],
    start: usize,
    mass_da: f64,
) -> usize {
    let mut low = start.min(nodes.len());
    let mut high = nodes.len();
    while low < high {
        let middle = low + (high - low) / 2;
        if nodes[middle].mass_da < mass_da {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    low
}

fn closest_node_within_tolerance(graph: &FoundationCleavageGraph, mass_da: f64) -> Option<usize> {
    graph
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, node)| !node.is_source && !node.is_sink)
        .map(|(index, node)| (index, (node.mass_da - mass_da).abs()))
        .filter(|(_, error)| *error <= FOUNDATION_CLEAVAGE_GRAPH_MASS_TOLERANCE_DA_V01316)
        .min_by(|left, right| {
            left.1
                .total_cmp(&right.1)
                .then_with(|| left.0.cmp(&right.0))
        })
        .map(|(index, _)| index)
}

fn residue_mass_da(residue: char) -> Option<f64> {
    (FOUNDATION_DIFFUSION_FIRST_RESIDUE..FOUNDATION_DIFFUSION_FIRST_RESIDUE + 20)
        .find(|&token| foundation_diffusion_token_residue(token) == Some(residue))
        .and_then(foundation_diffusion_token_mass_da)
}

fn unimod_token(unimod_id: u32) -> std::result::Result<u32, String> {
    match unimod_id {
        1 => Ok(FOUNDATION_DIFFUSION_RESIDUE_ACETYL),
        4 => Ok(FOUNDATION_DIFFUSION_CARBAMIDOMETHYL),
        7 => Ok(FOUNDATION_DIFFUSION_DEAMIDATED),
        35 => Ok(FOUNDATION_DIFFUSION_OXIDATION),
        _ => Err(format!("unsupported cleavage-graph UniMod:{unimod_id}")),
    }
}

fn canonical_candidate_key(peptide: &PeptidoformInput) -> String {
    let mut modifications = peptide
        .modifications
        .iter()
        .map(|modification| {
            format!(
                "{:?}:{}:{}",
                modification.site,
                modification.unimod_id.unwrap_or(u32::MAX),
                modification.mass_delta.to_bits()
            )
        })
        .collect::<Vec<_>>();
    modifications.sort();
    format!("{}|{}", peptide.sequence, modifications.join(";"))
}

fn log_softmax(scores: &[f64]) -> Vec<f64> {
    if scores.is_empty() {
        return Vec::new();
    }
    let maximum = scores
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .fold(f64::NEG_INFINITY, f64::max);
    if !maximum.is_finite() {
        return vec![f64::NEG_INFINITY; scores.len()];
    }
    let sum_exp = scores
        .iter()
        .filter(|value| value.is_finite())
        .map(|value| (value - maximum).exp())
        .sum::<f64>();
    let log_normalizer = maximum + sum_exp.max(f64::MIN_POSITIVE).ln();
    scores
        .iter()
        .map(|value| {
            if value.is_finite() {
                value - log_normalizer
            } else {
                f64::NEG_INFINITY
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::foundation::{ObservedSpectrumPeak, TrainingContext};

    fn synthetic_record(peptide: PeptidoformInput, charge: i32) -> FoundationTrainingRecord {
        let neutral =
            super::super::diffusion::foundation_peptidoform_neutral_mass(&peptide).unwrap();
        let precursor_mz = ((neutral + PROTON_MASS_DA * charge as f64) / charge as f64) as f32;
        let units = true_units(&peptide).unwrap();
        let mut prefix = 0.0f64;
        let mut peaks = Vec::<ObservedSpectrumPeak>::new();
        for (index, unit) in units.iter().copied().enumerate() {
            prefix += unit.mass_da().unwrap();
            if index + 1 == units.len() {
                break;
            }
            let b_mz = prefix + PROTON_MASS_DA;
            let suffix_with_water = neutral - prefix;
            let y_mz = suffix_with_water + PROTON_MASS_DA;
            peaks.push(ObservedSpectrumPeak {
                mz: b_mz as f32,
                intensity: 1.0,
            });
            peaks.push(ObservedSpectrumPeak {
                mz: y_mz as f32,
                intensity: 0.8,
            });
        }
        FoundationTrainingRecord {
            peptidoform: peptide,
            retention_time: Default::default(),
            ccs: None,
            fragments: Vec::new(),
            observed_spectrum_peaks: peaks,
            context: TrainingContext {
                charge: Some(charge),
                precursor_mz: Some(precursor_mz),
                ..TrainingContext::default()
            },
            run_id: None,
        }
    }

    #[test]
    fn exact_b_y_ladder_contains_true_path_before_scoring() {
        let peptide = PeptidoformInput {
            sequence: "PEPTIDEK".into(),
            modifications: Vec::new(),
        };
        let record = synthetic_record(peptide.clone(), 2);
        let spectrum = FoundationSpectrum::from_training_record(&record).unwrap();
        let graph = foundation_build_cleavage_graph(&record, &spectrum)
            .unwrap()
            .unwrap();
        let audit = foundation_cleavage_graph_true_path_audit(&graph, &peptide).unwrap();
        assert!(audit.structurally_present);
        assert_eq!(audit.present_nodes, audit.total_nodes);
        assert_eq!(audit.present_edges, audit.total_edges);
        assert_eq!(audit.training_groups.len(), peptide.sequence.len());
    }

    #[test]
    fn graph_supports_existing_ptm_vocabulary_units() {
        let peptide = PeptidoformInput {
            sequence: "CMNK".into(),
            modifications: vec![
                FoundationModification::unimod(
                    FoundationModificationSite::NTerm,
                    0,
                    1,
                    common_unimod_definition(1).unwrap().mass_delta,
                ),
                FoundationModification::unimod(
                    FoundationModificationSite::Residue(0),
                    0,
                    4,
                    common_unimod_definition(4).unwrap().mass_delta,
                ),
                FoundationModification::unimod(
                    FoundationModificationSite::Residue(1),
                    1,
                    35,
                    common_unimod_definition(35).unwrap().mass_delta,
                ),
                FoundationModification::unimod(
                    FoundationModificationSite::Residue(2),
                    2,
                    7,
                    common_unimod_definition(7).unwrap().mass_delta,
                ),
                FoundationModification::unimod(
                    FoundationModificationSite::Residue(3),
                    3,
                    1,
                    common_unimod_definition(1).unwrap().mass_delta,
                ),
            ],
        };
        let units = true_units(&peptide).unwrap();
        assert!(units[0].n_terminal_acetyl);
        assert_eq!(units[0].residue_unimod_id, Some(4));
        assert_eq!(units[1].residue_unimod_id, Some(35));
        assert_eq!(units[2].residue_unimod_id, Some(7));
        assert_eq!(units[3].residue_unimod_id, Some(1));
    }

    #[test]
    fn deterministic_k_best_reconstructs_high_scoring_true_path() {
        let peptide = PeptidoformInput {
            sequence: "PEPTIDEK".into(),
            modifications: Vec::new(),
        };
        let record = synthetic_record(peptide.clone(), 2);
        let spectrum = FoundationSpectrum::from_training_record(&record).unwrap();
        let graph = foundation_build_cleavage_graph(&record, &spectrum)
            .unwrap()
            .unwrap();
        let audit = foundation_cleavage_graph_true_path_audit(&graph, &peptide).unwrap();
        assert!(audit.structurally_present);
        let true_edges = audit
            .training_groups
            .iter()
            .map(|group| (group.source_node, group.target_outgoing_index))
            .collect::<HashSet<_>>();
        let scores = graph
            .outgoing
            .iter()
            .enumerate()
            .map(|(source_node, outgoing)| {
                outgoing
                    .iter()
                    .enumerate()
                    .map(|(edge_index, _)| {
                        if true_edges.contains(&(source_node, edge_index)) {
                            10.0
                        } else {
                            -10.0
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let candidates =
            foundation_cleavage_graph_k_best_from_edge_scores(&graph, &scores).unwrap();
        assert_eq!(candidates.first().unwrap().peptide, peptide);
    }
}
