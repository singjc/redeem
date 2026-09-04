//! Isolated spectrum-conditioned cleavage-mass graph proposal branch.
//!
//! v0.13.18 retains the v0.13.17 anchor-and-bridge graph construction but
//! replaces independent local outgoing-edge classification with a contextual,
//! globally normalized source-to-sink path model. Exact DAG forward/backward
//! dynamic programming computes the path partition and edge marginals; training
//! uses the exact structured-NLL gradient while deterministic k-best decoding
//! ranks paths by unnormalized contextual edge energies.

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
use std::sync::OnceLock;

/// Isolated parameter namespace for v0.13.16.
pub const FOUNDATION_CLEAVAGE_GRAPH_NAMESPACE_V01316: &str = "cleavage_graph";
/// Fixed training objective identifier.
pub const FOUNDATION_CLEAVAGE_GRAPH_OBJECTIVE_V01316: &str = "outgoing_edge_cross_entropy_v01316";
/// Fixed v0.13.17 objective identifier for the anchor-and-bridge redesign.
pub const FOUNDATION_CLEAVAGE_GRAPH_OBJECTIVE_V01317: &str =
    "outgoing_edge_cross_entropy_anchor_bridge_v01317";
/// Fresh isolated namespace for the v0.13.18 structured scorer.
pub const FOUNDATION_CLEAVAGE_GRAPH_STRUCTURED_NAMESPACE_V01318: &str = "cleavage_graph_structured";
/// Fixed globally normalized path objective for v0.13.18.
pub const FOUNDATION_CLEAVAGE_GRAPH_STRUCTURED_OBJECTIVE_V01318: &str =
    "global_path_nll_contextual_v01318";
/// Context-augmented edge feature width used by v0.13.18.
pub const FOUNDATION_CLEAVAGE_GRAPH_STRUCTURED_FEATURE_DIM_V01318: usize = 58;
/// Hidden width of the first contextual structured scorer.
pub const FOUNDATION_CLEAVAGE_GRAPH_STRUCTURED_HIDDEN_DIM_V01318: usize = 64;
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
/// Longest anchor-to-anchor chemical bridge admitted by the structural redesign.
///
/// Three edges can restore up to two consecutive unobserved cleavage nodes while
/// keeping every inferred node bracketed by observed/source/sink anchors.
pub const FOUNDATION_CLEAVAGE_GRAPH_MAX_ANCHOR_BRIDGE_EDGES_V01317: usize = 3;

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
    /// Decoder path score (local log probability for v0.13.16/17; raw structured energy for v0.13.18).
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

/// Aggregate metrics from one exact structured-path loss evaluation.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct FoundationCleavageGraphStructuredLossStats {
    /// Mean exact `log Z - score(true path)` over structurally present graphs.
    pub mean_nll: f64,
    /// Number of structurally present graphs contributing to the loss.
    pub graphs: usize,
    /// Number of directed edges scored across those graphs.
    pub edges: usize,
}

/// One structured decode plus exact partition diagnostics.
#[derive(Debug, Clone, PartialEq)]
pub struct FoundationCleavageGraphStructuredDecode {
    /// Deterministic source-to-sink candidates ranked by raw structured path score.
    pub candidates: Vec<FoundationCleavageGraphCandidate>,
    /// Exact log partition over all valid source-to-sink graph paths.
    pub log_partition: f64,
    /// Raw score of the known true path when a complete structural audit is supplied.
    pub true_path_score: Option<f64>,
}

/// v0.13.18 contextual edge-energy model in a fresh isolated namespace.
pub struct PeptideSpectrumCleavageGraphStructuredScorer {
    input: Linear,
    hidden: Linear,
    output: Linear,
}

impl PeptideSpectrumCleavageGraphStructuredScorer {
    /// Construct the structured scorer under `cleavage_graph_structured.*`.
    pub fn new(vb: VarBuilder<'_>) -> Result<Self> {
        let vb = vb.pp(FOUNDATION_CLEAVAGE_GRAPH_STRUCTURED_NAMESPACE_V01318);
        let input = nn::linear(
            FOUNDATION_CLEAVAGE_GRAPH_STRUCTURED_FEATURE_DIM_V01318,
            FOUNDATION_CLEAVAGE_GRAPH_STRUCTURED_HIDDEN_DIM_V01318,
            vb.pp("contextual_mlp.input"),
        )?;
        let hidden = nn::linear(
            FOUNDATION_CLEAVAGE_GRAPH_STRUCTURED_HIDDEN_DIM_V01318,
            FOUNDATION_CLEAVAGE_GRAPH_STRUCTURED_HIDDEN_DIM_V01318,
            vb.pp("contextual_mlp.hidden"),
        )?;
        let output = nn::linear(
            FOUNDATION_CLEAVAGE_GRAPH_STRUCTURED_HIDDEN_DIM_V01318,
            1,
            vb.pp("contextual_mlp.output"),
        )?;
        Ok(Self {
            input,
            hidden,
            output,
        })
    }

    /// Score a flat `[edge, contextual_feature]` tensor and return raw edge energies.
    pub fn forward(&self, features: &Tensor) -> Result<Tensor> {
        let (_, feature_dim) = features.dims2()?;
        if feature_dim != FOUNDATION_CLEAVAGE_GRAPH_STRUCTURED_FEATURE_DIM_V01318 {
            candle_core::bail!(
                "structured cleavage-graph feature width {feature_dim} does not match fixed {}",
                FOUNDATION_CLEAVAGE_GRAPH_STRUCTURED_FEATURE_DIM_V01318
            );
        }
        let hidden = self.input.forward(features)?.relu()?;
        let hidden = self.hidden.forward(&hidden)?.relu()?;
        self.output.forward(&hidden)?.squeeze(1)
    }
}

/// Validate that v0.13.18 variables are isolated to `cleavage_graph_structured.*`.
pub fn validate_cleavage_graph_structured_namespace(varmap: &VarMap) -> Result<()> {
    let data = varmap.data().lock().map_err(|_| {
        candle_core::Error::Msg("structured cleavage-graph VarMap lock poisoned".into())
    })?;
    if data.is_empty() {
        candle_core::bail!("structured cleavage-graph VarMap contains no variables");
    }
    let prefix = format!("{FOUNDATION_CLEAVAGE_GRAPH_STRUCTURED_NAMESPACE_V01318}.");
    if let Some(name) = data.keys().find(|name| !name.starts_with(&prefix)) {
        candle_core::bail!(
            "structured cleavage-graph variable '{name}' is outside expected namespace '{prefix}'"
        );
    }
    Ok(())
}

#[derive(Debug, Clone)]
struct FoundationCleavageGraphStructuredContext {
    incoming_degree: Vec<usize>,
    min_hops_from_source: Vec<usize>,
    min_hops_to_sink: Vec<usize>,
    forward_support_context: Vec<f64>,
    backward_support_context: Vec<f64>,
    forward_log_path_count: Vec<f64>,
    backward_log_path_count: Vec<f64>,
}

fn foundation_cleavage_graph_structured_context(
    graph: &FoundationCleavageGraph,
) -> FoundationCleavageGraphStructuredContext {
    let n = graph.nodes.len();
    let mut incoming_degree = vec![0usize; n];
    for outgoing in &graph.outgoing {
        for edge in outgoing {
            incoming_degree[edge.target_node] += 1;
        }
    }

    let unreachable = usize::MAX / 4;
    let mut min_hops_from_source = vec![unreachable; n];
    if n > 0 {
        min_hops_from_source[0] = 0;
    }
    for source in 0..n {
        if min_hops_from_source[source] == unreachable {
            continue;
        }
        let next_hops = min_hops_from_source[source].saturating_add(1);
        for edge in &graph.outgoing[source] {
            min_hops_from_source[edge.target_node] =
                min_hops_from_source[edge.target_node].min(next_hops);
        }
    }

    let mut min_hops_to_sink = vec![unreachable; n];
    if n > 0 {
        min_hops_to_sink[graph.sink_index()] = 0;
    }
    for source in (0..n.saturating_sub(1)).rev() {
        let mut best = unreachable;
        for edge in &graph.outgoing[source] {
            if min_hops_to_sink[edge.target_node] != unreachable {
                best = best.min(min_hops_to_sink[edge.target_node].saturating_add(1));
            }
        }
        min_hops_to_sink[source] = best;
    }

    let mut forward_support_context = vec![f64::NEG_INFINITY; n];
    let mut forward_log_path_count = vec![f64::NEG_INFINITY; n];
    if n > 0 {
        forward_support_context[0] = 0.0;
        forward_log_path_count[0] = 0.0;
    }
    for source in 0..n {
        if !forward_support_context[source].is_finite() {
            continue;
        }
        for edge in &graph.outgoing[source] {
            let support = graph.nodes[edge.target_node].total_support() as f64;
            let candidate = 0.65 * forward_support_context[source] + 0.35 * support;
            forward_support_context[edge.target_node] =
                forward_support_context[edge.target_node].max(candidate);
            forward_log_path_count[edge.target_node] = logaddexp(
                forward_log_path_count[edge.target_node],
                forward_log_path_count[source],
            );
        }
    }

    let mut backward_support_context = vec![f64::NEG_INFINITY; n];
    let mut backward_log_path_count = vec![f64::NEG_INFINITY; n];
    if n > 0 {
        backward_support_context[graph.sink_index()] = 0.0;
        backward_log_path_count[graph.sink_index()] = 0.0;
    }
    for source in (0..n.saturating_sub(1)).rev() {
        for edge in &graph.outgoing[source] {
            if backward_support_context[edge.target_node].is_finite() {
                let support = graph.nodes[source].total_support() as f64;
                let candidate = 0.65 * backward_support_context[edge.target_node] + 0.35 * support;
                backward_support_context[source] = backward_support_context[source].max(candidate);
                backward_log_path_count[source] = logaddexp(
                    backward_log_path_count[source],
                    backward_log_path_count[edge.target_node],
                );
            }
        }
    }

    for values in [&mut forward_support_context, &mut backward_support_context] {
        for value in values.iter_mut() {
            if !value.is_finite() {
                *value = 0.0;
            }
        }
    }

    FoundationCleavageGraphStructuredContext {
        incoming_degree,
        min_hops_from_source,
        min_hops_to_sink,
        forward_support_context,
        backward_support_context,
        forward_log_path_count,
        backward_log_path_count,
    }
}

/// Context-augmented v0.13.18 edge feature vector.
pub fn foundation_cleavage_graph_structured_edge_features(
    graph: &FoundationCleavageGraph,
    edge: &FoundationCleavageGraphEdge,
) -> [f32; FOUNDATION_CLEAVAGE_GRAPH_STRUCTURED_FEATURE_DIM_V01318] {
    let context = foundation_cleavage_graph_structured_context(graph);
    foundation_cleavage_graph_structured_edge_features_with_context(graph, &context, edge)
}

fn foundation_cleavage_graph_structured_edge_features_with_context(
    graph: &FoundationCleavageGraph,
    context: &FoundationCleavageGraphStructuredContext,
    edge: &FoundationCleavageGraphEdge,
) -> [f32; FOUNDATION_CLEAVAGE_GRAPH_STRUCTURED_FEATURE_DIM_V01318] {
    let base = foundation_cleavage_graph_edge_features(graph, edge);
    let mut features = [0.0f32; FOUNDATION_CLEAVAGE_GRAPH_STRUCTURED_FEATURE_DIM_V01318];
    features[..FOUNDATION_CLEAVAGE_GRAPH_FEATURE_DIM_V01316].copy_from_slice(&base);
    let source = edge.source_node;
    let target = edge.target_node;
    let source_out = graph.outgoing[source].len();
    let target_out = graph.outgoing[target].len();
    let hop = |value: usize| -> f32 {
        if value >= usize::MAX / 8 {
            1.0
        } else {
            (value as f32 / 64.0).min(1.0)
        }
    };
    let path_count = |value: f64| -> f32 {
        if value.is_finite() {
            (value / 20.0).clamp(0.0, 1.0) as f32
        } else {
            0.0
        }
    };
    features[42] = (context.incoming_degree[source] as f32 / 64.0).min(1.0);
    features[43] = (source_out as f32 / 64.0).min(1.0);
    features[44] = (context.incoming_degree[target] as f32 / 64.0).min(1.0);
    features[45] = (target_out as f32 / 64.0).min(1.0);
    features[46] = hop(context.min_hops_from_source[source]);
    features[47] = hop(context.min_hops_from_source[target]);
    features[48] = hop(context.min_hops_to_sink[source]);
    features[49] = hop(context.min_hops_to_sink[target]);
    features[50] = context.forward_support_context[source].clamp(0.0, 1.0) as f32;
    features[51] = context.forward_support_context[target].clamp(0.0, 1.0) as f32;
    features[52] = context.backward_support_context[source].clamp(0.0, 1.0) as f32;
    features[53] = context.backward_support_context[target].clamp(0.0, 1.0) as f32;
    features[54] = path_count(context.forward_log_path_count[source]);
    features[55] = path_count(context.backward_log_path_count[target]);
    features[56] = ((graph.residue_mass_total - graph.nodes[target].mass_da)
        / graph.residue_mass_total.max(f64::EPSILON))
    .clamp(0.0, 1.0) as f32;
    features[57] = if graph.nodes[source].total_support() == 0.0
        && graph.nodes[target].total_support() == 0.0
    {
        1.0
    } else {
        0.0
    };
    features
}

/// Exact globally normalized structured-path NLL gradient for one graph batch.
///
/// Forward/backward path dynamic programming is performed in stable `f64` on
/// detached edge scores. For a log-linear DAG path model the exact derivative
/// with respect to each edge energy is `p(edge|graph) - 1[edge in true path]`.
/// Multiplying those detached coefficients by the original Candle logits gives
/// a compact surrogate scalar whose autograd gradient is exactly the structured
/// NLL gradient while avoiding thousands of tiny differentiable DP operations.
pub fn foundation_cleavage_graph_structured_loss(
    model: &PeptideSpectrumCleavageGraphStructuredScorer,
    examples: &[(
        &FoundationCleavageGraph,
        &FoundationCleavageGraphTruePathAudit,
    )],
    device: &Device,
) -> Result<Option<(Tensor, FoundationCleavageGraphStructuredLossStats)>> {
    let examples = examples
        .iter()
        .copied()
        .filter(|(_, audit)| audit.structurally_present)
        .collect::<Vec<_>>();
    if examples.is_empty() {
        return Ok(None);
    }

    let total_edges = examples
        .iter()
        .map(|(graph, _)| graph.edge_count())
        .sum::<usize>();
    if total_edges == 0 {
        return Ok(None);
    }
    let mut features = Vec::<f32>::with_capacity(
        total_edges * FOUNDATION_CLEAVAGE_GRAPH_STRUCTURED_FEATURE_DIM_V01318,
    );
    let mut graph_ranges = Vec::<(usize, usize)>::with_capacity(examples.len());
    for (graph, _) in &examples {
        let start = features.len() / FOUNDATION_CLEAVAGE_GRAPH_STRUCTURED_FEATURE_DIM_V01318;
        let context = foundation_cleavage_graph_structured_context(graph);
        for outgoing in &graph.outgoing {
            for edge in outgoing {
                features.extend_from_slice(
                    &foundation_cleavage_graph_structured_edge_features_with_context(
                        graph, &context, edge,
                    ),
                );
            }
        }
        let end = features.len() / FOUNDATION_CLEAVAGE_GRAPH_STRUCTURED_FEATURE_DIM_V01318;
        graph_ranges.push((start, end));
    }

    let feature_tensor = Tensor::from_vec(
        features,
        (
            total_edges,
            FOUNDATION_CLEAVAGE_GRAPH_STRUCTURED_FEATURE_DIM_V01318,
        ),
        device,
    )?;
    let logits = model.forward(&feature_tensor)?;
    let detached_scores = logits.to_vec1::<f32>()?;
    let mut coefficients = vec![0.0f32; total_edges];
    let mut nll_sum = 0.0f64;

    for (example_index, &(graph, audit)) in examples.iter().enumerate() {
        let (start, end) = graph_ranges[example_index];
        let nested = reshape_flat_edge_scores(graph, &detached_scores[start..end])?;
        let stats = structured_path_statistics(graph, audit, &nested)?;
        nll_sum += stats.nll;
        let mut cursor = start;
        for row in stats.gradient_coefficients {
            for coefficient in row {
                coefficients[cursor] = coefficient as f32;
                cursor += 1;
            }
        }
        if cursor != end {
            candle_core::bail!("structured cleavage-graph coefficient arity mismatch");
        }
    }

    let coefficient_tensor = Tensor::from_vec(coefficients, total_edges, device)?;
    let surrogate = (&logits * &coefficient_tensor)?
        .sum_all()?
        .affine(1.0 / examples.len() as f64, 0.0)?;
    let mean_nll = nll_sum / examples.len() as f64;
    // Shift only the scalar value, not the derivative, so logging sees the exact
    // structured NLL while autograd follows the exact analytic NLL gradient.
    let surrogate_value = f64::from(surrogate.to_scalar::<f32>()?);
    let loss = surrogate.affine(1.0, mean_nll - surrogate_value)?;
    Ok(Some((
        loss,
        FoundationCleavageGraphStructuredLossStats {
            mean_nll,
            graphs: examples.len(),
            edges: total_edges,
        },
    )))
}

/// Decode v0.13.18 paths and return exact partition/true-path diagnostics.
pub fn foundation_cleavage_graph_structured_decode(
    model: &PeptideSpectrumCleavageGraphStructuredScorer,
    graph: &FoundationCleavageGraph,
    audit: Option<&FoundationCleavageGraphTruePathAudit>,
    device: &Device,
) -> Result<FoundationCleavageGraphStructuredDecode> {
    let edge_scores = foundation_cleavage_graph_structured_edge_scores(model, graph, device)?;
    let log_partition =
        structured_log_partition(graph, &edge_scores).map_err(candle_core::Error::Msg)?;
    let true_path_score = match audit {
        Some(audit) if audit.structurally_present => Some(
            structured_true_path_score(graph, audit, &edge_scores)
                .map_err(candle_core::Error::Msg)?,
        ),
        _ => None,
    };
    let candidates =
        foundation_cleavage_graph_structured_k_best_from_edge_scores(graph, &edge_scores)
            .map_err(candle_core::Error::Msg)?;
    Ok(FoundationCleavageGraphStructuredDecode {
        candidates,
        log_partition,
        true_path_score,
    })
}

/// Decode deterministic v0.13.18 k-best paths by sums of raw contextual energies.
pub fn foundation_cleavage_graph_structured_k_best_candidates(
    model: &PeptideSpectrumCleavageGraphStructuredScorer,
    graph: &FoundationCleavageGraph,
    device: &Device,
) -> Result<Vec<FoundationCleavageGraphCandidate>> {
    Ok(foundation_cleavage_graph_structured_decode(model, graph, None, device)?.candidates)
}

/// Deterministic raw-energy k-best decoder used by v0.13.18 and unit tests.
pub fn foundation_cleavage_graph_structured_k_best_from_edge_scores(
    graph: &FoundationCleavageGraph,
    edge_scores: &[Vec<f64>],
) -> std::result::Result<Vec<FoundationCleavageGraphCandidate>, String> {
    if edge_scores.len() != graph.outgoing.len() {
        return Err("structured cleavage-graph edge-score/source-node arity mismatch".into());
    }
    for (scores, outgoing) in edge_scores.iter().zip(&graph.outgoing) {
        if scores.len() != outgoing.len() {
            return Err("structured cleavage-graph edge-score/outgoing arity mismatch".into());
        }
    }

    #[derive(Debug, Clone, PartialEq)]
    struct PathState {
        score: f64,
        units: Vec<FoundationCleavageGraphUnit>,
    }

    let mut paths = vec![Vec::<PathState>::new(); graph.nodes.len()];
    if paths.is_empty() {
        return Ok(Vec::new());
    }
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
                    score: state.score + edge_scores[source_node][edge_index],
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

fn foundation_cleavage_graph_structured_edge_scores(
    model: &PeptideSpectrumCleavageGraphStructuredScorer,
    graph: &FoundationCleavageGraph,
    device: &Device,
) -> Result<Vec<Vec<f64>>> {
    let edge_count = graph.edge_count();
    if edge_count == 0 {
        return Ok(vec![Vec::new(); graph.nodes.len()]);
    }
    let context = foundation_cleavage_graph_structured_context(graph);
    let mut features = Vec::<f32>::with_capacity(
        edge_count * FOUNDATION_CLEAVAGE_GRAPH_STRUCTURED_FEATURE_DIM_V01318,
    );
    let mut lengths = Vec::<usize>::with_capacity(graph.outgoing.len());
    for outgoing in &graph.outgoing {
        lengths.push(outgoing.len());
        for edge in outgoing {
            features.extend_from_slice(
                &foundation_cleavage_graph_structured_edge_features_with_context(
                    graph, &context, edge,
                ),
            );
        }
    }
    let tensor = Tensor::from_vec(
        features,
        (
            edge_count,
            FOUNDATION_CLEAVAGE_GRAPH_STRUCTURED_FEATURE_DIM_V01318,
        ),
        device,
    )?;
    let flat_scores = model.forward(&tensor)?.to_vec1::<f32>()?;
    reshape_flat_edge_scores(graph, &flat_scores)
}

fn reshape_flat_edge_scores<T: Copy + Into<f64>>(
    graph: &FoundationCleavageGraph,
    flat_scores: &[T],
) -> Result<Vec<Vec<f64>>> {
    if flat_scores.len() != graph.edge_count() {
        candle_core::bail!("structured cleavage-graph flat edge-score arity mismatch");
    }
    let mut cursor = 0usize;
    let mut scores = Vec::<Vec<f64>>::with_capacity(graph.outgoing.len());
    for outgoing in &graph.outgoing {
        let length = outgoing.len();
        scores.push(
            flat_scores[cursor..cursor + length]
                .iter()
                .copied()
                .map(Into::into)
                .collect(),
        );
        cursor += length;
    }
    Ok(scores)
}

#[derive(Debug, Clone)]
struct FoundationCleavageGraphStructuredPathStats {
    nll: f64,
    gradient_coefficients: Vec<Vec<f64>>,
}

fn structured_path_statistics(
    graph: &FoundationCleavageGraph,
    audit: &FoundationCleavageGraphTruePathAudit,
    edge_scores: &[Vec<f64>],
) -> Result<FoundationCleavageGraphStructuredPathStats> {
    if !audit.structurally_present {
        candle_core::bail!("structured path statistics require a complete true path");
    }
    let (alpha, beta, log_partition) =
        structured_forward_backward(graph, edge_scores).map_err(candle_core::Error::Msg)?;
    let true_score =
        structured_true_path_score(graph, audit, edge_scores).map_err(candle_core::Error::Msg)?;
    let true_edges = audit
        .training_groups
        .iter()
        .map(|group| (group.source_node, group.target_outgoing_index))
        .collect::<HashSet<_>>();
    let mut gradient_coefficients = Vec::<Vec<f64>>::with_capacity(graph.outgoing.len());
    for (source, outgoing) in graph.outgoing.iter().enumerate() {
        let mut row = Vec::with_capacity(outgoing.len());
        for (edge_index, edge) in outgoing.iter().enumerate() {
            let log_marginal =
                alpha[source] + edge_scores[source][edge_index] + beta[edge.target_node]
                    - log_partition;
            let marginal = if log_marginal.is_finite() {
                log_marginal.exp().clamp(0.0, 1.0)
            } else {
                0.0
            };
            row.push(
                marginal
                    - if true_edges.contains(&(source, edge_index)) {
                        1.0
                    } else {
                        0.0
                    },
            );
        }
        gradient_coefficients.push(row);
    }
    Ok(FoundationCleavageGraphStructuredPathStats {
        nll: (log_partition - true_score).max(0.0),
        gradient_coefficients,
    })
}

fn structured_true_path_score(
    graph: &FoundationCleavageGraph,
    audit: &FoundationCleavageGraphTruePathAudit,
    edge_scores: &[Vec<f64>],
) -> std::result::Result<f64, String> {
    if !audit.structurally_present {
        return Err("true-path score requested for structurally incomplete graph".into());
    }
    let mut score = 0.0f64;
    for group in &audit.training_groups {
        let outgoing = graph
            .outgoing
            .get(group.source_node)
            .ok_or_else(|| "true-path source node out of range".to_string())?;
        if group.target_outgoing_index >= outgoing.len()
            || group.target_outgoing_index >= edge_scores[group.source_node].len()
        {
            return Err("true-path outgoing edge index out of range".into());
        }
        score += edge_scores[group.source_node][group.target_outgoing_index];
    }
    Ok(score)
}

fn structured_log_partition(
    graph: &FoundationCleavageGraph,
    edge_scores: &[Vec<f64>],
) -> std::result::Result<f64, String> {
    let (_, _, log_partition) = structured_forward_backward(graph, edge_scores)?;
    Ok(log_partition)
}

fn structured_forward_backward(
    graph: &FoundationCleavageGraph,
    edge_scores: &[Vec<f64>],
) -> std::result::Result<(Vec<f64>, Vec<f64>, f64), String> {
    if edge_scores.len() != graph.outgoing.len() {
        return Err("structured cleavage-graph edge-score/source-node arity mismatch".into());
    }
    let n = graph.nodes.len();
    if n == 0 {
        return Err("structured cleavage graph has no nodes".into());
    }
    let mut alpha = vec![f64::NEG_INFINITY; n];
    alpha[0] = 0.0;
    for source in 0..n {
        if edge_scores[source].len() != graph.outgoing[source].len() {
            return Err("structured cleavage-graph edge-score/outgoing arity mismatch".into());
        }
        if !alpha[source].is_finite() {
            continue;
        }
        for (edge_index, edge) in graph.outgoing[source].iter().enumerate() {
            let candidate = alpha[source] + edge_scores[source][edge_index];
            alpha[edge.target_node] = logaddexp(alpha[edge.target_node], candidate);
        }
    }
    let log_partition = alpha[graph.sink_index()];
    if !log_partition.is_finite() {
        return Err("structured cleavage graph has no finite source-to-sink path".into());
    }

    let mut beta = vec![f64::NEG_INFINITY; n];
    beta[graph.sink_index()] = 0.0;
    for source in (0..n.saturating_sub(1)).rev() {
        let mut value = f64::NEG_INFINITY;
        for (edge_index, edge) in graph.outgoing[source].iter().enumerate() {
            if beta[edge.target_node].is_finite() {
                value = logaddexp(
                    value,
                    edge_scores[source][edge_index] + beta[edge.target_node],
                );
            }
        }
        beta[source] = value;
    }
    Ok((alpha, beta, log_partition))
}

fn logaddexp(left: f64, right: f64) -> f64 {
    if !left.is_finite() {
        return right;
    }
    if !right.is_finite() {
        return left;
    }
    let maximum = left.max(right);
    maximum + ((left - maximum).exp() + (right - maximum).exp()).ln()
}

/// Build the deterministic v0.13.17 anchor-and-bridge graph from spectrum + precursor only.
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

    let source = FoundationCleavageGraphNode {
        mass_da: 0.0,
        n_terminal_support: 1.0,
        c_terminal_support: 0.0,
        is_source: true,
        is_sink: false,
    };
    let sink = FoundationCleavageGraphNode {
        mass_da: residue_mass_total,
        n_terminal_support: 0.0,
        c_terminal_support: 1.0,
        is_source: false,
        is_sink: true,
    };

    // v0.13.16 required direct observed-fragment support at every cleavage, which
    // produced only 40/128 complete true paths even though 75.8% of true nodes
    // were individually present. v0.13.17 keeps those observed nodes as immutable
    // anchors and fills only chemically exact short gaps bracketed by anchor pairs.
    // The inferred nodes carry zero direct spectrum support, so the edge scorer can
    // distinguish them naturally from observed evidence without target identity.
    let mut anchors = Vec::with_capacity(clustered.len() + 2);
    anchors.push(source);
    anchors.extend(clustered.iter().copied());
    anchors.push(sink);
    let normal_bridge_patterns = cleavage_graph_bridge_patterns(false)?;
    let source_bridge_patterns = cleavage_graph_bridge_patterns(true)?;
    let mut bridge_nodes = Vec::<FoundationCleavageGraphNode>::new();
    for (left_index, left) in anchors.iter().copied().enumerate().take(anchors.len() - 1) {
        let patterns = if left.is_source {
            source_bridge_patterns
        } else {
            normal_bridge_patterns
        };
        for right in anchors.iter().copied().skip(left_index + 1) {
            let gap = right.mass_da - left.mass_da;
            if gap <= 0.0 {
                continue;
            }
            let low = gap - FOUNDATION_CLEAVAGE_GRAPH_MASS_TOLERANCE_DA_V01316;
            let high = gap + FOUNDATION_CLEAVAGE_GRAPH_MASS_TOLERANCE_DA_V01316;
            let first = lower_bound_bridge_pattern(patterns, low);
            for pattern in patterns.iter().skip(first) {
                if pattern.total_mass > high {
                    break;
                }
                for &relative_mass in
                    pattern.intermediate_masses[..pattern.intermediate_count].iter()
                {
                    let mass_da = left.mass_da + relative_mass;
                    if mass_da > FOUNDATION_CLEAVAGE_GRAPH_MASS_TOLERANCE_DA_V01316
                        && mass_da
                            < residue_mass_total
                                - FOUNDATION_CLEAVAGE_GRAPH_MASS_TOLERANCE_DA_V01316
                    {
                        bridge_nodes.push(FoundationCleavageGraphNode {
                            mass_da,
                            n_terminal_support: 0.0,
                            c_terminal_support: 0.0,
                            is_source: false,
                            is_sink: false,
                        });
                    }
                }
            }
        }
    }
    clustered.extend(bridge_nodes);
    clustered = merge_cleavage_graph_nodes(clustered);

    let mut nodes = Vec::with_capacity(clustered.len() + 2);
    nodes.push(source);
    nodes.extend(clustered);
    nodes.push(sink);

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

#[derive(Debug, Clone, Copy, PartialEq)]
struct FoundationCleavageGraphBridgePattern {
    total_mass: f64,
    intermediate_masses: [f64; 2],
    intermediate_count: usize,
}

fn cleavage_graph_bridge_patterns(
    include_n_terminal_acetyl: bool,
) -> std::result::Result<&'static [FoundationCleavageGraphBridgePattern], String> {
    static NORMAL: OnceLock<
        std::result::Result<Vec<FoundationCleavageGraphBridgePattern>, String>,
    > = OnceLock::new();
    static SOURCE: OnceLock<
        std::result::Result<Vec<FoundationCleavageGraphBridgePattern>, String>,
    > = OnceLock::new();
    let cached = if include_n_terminal_acetyl {
        SOURCE.get_or_init(|| build_cleavage_graph_bridge_patterns(true))
    } else {
        NORMAL.get_or_init(|| build_cleavage_graph_bridge_patterns(false))
    };
    match cached {
        Ok(patterns) => Ok(patterns.as_slice()),
        Err(error) => Err(error.clone()),
    }
}

fn build_cleavage_graph_bridge_patterns(
    include_n_terminal_acetyl: bool,
) -> std::result::Result<Vec<FoundationCleavageGraphBridgePattern>, String> {
    let first_units = allowed_units(include_n_terminal_acetyl)?;
    let following_units = allowed_units(false)?;
    let mut patterns = Vec::<FoundationCleavageGraphBridgePattern>::new();

    for first in first_units {
        let first_mass = first.mass_da()?;
        for second in following_units.iter().copied() {
            let second_mass = second.mass_da()?;
            patterns.push(FoundationCleavageGraphBridgePattern {
                total_mass: first_mass + second_mass,
                intermediate_masses: [first_mass, 0.0],
                intermediate_count: 1,
            });
            if FOUNDATION_CLEAVAGE_GRAPH_MAX_ANCHOR_BRIDGE_EDGES_V01317 >= 3 {
                for third in following_units.iter().copied() {
                    let third_mass = third.mass_da()?;
                    patterns.push(FoundationCleavageGraphBridgePattern {
                        total_mass: first_mass + second_mass + third_mass,
                        intermediate_masses: [first_mass, first_mass + second_mass],
                        intermediate_count: 2,
                    });
                }
            }
        }
    }
    patterns.sort_by(|left, right| {
        left.total_mass
            .total_cmp(&right.total_mass)
            .then_with(|| left.intermediate_count.cmp(&right.intermediate_count))
            .then_with(|| left.intermediate_masses[0].total_cmp(&right.intermediate_masses[0]))
            .then_with(|| left.intermediate_masses[1].total_cmp(&right.intermediate_masses[1]))
    });
    patterns.dedup_by(|left, right| {
        left.total_mass.to_bits() == right.total_mass.to_bits()
            && left.intermediate_count == right.intermediate_count
            && left.intermediate_masses[0].to_bits() == right.intermediate_masses[0].to_bits()
            && left.intermediate_masses[1].to_bits() == right.intermediate_masses[1].to_bits()
    });
    Ok(patterns)
}

fn lower_bound_bridge_pattern(
    patterns: &[FoundationCleavageGraphBridgePattern],
    total_mass: f64,
) -> usize {
    let mut low = 0usize;
    let mut high = patterns.len();
    while low < high {
        let middle = low + (high - low) / 2;
        if patterns[middle].total_mass < total_mass {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    low
}

fn merge_cleavage_graph_nodes(
    mut nodes: Vec<FoundationCleavageGraphNode>,
) -> Vec<FoundationCleavageGraphNode> {
    nodes.sort_by(|left, right| left.mass_da.total_cmp(&right.mass_da));
    let mut merged = Vec::<FoundationCleavageGraphNode>::with_capacity(nodes.len());
    for node in nodes {
        if let Some(last) = merged.last_mut() {
            if (node.mass_da - last.mass_da).abs()
                <= FOUNDATION_CLEAVAGE_GRAPH_MASS_TOLERANCE_DA_V01316
            {
                if node.total_support() > last.total_support() {
                    last.mass_da = node.mass_da;
                }
                last.n_terminal_support = last.n_terminal_support.max(node.n_terminal_support);
                last.c_terminal_support = last.c_terminal_support.max(node.c_terminal_support);
                continue;
            }
        }
        merged.push(node);
    }
    merged
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
    fn anchor_bridge_restores_two_consecutive_missing_cleavages() {
        let peptide = PeptidoformInput {
            sequence: "PEPTIDEK".into(),
            modifications: Vec::new(),
        };
        let mut record = synthetic_record(peptide.clone(), 2);
        // synthetic_record emits [b_i, y_i] for each internal cleavage. Remove
        // both orientations for two consecutive true cleavages. The graph builder
        // must recover the missing cumulative masses only from chemistry between
        // the neighboring observed anchors; peptide identity is not an input.
        record.observed_spectrum_peaks = record
            .observed_spectrum_peaks
            .into_iter()
            .enumerate()
            .filter_map(|(index, peak)| (!matches!(index, 4 | 5 | 6 | 7)).then_some(peak))
            .collect();
        let spectrum = FoundationSpectrum::from_training_record(&record).unwrap();
        let graph = foundation_build_cleavage_graph(&record, &spectrum)
            .unwrap()
            .unwrap();
        let audit = foundation_cleavage_graph_true_path_audit(&graph, &peptide).unwrap();
        assert!(audit.structurally_present);
        assert_eq!(audit.present_nodes, audit.total_nodes);
        assert_eq!(audit.present_edges, audit.total_edges);
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

    #[test]
    fn structured_raw_energy_k_best_reconstructs_high_scoring_true_path() {
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
                            4.0
                        } else {
                            -4.0
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let candidates =
            foundation_cleavage_graph_structured_k_best_from_edge_scores(&graph, &scores).unwrap();
        assert_eq!(candidates.first().unwrap().peptide, peptide);
        let stats = structured_path_statistics(&graph, &audit, &scores).unwrap();
        assert!(stats.nll.is_finite());
        assert!(stats.nll >= 0.0);
    }

    #[test]
    fn structured_context_features_are_finite_and_fixed_width() {
        let peptide = PeptidoformInput {
            sequence: "PEPTIDEK".into(),
            modifications: Vec::new(),
        };
        let record = synthetic_record(peptide, 2);
        let spectrum = FoundationSpectrum::from_training_record(&record).unwrap();
        let graph = foundation_build_cleavage_graph(&record, &spectrum)
            .unwrap()
            .unwrap();
        let edge = graph
            .outgoing
            .iter()
            .find_map(|outgoing| outgoing.first())
            .unwrap();
        let features = foundation_cleavage_graph_structured_edge_features(&graph, edge);
        assert_eq!(
            features.len(),
            FOUNDATION_CLEAVAGE_GRAPH_STRUCTURED_FEATURE_DIM_V01318
        );
        assert!(features.iter().all(|value| value.is_finite()));
    }
}
