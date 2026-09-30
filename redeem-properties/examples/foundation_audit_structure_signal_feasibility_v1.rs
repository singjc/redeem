//! TRAIN-only Stage A inventory for structure-signal feasibility.
//!
//! This audit deliberately performs no model training and does not inspect measured
//! CCS / ion-mobility labels. It materializes the exact benchmark TRAIN partition,
//! deduplicates peptidoform+charge identities, and inventories how much chemical
//! information ReDeeM already has for the observed PTMs.
//!
//! The key distinction is intentional:
//! - the current foundation graph can featurize unresolved PTMs with pseudo-mass nodes;
//! - a 3D conformer workflow instead needs explicit attachment/topology information.
//!
//! Stage A therefore reports several chemistry tiers without claiming that a 3D
//! molecule builder or conformer generator already exists.

use anyhow::{bail, Context, Result};
use redeem_properties::foundation::{
    common_unimod_definition, exact_graph_modification_for, foundation_peptidoform_neutral_mass,
    load_foundation_corpus, read_foundation_training_run_config, ElementalComposition,
    FoundationBenchmarkManifest, FoundationModification, FoundationModificationSite,
    FoundationPartition, FoundationTrainingRecord, PeptidoformInput,
};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs::{create_dir_all, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

const AUDIT_VERSION: &str = "structure_signal_feasibility_v1_stage_a";
const MASS_MATCH_TOLERANCE_DA: f64 = 0.002;
const MODIFICATION_TSV_BYTES: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/assets/modification.tsv"
));

#[derive(Debug, Clone)]
struct AssetModification {
    name: String,
    site_spec: String,
    mass_delta: f64,
    composition: String,
    unimod_id: Option<u32>,
    smiles: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SiteCompatibility {
    Exact,
    AmbiguousContext,
    Incompatible,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum ModificationChemistryStatus {
    ExactInternalTopology,
    AssetAttachmentTemplate,
    CompositionOnly,
    CanonicalWithoutSiteChemistry,
    MassOnlyUniqueAssetCandidate,
    MassOnlyAmbiguousAssetCandidates,
    MassOnlyUnmatched,
    InvalidSite,
}

impl ModificationChemistryStatus {
    fn as_str(self) -> &'static str {
        match self {
            Self::ExactInternalTopology => "exact_internal_ptm_local_topology",
            Self::AssetAttachmentTemplate => "asset_attachment_template",
            Self::CompositionOnly => "composition_only",
            Self::CanonicalWithoutSiteChemistry => "canonical_without_site_chemistry",
            Self::MassOnlyUniqueAssetCandidate => "mass_only_unique_asset_candidate",
            Self::MassOnlyAmbiguousAssetCandidates => "mass_only_ambiguous_asset_candidates",
            Self::MassOnlyUnmatched => "mass_only_unmatched",
            Self::InvalidSite => "invalid_site",
        }
    }
}

#[derive(Debug, Clone)]
struct ModificationAssessment {
    ptm_label: String,
    unimod_id: Option<u32>,
    site_label: String,
    residue: char,
    mass_delta: f64,
    registry_name: String,
    registry_composition: String,
    internal_exact_topology: bool,
    exact_asset_rows: usize,
    ambiguous_asset_rows: usize,
    asset_names: Vec<String>,
    asset_compositions: Vec<String>,
    asset_smiles: Vec<String>,
    ambiguous_asset_names: Vec<String>,
    ambiguous_asset_compositions: Vec<String>,
    ambiguous_asset_smiles: Vec<String>,
    mass_candidate_unimod_ids: Vec<u32>,
    status: ModificationChemistryStatus,
    failure_reason: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum IdentityChemistryStatus {
    Unmodified,
    CurrentExactTopology,
    ExistingAttachmentTemplateCandidate,
    CompositionOnlyExtensionRequired,
    AmbiguousOrUnsupported,
}

impl IdentityChemistryStatus {
    fn as_str(self) -> &'static str {
        match self {
            Self::Unmodified => "unmodified",
            Self::CurrentExactTopology => "current_exact_ptm_local_topology",
            Self::ExistingAttachmentTemplateCandidate => "existing_attachment_template_candidate",
            Self::CompositionOnlyExtensionRequired => "composition_only_extension_required",
            Self::AmbiguousOrUnsupported => "ambiguous_or_unsupported",
        }
    }
}

#[derive(Debug, Clone)]
struct IdentityChemistryAssessment {
    status: IdentityChemistryStatus,
    current_exact_topology_ready: bool,
    existing_attachment_template_candidate: bool,
    all_modifications_composition_known: bool,
    failure_reasons: Vec<String>,
    modifications: Vec<ModificationAssessment>,
}

#[derive(Debug, Clone, Default)]
struct NumericAccumulator {
    count: usize,
    sum: f64,
    min: Option<f64>,
    max: Option<f64>,
}

impl NumericAccumulator {
    fn push(&mut self, value: f64) {
        if !value.is_finite() {
            return;
        }
        self.count += 1;
        self.sum += value;
        self.min = Some(self.min.map_or(value, |current| current.min(value)));
        self.max = Some(self.max.map_or(value, |current| current.max(value)));
    }

    fn mean(&self) -> Option<f64> {
        (self.count > 0).then(|| self.sum / self.count as f64)
    }
}

#[derive(Debug, Clone)]
struct IdentityRecord {
    identity_key: String,
    peptidoform: String,
    sequence: String,
    charge: i32,
    sequence_length: usize,
    record_count: usize,
    source_ids: BTreeSet<String>,
    precursor_mz: NumericAccumulator,
    theoretical_neutral_mass: Option<f64>,
    chemistry: IdentityChemistryAssessment,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct PtmInventoryKey {
    ptm_label: String,
    site_label: String,
    residue: char,
}

#[derive(Debug, Clone, Default)]
struct PtmCounts {
    record_occurrences: usize,
    identity_occurrences: usize,
}

#[derive(Debug, Clone, Default)]
struct CoverageCounts {
    records: usize,
    identities: usize,
    unmodified_identities: usize,
    modified_identities: usize,
    current_exact_topology: usize,
    existing_attachment_template_candidate: usize,
    composition_only_extension_required: usize,
    ambiguous_or_unsupported: usize,
}

#[derive(Debug, Clone, Default)]
struct TrainInventory {
    train_records: usize,
    records_with_positive_charge: usize,
    records_missing_charge: usize,
    records_nonpositive_charge: usize,
    records_modified: usize,
    records_unmodified: usize,
    unique_peptidoforms: BTreeSet<String>,
    missing_charge_peptidoforms: BTreeSet<String>,
    nonpositive_charge_peptidoforms: BTreeSet<String>,
    record_charge_counts: BTreeMap<String, usize>,
    record_length_counts: BTreeMap<String, usize>,
    identities: BTreeMap<String, IdentityRecord>,
    record_ptm_counts: BTreeMap<PtmInventoryKey, usize>,
    ptm_assessments: BTreeMap<PtmInventoryKey, ModificationAssessment>,
}

#[derive(Debug, Clone)]
struct AssetChemistryIndex {
    rows: Vec<AssetModification>,
    by_unimod: BTreeMap<u32, Vec<usize>>,
    by_mass_millidalton: BTreeMap<i64, Vec<usize>>,
}

impl AssetChemistryIndex {
    fn load_embedded() -> Result<Self> {
        let mut reader = csv::ReaderBuilder::new()
            .delimiter(b'\t')
            .from_reader(MODIFICATION_TSV_BYTES);
        let headers = reader.headers()?.clone();
        let index = |name: &str| -> Result<usize> {
            headers
                .iter()
                .position(|header| header == name)
                .ok_or_else(|| anyhow::anyhow!("modification.tsv is missing '{name}'"))
        };
        let name_idx = index("mod_name")?;
        let mass_idx = index("unimod_mass")?;
        let composition_idx = index("composition")?;
        let unimod_idx = index("unimod_id")?;
        let smiles_idx = index("smiles")?;
        let mut rows = Vec::new();
        for result in reader.records() {
            let row = result?;
            let name = row.get(name_idx).unwrap_or_default().trim().to_string();
            let site_spec = name
                .split_once('@')
                .map(|(_, site)| site.trim().to_string())
                .unwrap_or_default();
            let mass_delta = row
                .get(mass_idx)
                .unwrap_or_default()
                .trim()
                .parse::<f64>()
                .with_context(|| format!("invalid unimod_mass for '{name}'"))?;
            let unimod_id = row
                .get(unimod_idx)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::parse::<u32>)
                .transpose()
                .with_context(|| format!("invalid unimod_id for '{name}'"))?;
            rows.push(AssetModification {
                name,
                site_spec,
                mass_delta,
                composition: row
                    .get(composition_idx)
                    .unwrap_or_default()
                    .trim()
                    .to_string(),
                unimod_id,
                smiles: row.get(smiles_idx).unwrap_or_default().trim().to_string(),
            });
        }
        if rows.is_empty() {
            bail!("embedded modification.tsv contains no rows");
        }
        let mut by_unimod = BTreeMap::<u32, Vec<usize>>::new();
        let mut by_mass_millidalton = BTreeMap::<i64, Vec<usize>>::new();
        for (index, row) in rows.iter().enumerate() {
            if let Some(unimod_id) = row.unimod_id {
                by_unimod.entry(unimod_id).or_default().push(index);
            }
            let bin = (row.mass_delta * 1000.0).round() as i64;
            by_mass_millidalton.entry(bin).or_default().push(index);
        }
        Ok(Self {
            rows,
            by_unimod,
            by_mass_millidalton,
        })
    }

    fn unimod_matches<'a>(
        &'a self,
        unimod_id: u32,
        site: FoundationModificationSite,
        residue: char,
    ) -> (Vec<&'a AssetModification>, Vec<&'a AssetModification>) {
        let mut exact = Vec::new();
        let mut ambiguous = Vec::new();
        for &row_index in self.by_unimod.get(&unimod_id).into_iter().flatten() {
            let row = &self.rows[row_index];
            match asset_site_compatibility(&row.site_spec, site, residue) {
                SiteCompatibility::Exact => exact.push(row),
                SiteCompatibility::AmbiguousContext => ambiguous.push(row),
                SiteCompatibility::Incompatible => {}
            }
        }
        (exact, ambiguous)
    }

    fn mass_candidates(
        &self,
        mass_delta: f64,
        site: FoundationModificationSite,
        residue: char,
    ) -> Vec<u32> {
        let mut ids = BTreeSet::<u32>::new();
        let center = (mass_delta * 1000.0).round() as i64;
        for bin in (center - 3)..=(center + 3) {
            for &row_index in self.by_mass_millidalton.get(&bin).into_iter().flatten() {
                let row = &self.rows[row_index];
                if (row.mass_delta - mass_delta).abs() > MASS_MATCH_TOLERANCE_DA {
                    continue;
                }
                if !matches!(
                    asset_site_compatibility(&row.site_spec, site, residue),
                    SiteCompatibility::Exact
                ) {
                    continue;
                }
                if let Some(id) = row.unimod_id {
                    ids.insert(id);
                }
            }
        }
        ids.into_iter().collect()
    }
}

fn asset_site_compatibility(
    site_spec: &str,
    site: FoundationModificationSite,
    residue: char,
) -> SiteCompatibility {
    let spec = site_spec.trim();
    match site {
        FoundationModificationSite::NTerm => {
            if spec.eq_ignore_ascii_case("Any_N-term") {
                SiteCompatibility::Exact
            } else if spec.eq_ignore_ascii_case("Protein_N-term") {
                SiteCompatibility::AmbiguousContext
            } else {
                SiteCompatibility::Incompatible
            }
        }
        FoundationModificationSite::CTerm => {
            if spec.eq_ignore_ascii_case("Any_C-term") {
                SiteCompatibility::Exact
            } else if spec.eq_ignore_ascii_case("Protein_C-term") {
                SiteCompatibility::AmbiguousContext
            } else {
                SiteCompatibility::Incompatible
            }
        }
        FoundationModificationSite::Residue(_) => {
            let expected = residue.to_ascii_uppercase().to_string();
            if spec.eq_ignore_ascii_case(&expected) {
                SiteCompatibility::Exact
            } else if spec
                .split_once('^')
                .is_some_and(|(base, _)| base.eq_ignore_ascii_case(&expected))
                || spec.eq_ignore_ascii_case("Anywhere")
            {
                SiteCompatibility::AmbiguousContext
            } else {
                SiteCompatibility::Incompatible
            }
        }
    }
}

fn composition_label(composition: ElementalComposition) -> String {
    let fields = [
        ("C", composition.carbon),
        ("13C", composition.carbon_13),
        ("H", composition.hydrogen),
        ("N", composition.nitrogen),
        ("15N", composition.nitrogen_15),
        ("O", composition.oxygen),
        ("S", composition.sulfur),
        ("P", composition.phosphorus),
    ];
    fields
        .into_iter()
        .filter(|(_, count)| *count != 0)
        .map(|(element, count)| format!("{element}:{count}"))
        .collect::<Vec<_>>()
        .join(";")
}

fn residue_for_modification(
    peptide: &PeptidoformInput,
    modification: &FoundationModification,
) -> Option<char> {
    peptide.sequence.chars().nth(modification.residue_index)
}

fn site_label(site: FoundationModificationSite, residue: char) -> String {
    match site {
        FoundationModificationSite::NTerm => "N-term".to_string(),
        FoundationModificationSite::CTerm => "C-term".to_string(),
        FoundationModificationSite::Residue(_) => format!("Residue:{residue}"),
    }
}

fn ptm_inventory_key(assessment: &ModificationAssessment) -> PtmInventoryKey {
    PtmInventoryKey {
        ptm_label: assessment.ptm_label.clone(),
        site_label: assessment.site_label.clone(),
        residue: assessment.residue,
    }
}

fn raw_ptm_inventory_key(
    peptide: &PeptidoformInput,
    modification: &FoundationModification,
) -> PtmInventoryKey {
    let residue = residue_for_modification(peptide, modification).unwrap_or('?');
    PtmInventoryKey {
        ptm_label: modification.identity_label(),
        site_label: if residue == '?' {
            "invalid".to_string()
        } else {
            site_label(modification.site, residue)
        },
        residue,
    }
}

fn assess_modification(
    peptide: &PeptidoformInput,
    modification: &FoundationModification,
    asset: &AssetChemistryIndex,
) -> ModificationAssessment {
    let ptm_label = modification.identity_label();
    let Some(residue) = residue_for_modification(peptide, modification) else {
        return ModificationAssessment {
            ptm_label,
            unimod_id: modification.unimod_id,
            site_label: "invalid".to_string(),
            residue: '?',
            mass_delta: f64::from(modification.mass_delta),
            registry_name: String::new(),
            registry_composition: String::new(),
            internal_exact_topology: false,
            exact_asset_rows: 0,
            ambiguous_asset_rows: 0,
            asset_names: Vec::new(),
            asset_compositions: Vec::new(),
            asset_smiles: Vec::new(),
            ambiguous_asset_names: Vec::new(),
            ambiguous_asset_compositions: Vec::new(),
            ambiguous_asset_smiles: Vec::new(),
            mass_candidate_unimod_ids: Vec::new(),
            status: ModificationChemistryStatus::InvalidSite,
            failure_reason: Some("modification_site_out_of_bounds".to_string()),
        };
    };

    let site_label = site_label(modification.site, residue);
    let internal_exact_topology = exact_graph_modification_for(residue, modification).is_some();
    let mut registry_name = String::new();
    let mut registry_composition = String::new();
    let mut exact_asset_rows = Vec::<&AssetModification>::new();
    let mut ambiguous_asset_rows = Vec::<&AssetModification>::new();
    let mut mass_candidate_unimod_ids = Vec::<u32>::new();

    if let Some(unimod_id) = modification.unimod_id {
        if let Some(definition) = common_unimod_definition(unimod_id) {
            registry_name = definition.name.to_string();
            registry_composition = composition_label(definition.composition);
        }
        (exact_asset_rows, ambiguous_asset_rows) =
            asset.unimod_matches(unimod_id, modification.site, residue);
    } else {
        mass_candidate_unimod_ids = asset.mass_candidates(
            f64::from(modification.mass_delta),
            modification.site,
            residue,
        );
    }

    let asset_names = unique_nonempty(exact_asset_rows.iter().map(|row| &row.name));
    let asset_compositions = unique_nonempty(exact_asset_rows.iter().map(|row| &row.composition));
    let asset_smiles = unique_nonempty(exact_asset_rows.iter().map(|row| &row.smiles));
    let ambiguous_asset_names = unique_nonempty(ambiguous_asset_rows.iter().map(|row| &row.name));
    let ambiguous_asset_compositions =
        unique_nonempty(ambiguous_asset_rows.iter().map(|row| &row.composition));
    let ambiguous_asset_smiles =
        unique_nonempty(ambiguous_asset_rows.iter().map(|row| &row.smiles));
    let composition_known = !registry_composition.is_empty()
        || !asset_compositions.is_empty()
        || !ambiguous_asset_compositions.is_empty();

    let (status, failure_reason) = if internal_exact_topology {
        (ModificationChemistryStatus::ExactInternalTopology, None)
    } else if modification.unimod_id.is_some() && !asset_smiles.is_empty() {
        (ModificationChemistryStatus::AssetAttachmentTemplate, None)
    } else if modification.unimod_id.is_some()
        && exact_asset_rows.is_empty()
        && !ambiguous_asset_rows.is_empty()
    {
        (
            ModificationChemistryStatus::CanonicalWithoutSiteChemistry,
            Some("ptm_site_requires_unavailable_terminal_context".to_string()),
        )
    } else if modification.unimod_id.is_some() && composition_known {
        (
            ModificationChemistryStatus::CompositionOnly,
            Some("ptm_topology_not_explicit".to_string()),
        )
    } else if modification.unimod_id.is_some() {
        (
            ModificationChemistryStatus::CanonicalWithoutSiteChemistry,
            Some("canonical_ptm_has_no_site_specific_chemistry".to_string()),
        )
    } else {
        match mass_candidate_unimod_ids.len() {
            0 => (
                ModificationChemistryStatus::MassOnlyUnmatched,
                Some("mass_only_modification_unmatched".to_string()),
            ),
            1 => (
                ModificationChemistryStatus::MassOnlyUniqueAssetCandidate,
                Some("mass_only_modification_identity_not_explicit".to_string()),
            ),
            _ => (
                ModificationChemistryStatus::MassOnlyAmbiguousAssetCandidates,
                Some("mass_only_modification_ambiguous".to_string()),
            ),
        }
    };

    ModificationAssessment {
        ptm_label,
        unimod_id: modification.unimod_id,
        site_label,
        residue,
        mass_delta: f64::from(modification.mass_delta),
        registry_name,
        registry_composition,
        internal_exact_topology,
        exact_asset_rows: exact_asset_rows.len(),
        ambiguous_asset_rows: ambiguous_asset_rows.len(),
        asset_names,
        asset_compositions,
        asset_smiles,
        ambiguous_asset_names,
        ambiguous_asset_compositions,
        ambiguous_asset_smiles,
        mass_candidate_unimod_ids,
        status,
        failure_reason,
    }
}

fn unique_nonempty<'a>(values: impl Iterator<Item = &'a String>) -> Vec<String> {
    let mut unique = BTreeSet::<String>::new();
    for value in values {
        let value = value.trim();
        if !value.is_empty() {
            unique.insert(value.to_string());
        }
    }
    unique.into_iter().collect()
}

fn assess_identity(
    peptide: &PeptidoformInput,
    asset: &AssetChemistryIndex,
) -> IdentityChemistryAssessment {
    if peptide.modifications.is_empty() {
        return IdentityChemistryAssessment {
            status: IdentityChemistryStatus::Unmodified,
            current_exact_topology_ready: true,
            existing_attachment_template_candidate: true,
            all_modifications_composition_known: true,
            failure_reasons: Vec::new(),
            modifications: Vec::new(),
        };
    }

    let modifications: Vec<ModificationAssessment> = peptide
        .modifications
        .iter()
        .map(|modification| assess_modification(peptide, modification, asset))
        .collect();
    let current_exact_topology_ready = modifications
        .iter()
        .all(|assessment| assessment.internal_exact_topology);
    let existing_attachment_template_candidate = modifications.iter().all(|assessment| {
        matches!(
            assessment.status,
            ModificationChemistryStatus::ExactInternalTopology
                | ModificationChemistryStatus::AssetAttachmentTemplate
        )
    });
    let all_modifications_composition_known = modifications.iter().all(|assessment| {
        !assessment.registry_composition.is_empty()
            || !assessment.asset_compositions.is_empty()
            || !assessment.ambiguous_asset_compositions.is_empty()
    });
    let has_ambiguous_or_unsupported_modification = modifications.iter().any(|assessment| {
        matches!(
            assessment.status,
            ModificationChemistryStatus::CanonicalWithoutSiteChemistry
                | ModificationChemistryStatus::MassOnlyUniqueAssetCandidate
                | ModificationChemistryStatus::MassOnlyAmbiguousAssetCandidates
                | ModificationChemistryStatus::MassOnlyUnmatched
                | ModificationChemistryStatus::InvalidSite
        )
    });
    let status = if current_exact_topology_ready {
        IdentityChemistryStatus::CurrentExactTopology
    } else if existing_attachment_template_candidate {
        IdentityChemistryStatus::ExistingAttachmentTemplateCandidate
    } else if !has_ambiguous_or_unsupported_modification
        && all_modifications_composition_known
        && modifications
            .iter()
            .all(|assessment| assessment.unimod_id.is_some())
    {
        IdentityChemistryStatus::CompositionOnlyExtensionRequired
    } else {
        IdentityChemistryStatus::AmbiguousOrUnsupported
    };
    let mut failure_reasons = BTreeSet::<String>::new();
    for assessment in &modifications {
        if let Some(reason) = &assessment.failure_reason {
            failure_reasons.insert(reason.clone());
        }
    }

    IdentityChemistryAssessment {
        status,
        current_exact_topology_ready,
        existing_attachment_template_candidate,
        all_modifications_composition_known,
        failure_reasons: failure_reasons.into_iter().collect(),
        modifications,
    }
}

fn canonical_identity_key(peptidoform: &str, charge: i32) -> String {
    format!("{peptidoform}|z{charge}")
}

fn length_bin(length: usize) -> &'static str {
    match length {
        0..=7 => "01_1-7",
        8..=12 => "02_8-12",
        13..=18 => "03_13-18",
        19..=25 => "04_19-25",
        26..=35 => "05_26-35",
        _ => "06_36+",
    }
}

fn collect_train_inventory(
    records: &[FoundationTrainingRecord],
    provenance_source_ids: &[String],
    benchmark: &FoundationBenchmarkManifest,
    asset: &AssetChemistryIndex,
) -> Result<TrainInventory> {
    if records.len() != provenance_source_ids.len() {
        bail!(
            "records/provenance length mismatch: {} vs {}",
            records.len(),
            provenance_source_ids.len()
        );
    }
    benchmark.validate_against_records(records)?;

    let mut inventory = TrainInventory::default();
    for entry in benchmark
        .entries
        .iter()
        .filter(|entry| entry.partition == FoundationPartition::Train)
    {
        let record = records
            .get(entry.record_index)
            .ok_or_else(|| anyhow::anyhow!("benchmark record index is out of bounds"))?;
        let source_id = provenance_source_ids
            .get(entry.record_index)
            .ok_or_else(|| anyhow::anyhow!("provenance record index is out of bounds"))?;
        inventory.train_records += 1;
        inventory
            .unique_peptidoforms
            .insert(entry.peptidoform.clone());
        let record_sequence_length = record.peptidoform.sequence.chars().count();
        *inventory
            .record_length_counts
            .entry(length_bin(record_sequence_length).to_string())
            .or_default() += 1;
        let record_charge_key = record
            .context
            .charge
            .map_or_else(|| "missing".to_string(), |charge| charge.to_string());
        *inventory
            .record_charge_counts
            .entry(record_charge_key)
            .or_default() += 1;
        if record.peptidoform.modifications.is_empty() {
            inventory.records_unmodified += 1;
        } else {
            inventory.records_modified += 1;
        }

        for modification in &record.peptidoform.modifications {
            let key = raw_ptm_inventory_key(&record.peptidoform, modification);
            *inventory.record_ptm_counts.entry(key.clone()).or_default() += 1;
            if !inventory.ptm_assessments.contains_key(&key) {
                inventory.ptm_assessments.insert(
                    key,
                    assess_modification(&record.peptidoform, modification, asset),
                );
            }
        }

        let charge = match record.context.charge {
            Some(charge) if charge > 0 => {
                inventory.records_with_positive_charge += 1;
                charge
            }
            Some(_) => {
                inventory.records_nonpositive_charge += 1;
                inventory
                    .nonpositive_charge_peptidoforms
                    .insert(entry.peptidoform.clone());
                continue;
            }
            None => {
                inventory.records_missing_charge += 1;
                inventory
                    .missing_charge_peptidoforms
                    .insert(entry.peptidoform.clone());
                continue;
            }
        };

        let identity_key = canonical_identity_key(&entry.peptidoform, charge);
        let sequence_length = record_sequence_length;
        let identity = inventory
            .identities
            .entry(identity_key.clone())
            .or_insert_with(|| IdentityRecord {
                identity_key: identity_key.clone(),
                peptidoform: entry.peptidoform.clone(),
                sequence: record.peptidoform.sequence.clone(),
                charge,
                sequence_length,
                record_count: 0,
                source_ids: BTreeSet::new(),
                precursor_mz: NumericAccumulator::default(),
                theoretical_neutral_mass: foundation_peptidoform_neutral_mass(&record.peptidoform)
                    .ok(),
                chemistry: assess_identity(&record.peptidoform, asset),
            });
        if identity.peptidoform != entry.peptidoform
            || identity.sequence != record.peptidoform.sequence
            || identity.charge != charge
        {
            bail!("identity collision for '{identity_key}'");
        }
        identity.record_count += 1;
        identity.source_ids.insert(source_id.clone());
        if let Some(mz) = record.context.precursor_mz {
            identity.precursor_mz.push(f64::from(mz));
        }
    }
    Ok(inventory)
}

fn aggregate_ptm_counts(inventory: &TrainInventory) -> BTreeMap<PtmInventoryKey, PtmCounts> {
    let mut counts = BTreeMap::<PtmInventoryKey, PtmCounts>::new();
    for (key, count) in &inventory.record_ptm_counts {
        counts.entry(key.clone()).or_default().record_occurrences = *count;
    }
    for identity in inventory.identities.values() {
        let keys: BTreeSet<PtmInventoryKey> = identity
            .chemistry
            .modifications
            .iter()
            .map(ptm_inventory_key)
            .collect();
        for key in keys {
            counts.entry(key).or_default().identity_occurrences += 1;
        }
    }
    counts
}

fn coverage_counts(inventory: &TrainInventory) -> CoverageCounts {
    let mut counts = CoverageCounts {
        records: inventory.train_records,
        identities: inventory.identities.len(),
        ..CoverageCounts::default()
    };
    for identity in inventory.identities.values() {
        match identity.chemistry.status {
            IdentityChemistryStatus::Unmodified => counts.unmodified_identities += 1,
            IdentityChemistryStatus::CurrentExactTopology => {
                counts.modified_identities += 1;
                counts.current_exact_topology += 1;
            }
            IdentityChemistryStatus::ExistingAttachmentTemplateCandidate => {
                counts.modified_identities += 1;
                counts.existing_attachment_template_candidate += 1;
            }
            IdentityChemistryStatus::CompositionOnlyExtensionRequired => {
                counts.modified_identities += 1;
                counts.composition_only_extension_required += 1;
            }
            IdentityChemistryStatus::AmbiguousOrUnsupported => {
                counts.modified_identities += 1;
                counts.ambiguous_or_unsupported += 1;
            }
        }
    }
    counts.current_exact_topology += counts.unmodified_identities;
    counts.existing_attachment_template_candidate += counts.current_exact_topology;
    counts
}

fn identity_chemistry_fraction(
    inventory: &TrainInventory,
    predicate: impl Fn(&IdentityRecord) -> bool,
) -> f64 {
    if inventory.identities.is_empty() {
        return 0.0;
    }
    let count = inventory
        .identities
        .values()
        .filter(|identity| predicate(identity))
        .count();
    count as f64 / inventory.identities.len() as f64
}

fn write_summary(
    out_dir: &Path,
    run_path: &Path,
    corpus_fingerprint: u64,
    benchmark: &FoundationBenchmarkManifest,
    asset: &AssetChemistryIndex,
    inventory: &TrainInventory,
) -> Result<()> {
    let path = out_dir.join("summary.tsv");
    let mut writer = BufWriter::new(File::create(&path)?);
    let coverage = coverage_counts(inventory);
    let exact_fraction = identity_chemistry_fraction(inventory, |identity| {
        identity.chemistry.current_exact_topology_ready
    });
    let template_fraction = identity_chemistry_fraction(inventory, |identity| {
        identity.chemistry.existing_attachment_template_candidate
    });
    let composition_fraction = identity_chemistry_fraction(inventory, |identity| {
        identity.chemistry.all_modifications_composition_known
    });

    writeln!(writer, "metric\tvalue")?;
    summary_row(&mut writer, "audit_version", AUDIT_VERSION)?;
    summary_row(&mut writer, "partition_scope", "TRAIN_only")?;
    summary_row(&mut writer, "run_config", &run_path.to_string_lossy())?;
    summary_row(
        &mut writer,
        "benchmark_manifest_fingerprint",
        &format!("fnv1a64:{:016x}", benchmark.manifest_fingerprint()),
    )?;
    summary_row(
        &mut writer,
        "benchmark_dataset_fingerprint",
        &format!("fnv1a64:{:016x}", benchmark.dataset_fingerprint),
    )?;
    summary_row(
        &mut writer,
        "loaded_corpus_fingerprint",
        &format!("fnv1a64:{corpus_fingerprint:016x}"),
    )?;
    let asset_unique_unimod_ids: BTreeSet<u32> =
        asset.rows.iter().filter_map(|row| row.unimod_id).collect();
    let asset_unimod_ids_with_smiles: BTreeSet<u32> = asset
        .rows
        .iter()
        .filter(|row| !row.smiles.trim().is_empty())
        .filter_map(|row| row.unimod_id)
        .collect();
    summary_row(
        &mut writer,
        "embedded_modification_asset_rows",
        &asset.rows.len().to_string(),
    )?;
    summary_row(
        &mut writer,
        "embedded_modification_asset_unique_unimod_ids",
        &asset_unique_unimod_ids.len().to_string(),
    )?;
    summary_row(
        &mut writer,
        "embedded_modification_asset_rows_with_composition",
        &asset
            .rows
            .iter()
            .filter(|row| !row.composition.trim().is_empty())
            .count()
            .to_string(),
    )?;
    summary_row(
        &mut writer,
        "embedded_modification_asset_rows_with_smiles",
        &asset
            .rows
            .iter()
            .filter(|row| !row.smiles.trim().is_empty())
            .count()
            .to_string(),
    )?;
    summary_row(
        &mut writer,
        "embedded_modification_asset_unique_unimod_ids_with_smiles",
        &asset_unimod_ids_with_smiles.len().to_string(),
    )?;
    summary_row(&mut writer, "existing_rustyms_dependency", "YES")?;
    summary_row(&mut writer, "existing_whole_peptide_3d_builder", "NO")?;
    summary_row(
        &mut writer,
        "mass_only_candidate_match_tolerance_da",
        &format!("{MASS_MATCH_TOLERANCE_DA:.6}"),
    )?;
    summary_row(
        &mut writer,
        "train_records",
        &inventory.train_records.to_string(),
    )?;
    summary_row(
        &mut writer,
        "train_records_with_positive_charge",
        &inventory.records_with_positive_charge.to_string(),
    )?;
    summary_row(
        &mut writer,
        "train_records_missing_charge",
        &inventory.records_missing_charge.to_string(),
    )?;
    summary_row(
        &mut writer,
        "train_records_nonpositive_charge",
        &inventory.records_nonpositive_charge.to_string(),
    )?;
    summary_row(
        &mut writer,
        "unique_train_peptidoforms_missing_charge",
        &inventory.missing_charge_peptidoforms.len().to_string(),
    )?;
    summary_row(
        &mut writer,
        "unique_train_peptidoforms_nonpositive_charge",
        &inventory.nonpositive_charge_peptidoforms.len().to_string(),
    )?;
    summary_row(
        &mut writer,
        "train_records_unmodified",
        &inventory.records_unmodified.to_string(),
    )?;
    summary_row(
        &mut writer,
        "train_records_modified",
        &inventory.records_modified.to_string(),
    )?;
    summary_row(
        &mut writer,
        "unique_train_peptidoforms",
        &inventory.unique_peptidoforms.len().to_string(),
    )?;
    summary_row(
        &mut writer,
        "unique_train_peptidoform_charge_identities",
        &inventory.identities.len().to_string(),
    )?;
    summary_row(
        &mut writer,
        "unique_unmodified_peptidoform_charge_identities",
        &coverage.unmodified_identities.to_string(),
    )?;
    summary_row(
        &mut writer,
        "unique_modified_peptidoform_charge_identities",
        &coverage.modified_identities.to_string(),
    )?;
    summary_row(
        &mut writer,
        "identities_current_exact_ptm_local_topology_ready_including_unmodified",
        &coverage.current_exact_topology.to_string(),
    )?;
    summary_row(
        &mut writer,
        "identities_existing_attachment_template_candidate_including_exact",
        &coverage.existing_attachment_template_candidate.to_string(),
    )?;
    summary_row(
        &mut writer,
        "identities_composition_only_extension_required",
        &coverage.composition_only_extension_required.to_string(),
    )?;
    summary_row(
        &mut writer,
        "identities_ambiguous_or_unsupported",
        &coverage.ambiguous_or_unsupported.to_string(),
    )?;
    summary_row(
        &mut writer,
        "fraction_current_exact_ptm_local_topology_ready",
        &format!("{exact_fraction:.8}"),
    )?;
    summary_row(
        &mut writer,
        "fraction_existing_attachment_template_candidate",
        &format!("{template_fraction:.8}"),
    )?;
    summary_row(
        &mut writer,
        "fraction_all_modifications_composition_known",
        &format!("{composition_fraction:.8}"),
    )?;
    summary_row(
        &mut writer,
        "deduplicated_conformer_generation_identity_count_if_all_positive_charge_train",
        &inventory.identities.len().to_string(),
    )?;
    summary_row(
        &mut writer,
        "deduplicated_existing_chemistry_candidate_count",
        &coverage.existing_attachment_template_candidate.to_string(),
    )?;
    summary_row(&mut writer, "measured_ccs_used_for_analysis", "NO")?;
    summary_row(&mut writer, "dev_labels_used", "NO")?;
    summary_row(&mut writer, "holdout_used", "NO")?;
    summary_row(&mut writer, "heavy_chemistry_toolkit_added", "NO")?;
    writer.flush()?;
    Ok(())
}

fn summary_row(writer: &mut BufWriter<File>, metric: &str, value: &str) -> Result<()> {
    writeln!(writer, "{}\t{}", tsv_escape(metric), tsv_escape(value))?;
    Ok(())
}

fn write_identity_inventory(out_dir: &Path, inventory: &TrainInventory) -> Result<()> {
    let path = out_dir.join("identity_inventory.tsv");
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "identity_key\tpeptidoform\tsequence\tcharge\tsequence_length\tlength_bin\tmodified\tmodification_count\tmodification_labels\trecord_count\tsource_count\tprecursor_mz_count\tprecursor_mz_mean\tprecursor_mz_min\tprecursor_mz_max\ttheoretical_neutral_mass_da\tchemistry_status\tcurrent_exact_ptm_local_topology_ready\texisting_attachment_template_candidate\tall_modifications_composition_known\tfailure_reasons"
    )?;
    for identity in inventory.identities.values() {
        let modification_labels = identity
            .chemistry
            .modifications
            .iter()
            .map(|assessment| {
                format!(
                    "{}@{}:{}",
                    assessment.ptm_label,
                    assessment.site_label,
                    assessment.status.as_str()
                )
            })
            .collect::<Vec<_>>()
            .join(";");
        writeln!(
            writer,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            tsv_escape(&identity.identity_key),
            tsv_escape(&identity.peptidoform),
            tsv_escape(&identity.sequence),
            identity.charge,
            identity.sequence_length,
            length_bin(identity.sequence_length),
            yes_no(!identity.chemistry.modifications.is_empty()),
            identity.chemistry.modifications.len(),
            tsv_escape(&modification_labels),
            identity.record_count,
            identity.source_ids.len(),
            identity.precursor_mz.count,
            option_f64(identity.precursor_mz.mean()),
            option_f64(identity.precursor_mz.min),
            option_f64(identity.precursor_mz.max),
            option_f64(identity.theoretical_neutral_mass),
            identity.chemistry.status.as_str(),
            yes_no(identity.chemistry.current_exact_topology_ready),
            yes_no(identity.chemistry.existing_attachment_template_candidate),
            yes_no(identity.chemistry.all_modifications_composition_known),
            tsv_escape(&identity.chemistry.failure_reasons.join(";")),
        )?;
    }
    writer.flush()?;
    Ok(())
}

fn write_ptm_inventory(
    out_dir: &Path,
    inventory: &TrainInventory,
    ptm_counts: &BTreeMap<PtmInventoryKey, PtmCounts>,
) -> Result<()> {
    let path = out_dir.join("ptm_inventory.tsv");
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "ptm_label\tunimod_id\tsite\tresidue\tmass_delta_da\trecord_occurrences\tidentity_occurrences\tregistry_name\tregistry_composition\tinternal_exact_topology\texact_asset_site_rows\tambiguous_asset_context_rows\tasset_site_names\tasset_site_compositions\tasset_site_smiles\tambiguous_asset_names\tambiguous_asset_compositions\tambiguous_asset_smiles\tmass_candidate_unimod_ids\tchemistry_status\tfailure_reason"
    )?;
    for (key, counts) in ptm_counts {
        let Some(assessment) = inventory.ptm_assessments.get(key) else {
            continue;
        };
        writeln!(
            writer,
            "{}\t{}\t{}\t{}\t{:.6}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            tsv_escape(&assessment.ptm_label),
            assessment
                .unimod_id
                .map_or_else(String::new, |id| id.to_string()),
            tsv_escape(&assessment.site_label),
            assessment.residue,
            assessment.mass_delta,
            counts.record_occurrences,
            counts.identity_occurrences,
            tsv_escape(&assessment.registry_name),
            tsv_escape(&assessment.registry_composition),
            yes_no(assessment.internal_exact_topology),
            assessment.exact_asset_rows,
            assessment.ambiguous_asset_rows,
            tsv_escape(&assessment.asset_names.join("|")),
            tsv_escape(&assessment.asset_compositions.join("|")),
            tsv_escape(&assessment.asset_smiles.join("|")),
            tsv_escape(&assessment.ambiguous_asset_names.join("|")),
            tsv_escape(&assessment.ambiguous_asset_compositions.join("|")),
            tsv_escape(&assessment.ambiguous_asset_smiles.join("|")),
            assessment
                .mass_candidate_unimod_ids
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join("|"),
            assessment.status.as_str(),
            tsv_escape(assessment.failure_reason.as_deref().unwrap_or_default()),
        )?;
    }
    writer.flush()?;
    Ok(())
}

fn write_distribution_tables(out_dir: &Path, inventory: &TrainInventory) -> Result<()> {
    let mut charge_identity = BTreeMap::<i32, CoverageCounts>::new();
    let mut length_identity = BTreeMap::<String, CoverageCounts>::new();
    for identity in inventory.identities.values() {
        update_coverage(
            charge_identity.entry(identity.charge).or_default(),
            identity,
        );
        update_coverage(
            length_identity
                .entry(length_bin(identity.sequence_length).to_string())
                .or_default(),
            identity,
        );
    }

    write_coverage_table_i32(
        &out_dir.join("coverage_by_charge.tsv"),
        "charge",
        &inventory.record_charge_counts,
        &charge_identity,
    )?;
    write_coverage_table_string(
        &out_dir.join("coverage_by_length_bin.tsv"),
        "length_bin",
        &inventory.record_length_counts,
        &length_identity,
    )?;
    Ok(())
}

fn update_coverage(counts: &mut CoverageCounts, identity: &IdentityRecord) {
    counts.records += identity.record_count;
    counts.identities += 1;
    if identity.chemistry.modifications.is_empty() {
        counts.unmodified_identities += 1;
    } else {
        counts.modified_identities += 1;
    }
    if identity.chemistry.current_exact_topology_ready {
        counts.current_exact_topology += 1;
    }
    if identity.chemistry.existing_attachment_template_candidate {
        counts.existing_attachment_template_candidate += 1;
    }
    match identity.chemistry.status {
        IdentityChemistryStatus::CompositionOnlyExtensionRequired => {
            counts.composition_only_extension_required += 1;
        }
        IdentityChemistryStatus::AmbiguousOrUnsupported => {
            counts.ambiguous_or_unsupported += 1;
        }
        _ => {}
    }
}

fn write_coverage_table_i32(
    path: &Path,
    key_name: &str,
    record_counts: &BTreeMap<String, usize>,
    identity_counts: &BTreeMap<i32, CoverageCounts>,
) -> Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    write_coverage_header(&mut writer, key_name)?;
    let mut emitted = BTreeSet::<String>::new();
    for (charge, counts) in identity_counts {
        let key = charge.to_string();
        emitted.insert(key.clone());
        write_coverage_row(
            &mut writer,
            &key,
            record_counts.get(&key).copied().unwrap_or_default(),
            counts,
        )?;
    }
    for (key, records) in record_counts {
        if emitted.contains(key) {
            continue;
        }
        write_coverage_row(&mut writer, key, *records, &CoverageCounts::default())?;
    }
    writer.flush()?;
    Ok(())
}

fn write_coverage_table_string(
    path: &Path,
    key_name: &str,
    record_counts: &BTreeMap<String, usize>,
    identity_counts: &BTreeMap<String, CoverageCounts>,
) -> Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    write_coverage_header(&mut writer, key_name)?;
    let mut emitted = BTreeSet::<String>::new();
    for (key, counts) in identity_counts {
        emitted.insert(key.clone());
        write_coverage_row(
            &mut writer,
            key,
            record_counts.get(key).copied().unwrap_or_default(),
            counts,
        )?;
    }
    for (key, records) in record_counts {
        if emitted.contains(key) {
            continue;
        }
        write_coverage_row(&mut writer, key, *records, &CoverageCounts::default())?;
    }
    writer.flush()?;
    Ok(())
}

fn write_coverage_header(writer: &mut BufWriter<File>, key_name: &str) -> Result<()> {
    writeln!(
        writer,
        "{key_name}\trecords\tidentities\tunmodified_identities\tmodified_identities\tcurrent_exact_ptm_local_topology_ready\texisting_attachment_template_candidate\tcomposition_only_extension_required\tambiguous_or_unsupported"
    )?;
    Ok(())
}

fn write_coverage_row(
    writer: &mut BufWriter<File>,
    key: &str,
    records: usize,
    counts: &CoverageCounts,
) -> Result<()> {
    writeln!(
        writer,
        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
        tsv_escape(key),
        records,
        counts.identities,
        counts.unmodified_identities,
        counts.modified_identities,
        counts.current_exact_topology,
        counts.existing_attachment_template_candidate,
        counts.composition_only_extension_required,
        counts.ambiguous_or_unsupported,
    )?;
    Ok(())
}

fn write_failure_tables(out_dir: &Path, inventory: &TrainInventory) -> Result<()> {
    let mut reason_counts = BTreeMap::<String, (usize, usize)>::new();
    let mut ptm_failure_counts = BTreeMap::<PtmInventoryKey, (usize, usize, String)>::new();
    let mut charge_failure_counts = BTreeMap::<(String, String), (usize, usize)>::new();
    let mut length_failure_counts = BTreeMap::<(String, String), (usize, usize)>::new();

    for identity in inventory.identities.values() {
        for reason in &identity.chemistry.failure_reasons {
            reason_counts.entry(reason.clone()).or_default().1 += 1;

            let charge_entry = charge_failure_counts
                .entry((identity.charge.to_string(), reason.clone()))
                .or_default();
            charge_entry.0 += identity.record_count;
            charge_entry.1 += 1;

            let length_entry = length_failure_counts
                .entry((
                    length_bin(identity.sequence_length).to_string(),
                    reason.clone(),
                ))
                .or_default();
            length_entry.0 += identity.record_count;
            length_entry.1 += 1;
        }

        let mut identity_failures = BTreeMap::<PtmInventoryKey, String>::new();
        for assessment in &identity.chemistry.modifications {
            let Some(reason) = &assessment.failure_reason else {
                continue;
            };
            identity_failures
                .entry(ptm_inventory_key(assessment))
                .or_insert_with(|| reason.clone());
        }
        for (key, reason) in identity_failures {
            let entry = ptm_failure_counts
                .entry(key)
                .or_insert_with(|| (0, 0, reason));
            entry.1 += 1;
        }
    }

    // PTM-level record occurrence counts come from the complete TRAIN scan. The
    // identity counts above remain deduplicated by peptidoform+charge.
    for (key, record_count) in &inventory.record_ptm_counts {
        if let Some(entry) = ptm_failure_counts.get_mut(key) {
            entry.0 = *record_count;
            reason_counts.entry(entry.2.clone()).or_default().0 += *record_count;
        }
    }

    if inventory.records_missing_charge > 0 {
        reason_counts.insert(
            "missing_precursor_charge".to_string(),
            (
                inventory.records_missing_charge,
                inventory.missing_charge_peptidoforms.len(),
            ),
        );
        charge_failure_counts.insert(
            (
                "missing".to_string(),
                "missing_precursor_charge".to_string(),
            ),
            (
                inventory.records_missing_charge,
                inventory.missing_charge_peptidoforms.len(),
            ),
        );
    }
    if inventory.records_nonpositive_charge > 0 {
        reason_counts.insert(
            "nonpositive_precursor_charge".to_string(),
            (
                inventory.records_nonpositive_charge,
                inventory.nonpositive_charge_peptidoforms.len(),
            ),
        );
        charge_failure_counts.insert(
            (
                "nonpositive".to_string(),
                "nonpositive_precursor_charge".to_string(),
            ),
            (
                inventory.records_nonpositive_charge,
                inventory.nonpositive_charge_peptidoforms.len(),
            ),
        );
    }

    let mut reason_writer = BufWriter::new(File::create(out_dir.join("failure_reasons.tsv"))?);
    writeln!(
        reason_writer,
        "failure_reason\trecord_occurrences\tidentity_or_peptidoform_occurrences"
    )?;
    for (reason, (records, identities)) in reason_counts {
        writeln!(
            reason_writer,
            "{}\t{}\t{}",
            tsv_escape(&reason),
            records,
            identities
        )?;
    }
    reason_writer.flush()?;

    let mut ptm_writer = BufWriter::new(File::create(out_dir.join("failure_by_ptm.tsv"))?);
    writeln!(
        ptm_writer,
        "ptm_label\tsite\tresidue\tfailure_reason\trecord_occurrences\tidentity_occurrences"
    )?;
    for (key, (records, identities, reason)) in ptm_failure_counts {
        writeln!(
            ptm_writer,
            "{}\t{}\t{}\t{}\t{}\t{}",
            tsv_escape(&key.ptm_label),
            tsv_escape(&key.site_label),
            key.residue,
            tsv_escape(&reason),
            records,
            identities,
        )?;
    }
    ptm_writer.flush()?;

    let mut charge_writer = BufWriter::new(File::create(out_dir.join("failure_by_charge.tsv"))?);
    writeln!(
        charge_writer,
        "charge\tfailure_reason\trecord_occurrences\tidentity_or_peptidoform_occurrences"
    )?;
    for ((charge, reason), (records, identities)) in charge_failure_counts {
        writeln!(
            charge_writer,
            "{}\t{}\t{}\t{}",
            tsv_escape(&charge),
            tsv_escape(&reason),
            records,
            identities,
        )?;
    }
    charge_writer.flush()?;

    let mut length_writer =
        BufWriter::new(File::create(out_dir.join("failure_by_length_bin.tsv"))?);
    writeln!(
        length_writer,
        "length_bin\tfailure_reason\trecord_occurrences\tidentity_occurrences"
    )?;
    for ((bin, reason), (records, identities)) in length_failure_counts {
        writeln!(
            length_writer,
            "{}\t{}\t{}\t{}",
            tsv_escape(&bin),
            tsv_escape(&reason),
            records,
            identities,
        )?;
    }
    length_writer.flush()?;
    Ok(())
}

fn write_report(
    out_dir: &Path,
    run_path: &Path,
    benchmark_path: &Path,
    asset: &AssetChemistryIndex,
    inventory: &TrainInventory,
) -> Result<()> {
    let coverage = coverage_counts(inventory);
    let total = inventory.identities.len().max(1) as f64;
    let exact_pct = 100.0 * coverage.current_exact_topology as f64 / total;
    let template_pct = 100.0 * coverage.existing_attachment_template_candidate as f64 / total;
    let composition_or_better = inventory
        .identities
        .values()
        .filter(|identity| identity.chemistry.all_modifications_composition_known)
        .count();
    let composition_pct = 100.0 * composition_or_better as f64 / total;

    let mut writer = BufWriter::new(File::create(out_dir.join("report.md"))?);
    writeln!(writer, "# ReDeeM structure-signal feasibility v1 — Stage A")?;
    writeln!(writer)?;
    writeln!(writer, "- Audit version: `{AUDIT_VERSION}`")?;
    writeln!(writer, "- Run config: `{}`", run_path.display())?;
    writeln!(writer, "- Benchmark: `{}`", benchmark_path.display())?;
    writeln!(writer, "- Partition scope: **TRAIN only**")?;
    writeln!(writer, "- Measured CCS used for analysis/selection: **NO**")?;
    writeln!(writer, "- DEV labels used: **NO**")?;
    writeln!(writer, "- HOLDOUT used: **NO**")?;
    writeln!(writer, "- New heavy chemistry toolkit: **NO**")?;
    writeln!(
        writer,
        "- Benchmark integrity validation: **YES** (record fingerprints are checked only for corpus/partition fidelity; label values are not read by this audit's inventory logic)"
    )?;
    writeln!(writer)?;
    writeln!(writer, "## Existing source chemistry")?;
    writeln!(writer)?;
    writeln!(writer, "- Existing `rustyms` dependency: **YES**")?;
    writeln!(writer, "- Existing whole-peptide 3D builder: **NO**")?;
    writeln!(
        writer,
        "- Embedded modification asset rows: `{}`",
        asset.rows.len()
    )?;
    writeln!(
        writer,
        "- Asset rows with elemental composition: `{}`",
        asset
            .rows
            .iter()
            .filter(|row| !row.composition.trim().is_empty())
            .count()
    )?;
    writeln!(
        writer,
        "- Asset rows with non-empty structural SMILES/template text: `{}`",
        asset
            .rows
            .iter()
            .filter(|row| !row.smiles.trim().is_empty())
            .count()
    )?;
    writeln!(writer)?;
    writeln!(writer, "## TRAIN inventory")?;
    writeln!(writer)?;
    writeln!(writer, "- TRAIN records: `{}`", inventory.train_records)?;
    writeln!(
        writer,
        "- Positive-charge TRAIN records: `{}`",
        inventory.records_with_positive_charge
    )?;
    writeln!(
        writer,
        "- Unique peptidoform+charge identities: `{}`",
        inventory.identities.len()
    )?;
    writeln!(
        writer,
        "- TRAIN records missing precursor charge: `{}` across `{}` unique peptidoforms",
        inventory.records_missing_charge,
        inventory.missing_charge_peptidoforms.len()
    )?;
    writeln!(
        writer,
        "- TRAIN records with non-positive precursor charge: `{}` across `{}` unique peptidoforms",
        inventory.records_nonpositive_charge,
        inventory.nonpositive_charge_peptidoforms.len()
    )?;
    writeln!(
        writer,
        "- Deduplicated conformer-generation identities if all positive-charge TRAIN identities were attempted: `{}`",
        inventory.identities.len()
    )?;
    writeln!(writer)?;
    writeln!(writer, "## Existing chemistry coverage tiers")?;
    writeln!(writer)?;
    writeln!(
        writer,
        "- Current exact PTM-local topology ready (including unmodified): `{}` ({exact_pct:.2}%)",
        coverage.current_exact_topology
    )?;
    writeln!(
        writer,
        "- Existing attachment-template candidate (including exact): `{}` ({template_pct:.2}%)",
        coverage.existing_attachment_template_candidate
    )?;
    writeln!(
        writer,
        "- All PTM compositions known: `{}` ({composition_pct:.2}%)",
        composition_or_better
    )?;
    writeln!(
        writer,
        "- Composition-only extension required: `{}`",
        coverage.composition_only_extension_required
    )?;
    writeln!(
        writer,
        "- Ambiguous or unsupported chemistry: `{}`",
        coverage.ambiguous_or_unsupported
    )?;
    writeln!(writer)?;
    writeln!(writer, "## Interpretation boundary")?;
    writeln!(writer)?;
    writeln!(
        writer,
        "`current_exact_ptm_local_topology_ready` means every observed PTM is represented by an explicit ReDeeM local heavy-atom graph transformation (or the peptide is unmodified)."
    )?;
    writeln!(
        writer,
        "`existing_attachment_template_candidate` additionally accepts an exact-site non-empty SMILES entry from the embedded modification asset. Those SMILES contain attachment placeholders and still require a validated whole-peptide molecule builder before Stage B."
    )?;
    writeln!(
        writer,
        "Composition-only PTMs are not treated as topology-complete. Mass-only PTMs are never promoted to chemically complete, even when their mass has one apparent asset match, because the source annotation does not explicitly identify the modification."
    )?;
    writeln!(
        writer,
        "The existing foundation atom graph is residue-local and intentionally leaves peptide-bond geometry to sequence modeling; this audit therefore measures chemistry metadata coverage, not conformer-generation success."
    )?;
    writeln!(writer)?;
    writeln!(writer, "## Stage B gate inputs")?;
    writeln!(writer)?;
    writeln!(
        writer,
        "Use `identity_inventory.tsv`, `ptm_inventory.tsv`, `coverage_by_charge.tsv`, `coverage_by_length_bin.tsv`, `failure_by_ptm.tsv`, `failure_by_charge.tsv`, `failure_by_length_bin.tsv`, and `failure_reasons.tsv` to decide whether a bounded 5k–10k TRAIN-only conformer pilot is justified. Do not use measured CCS to make that decision."
    )?;
    writer.flush()?;
    Ok(())
}

fn yes_no(value: bool) -> &'static str {
    if value {
        "YES"
    } else {
        "NO"
    }
}

fn option_f64(value: Option<f64>) -> String {
    value.map_or_else(String::new, |value| format!("{value:.8}"))
}

fn tsv_escape(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('\t', "\\t")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
}

fn run_audit(run_path: &Path, out_dir: &Path) -> Result<()> {
    if out_dir.exists() {
        bail!("output directory must be fresh: {}", out_dir.display());
    }
    let run = read_foundation_training_run_config(run_path)?;
    let corpus = load_foundation_corpus(&run.corpus)?;
    let benchmark = FoundationBenchmarkManifest::read_tsv(&run.benchmark_manifest)?;
    benchmark
        .validate_against_records(&corpus.records)
        .context("foundation benchmark does not match the assembled corpus")?;
    let asset = AssetChemistryIndex::load_embedded()?;
    let provenance_source_ids: Vec<String> = corpus
        .provenance
        .iter()
        .map(|provenance| provenance.source_id.clone())
        .collect();
    let inventory =
        collect_train_inventory(&corpus.records, &provenance_source_ids, &benchmark, &asset)?;
    if inventory.train_records == 0 {
        bail!("benchmark TRAIN partition is empty");
    }

    create_dir_all(out_dir)?;
    let ptm_counts = aggregate_ptm_counts(&inventory);
    write_summary(
        out_dir,
        run_path,
        corpus.corpus_fingerprint,
        &benchmark,
        &asset,
        &inventory,
    )?;
    write_identity_inventory(out_dir, &inventory)?;
    write_ptm_inventory(out_dir, &inventory, &ptm_counts)?;
    write_distribution_tables(out_dir, &inventory)?;
    write_failure_tables(out_dir, &inventory)?;
    write_report(
        out_dir,
        run_path,
        &run.benchmark_manifest,
        &asset,
        &inventory,
    )?;

    println!("audit_version\t{AUDIT_VERSION}");
    println!("partition_scope\tTRAIN_only");
    println!("train_records\t{}", inventory.train_records);
    println!(
        "unique_train_peptidoform_charge_identities\t{}",
        inventory.identities.len()
    );
    println!("measured_ccs_used_for_analysis\tNO");
    println!("dev_labels_used\tNO");
    println!("holdout_used\tNO");
    println!("out\t{}", out_dir.display());
    Ok(())
}

fn self_test() -> Result<()> {
    let asset = AssetChemistryIndex::load_embedded()?;
    if asset.rows.len() < 100 {
        bail!("embedded modification asset unexpectedly small");
    }

    let cam = PeptidoformInput {
        sequence: "ACDMK".to_string(),
        modifications: vec![FoundationModification::unimod(
            FoundationModificationSite::Residue(1),
            1,
            4,
            57.021_465,
        )],
    };
    let cam_assessment = assess_identity(&cam, &asset);
    if cam_assessment.status != IdentityChemistryStatus::CurrentExactTopology {
        bail!("self-test: carbamidomethyl C should have current exact topology");
    }

    let phospho = PeptidoformInput {
        sequence: "PEPTIDEK".to_string(),
        modifications: vec![FoundationModification::unimod(
            FoundationModificationSite::Residue(3),
            3,
            21,
            79.966_33,
        )],
    };
    let phospho_assessment = assess_identity(&phospho, &asset);
    if phospho_assessment.status != IdentityChemistryStatus::ExistingAttachmentTemplateCandidate {
        bail!("self-test: phospho T should be covered by an asset attachment template");
    }

    let tmt = PeptidoformInput {
        sequence: "PEPTIDEK".to_string(),
        modifications: vec![FoundationModification::unimod(
            FoundationModificationSite::Residue(7),
            7,
            737,
            229.162_93,
        )],
    };
    let tmt_assessment = assess_identity(&tmt, &asset);
    if tmt_assessment.status != IdentityChemistryStatus::CompositionOnlyExtensionRequired {
        bail!("self-test: TMT6plex K should be composition-only in current assets");
    }

    let protein_nterm_only = PeptidoformInput {
        sequence: "PEPTIDEK".to_string(),
        modifications: vec![FoundationModification::unimod(
            FoundationModificationSite::NTerm,
            0,
            47,
            238.229_66,
        )],
    };
    let protein_nterm_assessment = assess_identity(&protein_nterm_only, &asset);
    if protein_nterm_assessment.status != IdentityChemistryStatus::AmbiguousOrUnsupported {
        bail!(
            "self-test: protein-N-terminal-only chemistry must not be promoted without protein-terminal context"
        );
    }
    if !protein_nterm_assessment.modifications[0]
        .ambiguous_asset_compositions
        .iter()
        .any(|composition| !composition.is_empty())
    {
        bail!("self-test: ambiguous terminal chemistry should still expose its composition");
    }

    let open_mass = PeptidoformInput {
        sequence: "ACDMK".to_string(),
        modifications: vec![FoundationModification::mass_delta(3, 15.994_915)],
    };
    let open_assessment = assess_identity(&open_mass, &asset);
    if open_assessment.status != IdentityChemistryStatus::AmbiguousOrUnsupported {
        bail!("self-test: mass-only modification must remain unresolved");
    }
    if open_assessment.modifications[0]
        .mass_candidate_unimod_ids
        .is_empty()
    {
        bail!("self-test: oxidation-like mass should expose an asset candidate");
    }

    if length_bin(7) != "01_1-7" || length_bin(8) != "02_8-12" || length_bin(36) != "06_36+" {
        bail!("self-test: length-bin boundaries changed");
    }

    println!("structure_signal_feasibility_v1_self_test=PASS");
    Ok(())
}

fn usage() -> &'static str {
    "usage:\n  foundation_audit_structure_signal_feasibility_v1 --self-test\n  foundation_audit_structure_signal_feasibility_v1 <run.yaml> <fresh-output-dir>"
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() == 1 && args[0] == "--self-test" {
        return self_test();
    }
    if args.len() != 2 {
        bail!("{}", usage());
    }
    let run_path = PathBuf::from(&args[0]);
    let out_dir = PathBuf::from(&args[1]);
    run_audit(&run_path, &out_dir)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_asset_and_chemistry_tiers_are_stable() {
        self_test().unwrap();
    }

    #[test]
    fn protein_terminal_asset_context_is_not_treated_as_exact() {
        assert_eq!(
            asset_site_compatibility("Protein_N-term", FoundationModificationSite::NTerm, 'A'),
            SiteCompatibility::AmbiguousContext
        );
        assert_eq!(
            asset_site_compatibility("Any_N-term", FoundationModificationSite::NTerm, 'A'),
            SiteCompatibility::Exact
        );
        assert_eq!(
            asset_site_compatibility(
                "F^Protein_N-term",
                FoundationModificationSite::Residue(0),
                'F'
            ),
            SiteCompatibility::AmbiguousContext
        );
    }

    #[test]
    fn canonical_identity_includes_charge() {
        assert_ne!(
            canonical_identity_key("PEPTIDEK", 2),
            canonical_identity_key("PEPTIDEK", 3)
        );
    }

    #[test]
    fn train_inventory_deduplicates_by_peptidoform_and_charge() {
        use redeem_properties::foundation::{
            build_foundation_benchmark_manifest, FoundationSplitConfig, FoundationSplitMode,
            RetentionTimeLabels, TrainingContext,
        };

        fn record(charge: Option<i32>, precursor_mz: f32) -> FoundationTrainingRecord {
            FoundationTrainingRecord {
                peptidoform: PeptidoformInput::unmodified("PEPTIDEK"),
                retention_time: RetentionTimeLabels::default(),
                ccs: None,
                fragments: Vec::new(),
                observed_spectrum_peaks: Vec::new(),
                context: TrainingContext {
                    charge,
                    precursor_mz: Some(precursor_mz),
                    ..TrainingContext::default()
                },
                run_id: None,
            }
        }

        let records = vec![
            record(Some(2), 450.0),
            record(Some(2), 451.0),
            record(Some(3), 300.0),
        ];
        let benchmark = build_foundation_benchmark_manifest(
            &records,
            FoundationSplitConfig {
                mode: FoundationSplitMode::Sequence,
                validation_fraction: 0.0,
                test_fraction: 0.0,
                seed: 7,
            },
            false,
        )
        .unwrap();
        let provenance = vec!["synthetic".to_string(); records.len()];
        let asset = AssetChemistryIndex::load_embedded().unwrap();
        let inventory = collect_train_inventory(&records, &provenance, &benchmark, &asset).unwrap();

        assert_eq!(inventory.train_records, 3);
        assert_eq!(inventory.identities.len(), 2);
        assert_eq!(
            inventory
                .identities
                .get(&canonical_identity_key("PEPTIDEK", 2))
                .unwrap()
                .record_count,
            2
        );
        assert_eq!(
            inventory
                .identities
                .get(&canonical_identity_key("PEPTIDEK", 3))
                .unwrap()
                .record_count,
            1
        );
    }
}
