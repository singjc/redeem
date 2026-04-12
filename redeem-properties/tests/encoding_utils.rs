//! Integration tests for featurize and peptdeep_utils.

use candle_core::Device;
use std::sync::Arc;

use redeem_properties::building_blocks::featurize::{
    aa_indices_tensor, aa_indices_tensor_from_arc,
};
use redeem_properties::utils::peptdeep_utils::{
    ccs_to_mobility_bruker, get_modification_indices, get_modification_string, remove_mass_shift,
    MODIFICATION_MAP,
};

// ---------------------------------------------------------------------------
// AA indices tensor
// ---------------------------------------------------------------------------

#[test]
fn aa_indices_tensor_simple_sequence() {
    let t = aa_indices_tensor("PEP", &Device::Cpu).unwrap();
    // Shape: (1, 5, 1) — padding start + 3 AAs + padding end
    let dims = t.dims3().unwrap();
    assert_eq!(dims.0, 1, "batch");
    assert_eq!(dims.1, 5, "seq_len = 3 + 2 padding");
    assert_eq!(dims.2, 1, "feature dim");
}

#[test]
fn aa_indices_tensor_from_arc_matches_str_version() {
    let seq = "PEPTIDEK";
    let t_str = aa_indices_tensor(seq, &Device::Cpu).unwrap();
    let arc: Arc<[u8]> = Arc::from(seq.as_bytes());
    let t_arc = aa_indices_tensor_from_arc(&arc, &Device::Cpu).unwrap();

    let v1 = t_str.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let v2 = t_arc.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(
        v1, v2,
        "str and arc versions should produce identical tensors"
    );
}

#[test]
fn aa_indices_tensor_empty_sequence() {
    let t = aa_indices_tensor("", &Device::Cpu).unwrap();
    let dims = t.dims3().unwrap();
    // Just the two padding tokens
    assert_eq!(dims.1, 2);
}

#[test]
fn aa_indices_tensor_all_amino_acids() {
    // All 20 standard AAs + some unusual ones
    let seq = "ACDEFGHIKLMNPQRSTVWY";
    let result = aa_indices_tensor(seq, &Device::Cpu);
    assert!(result.is_ok(), "should handle all standard AAs");
}

// ---------------------------------------------------------------------------
// Modification utilities
// ---------------------------------------------------------------------------

#[test]
fn remove_mass_shift_strips_brackets() {
    let naked = remove_mass_shift("PEPTM[+15.9949]IDE");
    assert_eq!(naked, "PEPTMIDE");
}

#[test]
fn remove_mass_shift_no_mods() {
    let naked = remove_mass_shift("PEPTIDE");
    assert_eq!(naked, "PEPTIDE");
}

#[test]
fn remove_mass_shift_multiple_mods() {
    let naked = remove_mass_shift("SEQU[+42.0106]ENC[+57.0215]E");
    assert_eq!(naked, "SEQUENCE");
}

#[test]
fn modification_map_is_loaded() {
    let map = &*MODIFICATION_MAP;
    assert!(!map.is_empty(), "MODIFICATION_MAP should not be empty");
}

#[test]
fn get_modification_string_unmodified() {
    let map = &*MODIFICATION_MAP;
    let mods = get_modification_string("PEPTIDE", map);
    // Unmodified peptides should produce an empty-ish string
    assert!(
        mods.is_empty() || mods.chars().all(|c| c == ';'),
        "no modifications expected, got: {}",
        mods
    );
}

#[test]
fn get_modification_indices_unmodified() {
    let sites = get_modification_indices("PEPTIDE");
    assert!(
        sites.is_empty() || sites.chars().all(|c| c == ';'),
        "no modification sites expected, got: {}",
        sites
    );
}

#[test]
fn get_modification_string_with_mod() {
    let map = &*MODIFICATION_MAP;
    let mods = get_modification_string("PEPTM[+15.9949]IDE", map);
    // Should contain some non-empty mod reference
    assert!(
        !mods.is_empty(),
        "modified peptide should produce a mod string"
    );
}

// ---------------------------------------------------------------------------
// CCS ↔ ion mobility conversion
// ---------------------------------------------------------------------------

#[test]
fn ccs_to_mobility_positive_values() {
    let mobility = ccs_to_mobility_bruker(400.0, 2.0, 500.0);
    assert!(
        mobility > 0.0,
        "mobility should be positive, got {}",
        mobility
    );
}

#[test]
fn ccs_to_mobility_higher_ccs_gives_higher_one_over_k0() {
    // Higher CCS → larger collisional cross section → slower drift → higher 1/K0
    let mob1 = ccs_to_mobility_bruker(400.0, 2.0, 400.0);
    let mob2 = ccs_to_mobility_bruker(400.0, 2.0, 600.0);
    assert!(
        mob2 > mob1,
        "higher CCS should give higher 1/K0: {} vs {}",
        mob1,
        mob2
    );
}
