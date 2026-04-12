//! Utility functions for computing precursor and product m/z values using `rustyms`.
//!
//! These functions parse peptide sequences in ProForma notation (which supports
//! mass-shift modifications like `[+42.0106]`) and compute theoretical fragment
//! m/z values for b and y ion series.

use anyhow::{Context, Result};
use rustyms::{
    chemistry::MassMode,
    fragment::FragmentKind,
    prelude::*,
    system::{e, isize::Charge},
};

/// Represents a single product (fragment) ion with its m/z and metadata.
#[derive(Debug, Clone)]
pub struct ProductIon {
    /// The ion type as a string: "b" or "y"
    pub ion_type: String,
    /// The charge state of the fragment ion
    pub charge: i32,
    /// The ordinal (series number, 1-based from the terminal)
    pub ordinal: usize,
    /// The monoisotopic m/z value
    pub mz: f64,
}

/// Represents the complete m/z information for a peptide at a given charge state.
#[derive(Debug, Clone)]
pub struct PeptideMzInfo {
    /// The precursor m/z value
    pub precursor_mz: f64,
    /// The product (fragment) ions
    pub product_ions: Vec<ProductIon>,
}

/// Compute the precursor m/z for a peptide in ProForma notation at a given charge state.
///
/// # Arguments
/// * `proforma_sequence` - Peptide sequence in ProForma notation, e.g. `"PEPTM[+15.9949]IDE"`
/// * `charge` - Precursor charge state (must be > 0)
///
/// # Returns
/// The monoisotopic precursor m/z value.
///
/// # Errors
/// Returns an error if the sequence cannot be parsed or if the charge is invalid.
pub fn compute_precursor_mz(proforma_sequence: &str, charge: i32) -> Result<f64> {
    if charge <= 0 {
        anyhow::bail!("Charge must be positive, got {}", charge);
    }

    let peptide = CompoundPeptidoformIon::pro_forma(proforma_sequence, None).map_err(|err| {
        anyhow::anyhow!(
            "Failed to parse ProForma sequence '{}': {}",
            proforma_sequence,
            err
        )
    })?;

    // Get the molecular formula(s) for the peptide
    let formulas = peptide.formulas();

    // Use the first (and typically only) formula for simple linear peptides
    let formula = formulas
        .first()
        .context("No molecular formula found for the peptide")?;

    // Compute precursor m/z: (neutral_mass + charge * proton_mass) / charge
    // Proton mass ≈ 1.00727646677 Da
    let proton_mass: f64 = 1.007_276_466_77;
    let neutral_mass = formula.monoisotopic_mass().value;
    let precursor_mz = (neutral_mass + (charge as f64) * proton_mass) / (charge as f64);

    Ok(precursor_mz)
}

/// Compute all theoretical product ion m/z values for b and y ions.
///
/// Generates theoretical fragments using rustyms for the given peptide,
/// filtering to only b and y ions (without neutral losses).
///
/// # Arguments
/// * `proforma_sequence` - Peptide sequence in ProForma notation
/// * `max_fragment_charge` - Maximum fragment charge state to consider (typically 1 or 2)
///
/// # Returns
/// A vector of `ProductIon` structs containing the ion type, charge, ordinal, and m/z.
///
/// # Errors
/// Returns an error if the sequence cannot be parsed.
pub fn compute_product_mzs(
    proforma_sequence: &str,
    max_fragment_charge: i32,
) -> Result<Vec<ProductIon>> {
    if max_fragment_charge <= 0 {
        anyhow::bail!(
            "max_fragment_charge must be positive, got {}",
            max_fragment_charge
        );
    }

    let peptide = CompoundPeptidoformIon::pro_forma(proforma_sequence, None).map_err(|err| {
        anyhow::anyhow!(
            "Failed to parse ProForma sequence '{}': {}",
            proforma_sequence,
            err
        )
    })?;

    // Generate theoretical fragments up to the specified charge
    let model = FragmentationModel::all();
    let fragments = peptide
        .generate_theoretical_fragments(Charge::new::<e>(max_fragment_charge as isize), model);

    let mut product_ions = Vec::new();

    for fragment in &fragments {
        let kind = fragment.ion.kind();

        // Only keep b and y ions
        let ion_type_str = match kind {
            FragmentKind::b => "b",
            FragmentKind::y => "y",
            _ => continue,
        };

        // Skip fragments with neutral losses
        if !fragment.neutral_loss.is_empty() {
            continue;
        }

        // Get the series number (ordinal, 1-based)
        let ordinal = match fragment.ion.position() {
            Some(pos) => pos.series_number,
            None => continue,
        };

        // Get the charge as a plain integer
        let frag_charge = fragment.charge.value as i32;

        // Get the m/z value
        let mz = match fragment.mz(MassMode::Monoisotopic) {
            Some(mz_val) => mz_val.value,
            None => continue,
        };

        product_ions.push(ProductIon {
            ion_type: ion_type_str.to_string(),
            charge: frag_charge,
            ordinal,
            mz,
        });
    }

    // Sort by ion_type, then charge, then ordinal for consistent ordering
    product_ions.sort_by(|a, b| {
        a.ion_type
            .cmp(&b.ion_type)
            .then(a.charge.cmp(&b.charge))
            .then(a.ordinal.cmp(&b.ordinal))
    });

    Ok(product_ions)
}

/// Compute both precursor and product m/z values for a peptide.
///
/// This is a convenience function that calls both `compute_precursor_mz` and
/// `compute_product_mzs` and bundles the results together.
///
/// # Arguments
/// * `proforma_sequence` - Peptide sequence in ProForma notation
/// * `charge` - Precursor charge state
/// * `max_fragment_charge` - Maximum fragment charge state
///
/// # Returns
/// A `PeptideMzInfo` struct containing the precursor m/z and all product ions.
pub fn compute_peptide_mz_info(
    proforma_sequence: &str,
    charge: i32,
    max_fragment_charge: i32,
) -> Result<PeptideMzInfo> {
    let precursor_mz = compute_precursor_mz(proforma_sequence, charge)?;
    let product_ions = compute_product_mzs(proforma_sequence, max_fragment_charge)?;

    Ok(PeptideMzInfo {
        precursor_mz,
        product_ions,
    })
}

/// Filter product ions to match a specific set of predicted ion types and charges.
///
/// Given a list of product ions and the predicted ion types/charges/ordinals from
/// the MS2 model, this function finds the matching product ion m/z for each predicted
/// fragment. If no match is found, `f64::NAN` is returned for that position.
///
/// # Arguments
/// * `product_ions` - The computed theoretical product ions
/// * `predicted_ion_types` - The ion types from MS2 prediction (e.g., ["b", "y", "b", "y"])
/// * `predicted_charges` - The charges from MS2 prediction (e.g., [1, 1, 2, 2])
/// * `predicted_ordinals` - The ordinals from MS2 prediction (e.g., [1, 1, 1, 1])
///
/// # Returns
/// A vector of m/z values aligned with the predicted arrays. NaN if no match found.
pub fn match_product_mzs(
    product_ions: &[ProductIon],
    predicted_ion_types: &[String],
    predicted_charges: &[i32],
    predicted_ordinals: &[usize],
) -> Vec<f64> {
    let n = predicted_ion_types.len();
    let mut mzs = vec![f64::NAN; n];

    for i in 0..n {
        let target_type = &predicted_ion_types[i];
        let target_charge = predicted_charges[i];
        let target_ordinal = predicted_ordinals[i];

        // Find matching product ion
        for ion in product_ions {
            if ion.ion_type == *target_type
                && ion.charge == target_charge
                && ion.ordinal == target_ordinal
            {
                mzs[i] = ion.mz;
                break;
            }
        }
    }

    mzs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_precursor_mz_simple() {
        // Simple unmodified peptide: PEPTIDE, charge 2
        let mz = compute_precursor_mz("PEPTIDE", 2).unwrap();
        // Monoisotopic mass of PEPTIDE ≈ 799.36 Da
        // (799.36 + 2 * 1.00728) / 2 ≈ 400.69
        assert!(mz > 400.0 && mz < 401.0, "Precursor m/z = {}", mz);
    }

    #[test]
    fn test_compute_precursor_mz_modified() {
        // Modified peptide with oxidation on M
        let mz = compute_precursor_mz("PEPTM[+15.9949]IDE", 2).unwrap();
        assert!(mz > 0.0, "Precursor m/z should be positive: {}", mz);
    }

    #[test]
    fn test_compute_precursor_mz_invalid_charge() {
        let result = compute_precursor_mz("PEPTIDE", 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_compute_product_mzs_simple() {
        let ions = compute_product_mzs("PEPTIDE", 2).unwrap();
        assert!(!ions.is_empty(), "Should have fragment ions");

        // Check that we have both b and y ions
        let has_b = ions.iter().any(|i| i.ion_type == "b");
        let has_y = ions.iter().any(|i| i.ion_type == "y");
        assert!(has_b, "Should have b ions");
        assert!(has_y, "Should have y ions");

        // Check all m/z values are positive
        for ion in &ions {
            assert!(ion.mz > 0.0, "m/z should be positive: {:?}", ion);
            assert!(ion.ordinal > 0, "ordinal should be >= 1: {:?}", ion);
            assert!(ion.charge > 0, "charge should be positive: {:?}", ion);
        }
    }

    #[test]
    fn test_compute_peptide_mz_info() {
        let info = compute_peptide_mz_info("PEPTIDE", 2, 2).unwrap();
        assert!(info.precursor_mz > 0.0);
        assert!(!info.product_ions.is_empty());
    }

    #[test]
    fn test_match_product_mzs() {
        let ions = compute_product_mzs("PEPTIDE", 2).unwrap();

        // Create a small subset of predicted types to match
        let predicted_types = vec!["b".to_string(), "y".to_string()];
        let predicted_charges = vec![1, 1];
        let predicted_ordinals = vec![1, 1];

        let mzs = match_product_mzs(
            &ions,
            &predicted_types,
            &predicted_charges,
            &predicted_ordinals,
        );
        assert_eq!(mzs.len(), 2);
        // b1 and y1 should both have valid m/z values
        assert!(!mzs[0].is_nan(), "b1+1 should have a match");
        assert!(!mzs[1].is_nan(), "y1+1 should have a match");
    }
}
