use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use redeem_properties::foundation::fragment_relation::{
    FOUNDATION_FRAGMENT_RELATION_INVERSE_CLAIM_OFFSET_V0240,
    FOUNDATION_FRAGMENT_RELATION_MATCHED_OFFSET_V0240,
};
use redeem_properties::foundation::{
    foundation_compatibility_listwise_loss, foundation_fragment_relation_features,
    foundation_fragment_relation_legacy_log_prior, FoundationSpectrum, FoundationSpectrumPeak,
    PeptideSpectrumFragmentRelationEnergy, PeptidoformInput,
    FOUNDATION_FRAGMENT_RELATION_FEATURE_DIM_V0240,
};

fn simple_spectrum() -> FoundationSpectrum {
    FoundationSpectrum {
        peaks: vec![
            FoundationSpectrumPeak {
                // b1(A)
                mz: 72.044_39,
                intensity: 100.0,
            },
            FoundationSpectrumPeak {
                mz: 130.0,
                intensity: 25.0,
            },
        ],
    }
}

fn predicted() -> Vec<Vec<f32>> {
    vec![vec![1.0, 0.25, 0.75, 0.2, 0.0, 0.0, 0.0, 0.0]]
}

#[test]
fn v0240_relation_features_expose_adjacent_residues_peak_match_and_competition() {
    let peptide = PeptidoformInput::unmodified("AG");
    let single = foundation_fragment_relation_features(
        &[peptide.clone()],
        &simple_spectrum(),
        &[predicted()],
        &[0.0],
        63,
    )
    .unwrap();
    assert_eq!(single.candidates, 1);
    assert_eq!(single.max_cleavages, 63);
    assert_eq!(
        single.features.len(),
        63 * FOUNDATION_FRAGMENT_RELATION_FEATURE_DIM_V0240
    );
    assert_eq!(single.mask.iter().filter(|&&value| value > 0.5).count(), 1);
    assert_eq!(
        single.features[FOUNDATION_FRAGMENT_RELATION_MATCHED_OFFSET_V0240],
        1.0
    );
    assert!(
        (single.features[FOUNDATION_FRAGMENT_RELATION_INVERSE_CLAIM_OFFSET_V0240] - 1.0).abs()
            < 1e-6
    );

    let shared = foundation_fragment_relation_features(
        &[peptide.clone(), peptide],
        &simple_spectrum(),
        &[predicted(), predicted()],
        &[0.0, 0.0],
        63,
    )
    .unwrap();
    assert_eq!(shared.contested_peaks, 1);
    assert!(
        (shared.features[FOUNDATION_FRAGMENT_RELATION_INVERSE_CLAIM_OFFSET_V0240] - 0.5).abs()
            < 1e-6
    );
}

#[test]
fn v0240_relation_model_returns_one_logit_per_same_spectrum_candidate() {
    let device = Device::Cpu;
    let peptide = PeptidoformInput::unmodified("AG");
    let rows = foundation_fragment_relation_features(
        &[peptide.clone(), peptide],
        &simple_spectrum(),
        &[predicted(), predicted()],
        &[0.0, 0.0],
        63,
    )
    .unwrap();
    let batch = rows.to_batch(&device).unwrap();
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = PeptideSpectrumFragmentRelationEnergy::new(vb).unwrap();
    let scores = model.forward_grouped(&batch, 2).unwrap();
    assert_eq!(scores.dims(), &[1, 2]);
}

#[test]
fn v0240_positive_first_listwise_objective_prefers_positive_margin() {
    let device = Device::Cpu;
    let good = candle_core::Tensor::from_vec(vec![3.0f32, 0.0, -1.0], (1, 3), &device).unwrap();
    let bad = candle_core::Tensor::from_vec(vec![-1.0f32, 3.0, 0.0], (1, 3), &device).unwrap();
    let good_loss = foundation_compatibility_listwise_loss(&good)
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    let bad_loss = foundation_compatibility_listwise_loss(&bad)
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    assert!(good_loss < bad_loss);
}

#[test]
fn v0240_fixed_legacy_prior_is_monotone_and_rank_one_is_neutral() {
    let rank1 = foundation_fragment_relation_legacy_log_prior(1);
    let rank2 = foundation_fragment_relation_legacy_log_prior(2);
    let rank32 = foundation_fragment_relation_legacy_log_prior(32);
    assert_eq!(rank1, 0.0);
    assert!(rank1 > rank2);
    assert!(rank2 > rank32);
}
