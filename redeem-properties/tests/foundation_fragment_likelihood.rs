use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use redeem_properties::foundation::{
    foundation_fragment_likelihood_score, FoundationConfig, FoundationSpectrum,
    PeptideFoundationMultiTaskModel, PeptideGraphFeaturizer, PeptidoformInput,
    PrecursorContextBatch,
};

#[test]
fn missing_fragment_peaks_are_soft_evidence_not_candidate_invalidity() {
    let peptide = PeptidoformInput::unmodified("PEPTIDE");
    let score = foundation_fragment_likelihood_score(
        &peptide,
        &FoundationSpectrum::default(),
        &vec![vec![1.0; 8]; 6],
    )
    .expect("chemically valid peptide must remain scoreable without observed peaks");

    assert_eq!(score.matched_core_ions, 0);
    assert_eq!(score.core_ions, 24);
    assert_eq!(score.core_cosine, 0.0);
}

#[test]
fn relative_fragment_intensity_pattern_changes_global_candidate_score() {
    let peptide = PeptidoformInput::unmodified("AG");
    // A b1 ~=72.0444, complementary G y1 ~=76.0393.  The observed b:y
    // intensity ratio is deliberately asymmetric so that merely matching the
    // same fragment masses is insufficient; the learned intensity pattern
    // must agree as well.
    let spectrum = FoundationSpectrum::from_pairs([(72.0444, 100.0), (76.0393, 9.0)]);

    let aligned = vec![vec![1.0, 0.0, 0.09, 0.0, 0.0, 0.0, 0.0, 0.0]];
    let inverted = vec![vec![0.09, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]];

    let aligned_score =
        foundation_fragment_likelihood_score(&peptide, &spectrum, &aligned).expect("aligned score");
    let inverted_score = foundation_fragment_likelihood_score(&peptide, &spectrum, &inverted)
        .expect("inverted score");

    assert!(
        aligned_score.core_cosine > inverted_score.core_cosine + 0.2,
        "aligned={aligned_score:?} inverted={inverted_score:?}"
    );
}

#[test]
fn complete_candidate_score_is_deterministic() {
    let peptide = PeptidoformInput::unmodified("AG");
    let spectrum = FoundationSpectrum::from_pairs([(72.0444, 100.0), (76.0393, 80.0)]);
    let predicted = vec![vec![1.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0]];

    let first = foundation_fragment_likelihood_score(&peptide, &spectrum, &predicted).unwrap();
    let second = foundation_fragment_likelihood_score(&peptide, &spectrum, &predicted).unwrap();
    assert_eq!(first, second);
}

#[test]
fn v0230_batch128_ms2_only_forward_uses_cuda_safe_projection_shape() {
    let device = Device::Cpu;
    let config = FoundationConfig {
        // Keep the exact v0.23 checkpoint geometry seen by the CUDA smoke:
        // 64 positions, 96 model dimensions -> 2*96 + 21 = 213 cleavage
        // features and 63 cleavages. FoundationConfig::default() uses a
        // wider model, so pin model_dim explicitly for this regression.
        model_dim: 96,
        transformer_layers: 1,
        dropout: 0.0,
        ..FoundationConfig::default()
    };
    assert_eq!(config.max_sequence_len, 64);
    assert_eq!(config.model_dim * 2 + 21, 213);

    let featurizer = PeptideGraphFeaturizer::new(config.clone()).unwrap();
    let peptides = vec![PeptidoformInput::unmodified("PEPTIDE"); 128];
    let batch = featurizer.featurize(&peptides, &device).unwrap();
    let context = PrecursorContextBatch::unknown(128, &device).unwrap();
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = PeptideFoundationMultiTaskModel::new(config.clone(), vb).unwrap();

    let ms2 = model.forward_ms2_t(&batch, &context, false).unwrap();
    assert_eq!(
        ms2.dims(),
        &[
            128,
            config.max_sequence_len - 1,
            config.ms2_fragment_channels
        ]
    );
}
