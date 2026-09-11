use redeem_properties::foundation::{
    foundation_fragment_likelihood_score, FoundationSpectrum, PeptidoformInput,
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
