use candle_core::Device;
use redeem_properties::foundation::{
    foundation_chemistry_diffusion_final_mass_valid,
    foundation_chemistry_diffusion_partition_isolated,
    foundation_chemistry_diffusion_project_mass_valid,
    foundation_chemistry_diffusion_refinement_timesteps,
    foundation_chemistry_diffusion_row_neutral_mass, foundation_peptidoform_neutral_mass,
    ChemistryDiffusionFeaturizer, FoundationDiffusionCollator, FoundationDiffusionConfig,
    FoundationDiffusionVocabulary, FoundationPartition, FoundationSpectrum,
    FoundationSpectrumConfig, PeptidoformInput, FOUNDATION_DIFFUSION_EOS,
    FOUNDATION_DIFFUSION_MASK, FOUNDATION_DIFFUSION_PAD, FOUNDATION_DIFFUSION_VOCAB_SIZE,
};

fn tiny_config() -> FoundationDiffusionConfig {
    FoundationDiffusionConfig {
        max_tokens: 16,
        model_dim: 32,
        num_attention_heads: 4,
        feed_forward_dim: 64,
        spectrum_layers: 1,
        decoder_layers: 1,
        diffusion_steps: 20,
        beta_start: 0.02,
        beta_end: 0.35,
        spectrum: FoundationSpectrumConfig {
            max_peaks: 8,
            ..FoundationSpectrumConfig::default()
        },
        precursor_mass_tolerance_da: 0.05,
        ..FoundationDiffusionConfig::default()
    }
}

#[test]
fn v0210_categorical_corruption_is_deterministic_and_preserves_active_length() {
    let device = Device::Cpu;
    let collator = FoundationDiffusionCollator::new(tiny_config()).unwrap();
    let peptides = vec![
        PeptidoformInput::unmodified("PEPTIDEK"),
        PeptidoformInput::unmodified("MELTQK"),
    ];
    let first = collator
        .collate(&peptides, &[20, 20], 20260921, &device)
        .unwrap();
    let second = collator
        .collate(&peptides, &[20, 20], 20260921, &device)
        .unwrap();
    assert_eq!(first.noisy_token_rows, second.noisy_token_rows);
    assert_eq!(first.active_lengths, second.active_lengths);
    assert!(first
        .noisy_token_rows
        .iter()
        .zip(first.clean_tokens.to_vec2::<u32>().unwrap())
        .any(|(noisy, clean)| noisy != &clean));
    for (row, &active) in first.noisy_token_rows.iter().zip(&first.active_lengths) {
        assert!(row[..active]
            .iter()
            .all(|&token| token != FOUNDATION_DIFFUSION_PAD));
        assert!(row[active..]
            .iter()
            .all(|&token| token == FOUNDATION_DIFFUSION_PAD));
    }
}

#[test]
fn v0210_timestep_schedule_is_fixed_decreasing_and_ends_at_one() {
    let steps = foundation_chemistry_diffusion_refinement_timesteps(20).unwrap();
    assert_eq!(steps, vec![8, 7, 6, 5, 4, 3, 2, 1]);
    assert!(steps.windows(2).all(|pair| pair[0] > pair[1]));
}

#[test]
fn v0210_denoising_targets_align_exactly_with_active_indices() {
    let device = Device::Cpu;
    let collator = FoundationDiffusionCollator::new(tiny_config()).unwrap();
    let peptide = PeptidoformInput::unmodified("PEPTIDEK");
    let batch = collator.collate(&[peptide], &[10], 77, &device).unwrap();
    let clean = batch.clean_tokens.to_vec2::<u32>().unwrap();
    let active = batch.active_indices.to_vec1::<u32>().unwrap();
    let classes = batch.target_classes.to_vec1::<u32>().unwrap();
    assert_eq!(active.len(), classes.len());
    for (&flat, &class) in active.iter().zip(&classes) {
        assert_eq!(clean[0][flat as usize], class);
    }
    assert_eq!(
        clean[0][batch.active_lengths[0] - 1],
        FOUNDATION_DIFFUSION_EOS
    );
}

#[test]
fn v0210_chemistry_recomputes_mass_state_from_current_noisy_hypothesis() {
    let device = Device::Cpu;
    let config = tiny_config();
    let vocabulary = FoundationDiffusionVocabulary;
    let peptide = PeptidoformInput::unmodified("AG");
    let clean = vocabulary.encode(&peptide, config.max_tokens).unwrap();
    let active = clean
        .iter()
        .position(|&token| token == FOUNDATION_DIFFUSION_PAD)
        .unwrap();
    let mut noisy = clean.clone();
    noisy[0] = FOUNDATION_DIFFUSION_MASK;
    let precursor_mass = foundation_peptidoform_neutral_mass(&peptide).unwrap();
    let spectrum = FoundationSpectrum::from_pairs([(72.0, 10.0), (147.0, 5.0)]);
    let featurizer = ChemistryDiffusionFeaturizer::new(&config).unwrap();
    let clean_features = featurizer
        .featurize(
            &[clean],
            &[active],
            &[spectrum.clone()],
            &[precursor_mass],
            &[2],
            &device,
        )
        .unwrap()
        .full_state_features
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let noisy_features = featurizer
        .featurize(
            &[noisy],
            &[active],
            &[spectrum],
            &[precursor_mass],
            &[2],
            &device,
        )
        .unwrap()
        .full_state_features
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    // Full-state feature 1 is the current complete-hypothesis precursor
    // residual. Masking A removes its known mass and must change this state.
    let full_state_dim = 4usize;
    let flat = (1 * FOUNDATION_DIFFUSION_VOCAB_SIZE + 3) * full_state_dim + 1;
    assert_ne!(clean_features[flat], noisy_features[flat]);
}

#[test]
fn v0210_final_mass_validity_requires_clean_grammar_and_exact_precursor_mass() {
    let config = tiny_config();
    let vocabulary = FoundationDiffusionVocabulary;
    let peptide = PeptidoformInput::unmodified("AG");
    let mut row = vocabulary.encode(&peptide, config.max_tokens).unwrap();
    let active = row
        .iter()
        .position(|&token| token == FOUNDATION_DIFFUSION_PAD)
        .unwrap();
    let precursor_mass = foundation_peptidoform_neutral_mass(&peptide).unwrap();
    assert!(foundation_chemistry_diffusion_final_mass_valid(
        &row,
        active,
        precursor_mass,
        config.precursor_mass_tolerance_da,
    ));
    row[0] = FOUNDATION_DIFFUSION_MASK;
    assert!(!foundation_chemistry_diffusion_final_mass_valid(
        &row,
        active,
        precursor_mass,
        config.precursor_mass_tolerance_da,
    ));
}

#[test]
fn v0210_final_projection_can_repair_one_wrong_residue_without_ar_decoding() {
    let config = tiny_config();
    let vocabulary = FoundationDiffusionVocabulary;
    let target = PeptidoformInput::unmodified("AG");
    let target_row = vocabulary.encode(&target, config.max_tokens).unwrap();
    let active = target_row
        .iter()
        .position(|&token| token == FOUNDATION_DIFFUSION_PAD)
        .unwrap();
    let mut current = target_row.clone();
    let glycine = vocabulary
        .encode(&PeptidoformInput::unmodified("G"), config.max_tokens)
        .unwrap()[0];
    current[0] = glycine;
    let mut logits = vec![vec![-20.0f32; FOUNDATION_DIFFUSION_VOCAB_SIZE]; active];
    for position in 0..active {
        logits[position][target_row[position] as usize] = 10.0;
        logits[position][current[position] as usize] = 5.0;
    }
    let precursor_mass = foundation_peptidoform_neutral_mass(&target).unwrap();
    let projected = foundation_chemistry_diffusion_project_mass_valid(
        &current,
        active,
        &logits,
        precursor_mass,
        config.precursor_mass_tolerance_da,
    )
    .unwrap()
    .expect("mass-valid projection");
    assert_eq!(&projected[..active], &target_row[..active]);
    assert!(
        (foundation_chemistry_diffusion_row_neutral_mass(&projected, active).unwrap()
            - precursor_mass)
            .abs()
            <= config.precursor_mass_tolerance_da
    );
}

#[test]
fn v0210_partition_contract_rejects_any_test_leakage() {
    assert!(foundation_chemistry_diffusion_partition_isolated(
        &[FoundationPartition::Train, FoundationPartition::Train],
        &[FoundationPartition::Validation],
    ));
    assert!(!foundation_chemistry_diffusion_partition_isolated(
        &[FoundationPartition::Train, FoundationPartition::Test],
        &[FoundationPartition::Validation],
    ));
    assert!(!foundation_chemistry_diffusion_partition_isolated(
        &[FoundationPartition::Train],
        &[FoundationPartition::Validation, FoundationPartition::Test],
    ));
}
