use candle_core::Device;
use redeem_properties::foundation::{
    foundation_peptidoform_neutral_mass, foundation_structured_edit_argmax,
    foundation_structured_edit_finalize, foundation_structured_edit_open_row,
    foundation_structured_edit_partition_isolated, foundation_structured_edit_set_targets,
    FoundationDiffusionCollator, FoundationDiffusionConfig, FoundationDiffusionVocabulary,
    FoundationPartition, FoundationSpectrumConfig, PeptidoformInput, FOUNDATION_DIFFUSION_EOS,
    FOUNDATION_DIFFUSION_MASK, FOUNDATION_DIFFUSION_PAD, FOUNDATION_DIFFUSION_VOCAB_SIZE,
    FOUNDATION_STRUCTURED_EDIT_CONTEXT_TIMESTEP_V0220,
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
fn v0220_open_state_preserves_v0200_content_but_reopens_length() {
    let config = tiny_config();
    let vocabulary = FoundationDiffusionVocabulary;
    let initial = vocabulary
        .encode(&PeptidoformInput::unmodified("PEPTIDE"), config.max_tokens)
        .unwrap();
    let active = initial
        .iter()
        .position(|&token| token == FOUNDATION_DIFFUSION_PAD)
        .unwrap();
    let open = foundation_structured_edit_open_row(&initial, active, config.max_tokens).unwrap();
    assert_eq!(&open[..active - 1], &initial[..active - 1]);
    assert!(open[active - 1..]
        .iter()
        .all(|&token| token == FOUNDATION_DIFFUSION_MASK));
    assert!(!open.contains(&FOUNDATION_DIFFUSION_EOS));
}

#[test]
fn v0220_target_alignment_uses_true_eos_and_true_length_not_open_canvas_width() {
    let device = Device::Cpu;
    let config = tiny_config();
    let vocabulary = FoundationDiffusionVocabulary;
    let initial = vocabulary
        .encode(&PeptidoformInput::unmodified("PEPTIDE"), config.max_tokens)
        .unwrap();
    let initial_active = initial
        .iter()
        .position(|&token| token == FOUNDATION_DIFFUSION_PAD)
        .unwrap();
    let open =
        foundation_structured_edit_open_row(&initial, initial_active, config.max_tokens).unwrap();
    let target = vocabulary
        .encode(&PeptidoformInput::unmodified("PEPTIDER"), config.max_tokens)
        .unwrap();
    let target_active = target
        .iter()
        .position(|&token| token == FOUNDATION_DIFFUSION_PAD)
        .unwrap();
    let collator = FoundationDiffusionCollator::new(config.clone()).unwrap();
    let batch = collator
        .collate_inference_tokens(
            &[open],
            &[config.max_tokens],
            FOUNDATION_STRUCTURED_EDIT_CONTEXT_TIMESTEP_V0220,
            &device,
        )
        .unwrap();
    let batch =
        foundation_structured_edit_set_targets(batch, &[target.clone()], &[target_active]).unwrap();
    assert_eq!(
        batch.target_classes.to_vec1::<u32>().unwrap(),
        target[..target_active].to_vec()
    );
    assert_eq!(
        batch.length_targets.to_vec1::<u32>().unwrap(),
        vec![(target_active - 1) as u32]
    );
}

#[test]
fn v0220_argmax_can_change_multiple_positions_and_move_eos_in_one_pass() {
    let config = tiny_config();
    let vocabulary = FoundationDiffusionVocabulary;
    let initial = vocabulary
        .encode(&PeptidoformInput::unmodified("AG"), config.max_tokens)
        .unwrap();
    let initial_active = initial
        .iter()
        .position(|&token| token == FOUNDATION_DIFFUSION_PAD)
        .unwrap();
    let open =
        foundation_structured_edit_open_row(&initial, initial_active, config.max_tokens).unwrap();
    let target = vocabulary
        .encode(&PeptidoformInput::unmodified("PEP"), config.max_tokens)
        .unwrap();
    let target_active = target
        .iter()
        .position(|&token| token == FOUNDATION_DIFFUSION_PAD)
        .unwrap();
    let mut token_logits = vec![vec![-20.0f32; FOUNDATION_DIFFUSION_VOCAB_SIZE]; config.max_tokens];
    let mut gate_logits = vec![vec![0.0f32, 5.0f32]; config.max_tokens];
    for position in 0..target_active - 1 {
        token_logits[position][target[position] as usize] = 10.0;
    }
    // Force CHANGE at the existing positions and all opened positions.
    gate_logits[0] = vec![0.0, 5.0];
    let mut length_logits = vec![-20.0f32; config.max_tokens];
    length_logits[target_active - 1] = 10.0;
    let (draft, active, changed) =
        foundation_structured_edit_argmax(&open, &token_logits, &gate_logits, &length_logits)
            .unwrap();
    assert_eq!(active, target_active);
    assert_eq!(&draft[..target_active], &target[..target_active]);
    assert!(changed >= 2);
}

#[test]
fn v0220_finalization_preserves_hard_precursor_mass_contract() {
    let config = tiny_config();
    let vocabulary = FoundationDiffusionVocabulary;
    let target = PeptidoformInput::unmodified("AG");
    let target_row = vocabulary.encode(&target, config.max_tokens).unwrap();
    let active = target_row
        .iter()
        .position(|&token| token == FOUNDATION_DIFFUSION_PAD)
        .unwrap();
    let gly = vocabulary
        .encode(&PeptidoformInput::unmodified("G"), config.max_tokens)
        .unwrap()[0];
    let mut draft = target_row.clone();
    draft[0] = gly;
    let mut logits = vec![vec![-20.0f32; FOUNDATION_DIFFUSION_VOCAB_SIZE]; config.max_tokens];
    for position in 0..active {
        logits[position][target_row[position] as usize] = 10.0;
        logits[position][draft[position] as usize] = 5.0;
    }
    let precursor_mass = foundation_peptidoform_neutral_mass(&target).unwrap();
    let (final_row, projected, fallback) = foundation_structured_edit_finalize(
        &draft,
        active,
        &logits,
        &target_row,
        active,
        precursor_mass,
        config.precursor_mass_tolerance_da,
    )
    .unwrap();
    assert!(projected);
    assert!(!fallback);
    assert_eq!(&final_row[..active], &target_row[..active]);
}

#[test]
fn v0220_partition_contract_rejects_test_leakage() {
    assert!(foundation_structured_edit_partition_isolated(
        &[FoundationPartition::Train, FoundationPartition::Train],
        &[FoundationPartition::Validation],
    ));
    assert!(!foundation_structured_edit_partition_isolated(
        &[FoundationPartition::Train, FoundationPartition::Test],
        &[FoundationPartition::Validation],
    ));
    assert!(!foundation_structured_edit_partition_isolated(
        &[FoundationPartition::Train],
        &[FoundationPartition::Validation, FoundationPartition::Test],
    ));
}
