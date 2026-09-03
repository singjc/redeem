use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use redeem_properties::foundation::{
    foundation_diffusion_length_loss, foundation_diffusion_x0_loss,
    foundation_peptidoform_neutral_mass, foundation_precursor_mass_error_da,
    foundation_spectrum_peptide_alignment_loss, FoundationAdamW, FoundationAdamWConfig,
    FoundationDiffusionCollator, FoundationDiffusionConfig, FoundationDiffusionVocabulary,
    FoundationSpectrum, FoundationSpectrumCollator, FoundationSpectrumConfig,
    PeptideSpectrumDiffusionModel, PeptidoformInput, PrecursorContextBatch,
    FOUNDATION_DIFFUSION_VOCAB_SIZE,
};

#[test]
fn diffusion_model_executes_one_real_backward_update() {
    let device = Device::Cpu;
    let config = FoundationDiffusionConfig {
        max_tokens: 24,
        model_dim: 32,
        num_attention_heads: 4,
        feed_forward_dim: 64,
        spectrum_layers: 1,
        decoder_layers: 1,
        spectrum: FoundationSpectrumConfig {
            max_peaks: 8,
            ..FoundationSpectrumConfig::default()
        },
        ..FoundationDiffusionConfig::default()
    };

    let spectra = vec![
        FoundationSpectrum::from_pairs([(101.1, 3.0), (247.2, 10.0), (504.3, 6.0)]),
        FoundationSpectrum::from_pairs([(120.2, 5.0), (333.3, 9.0), (701.4, 2.0)]),
    ];
    let spectrum_batch = FoundationSpectrumCollator::new(config.spectrum.clone())
        .unwrap()
        .collate(&spectra, &device)
        .unwrap();
    assert_eq!(
        spectrum_batch.peak_features.dims(),
        &[
            2,
            config.spectrum.max_peaks,
            config.spectrum.peak_feature_dim
        ]
    );
    assert_eq!(config.spectrum.peak_feature_dim, 32);
    let peptides = vec![
        PeptidoformInput::unmodified("PEPTIDEK"),
        PeptidoformInput::unmodified("MELTQK"),
    ];
    let diffusion_batch = FoundationDiffusionCollator::new(config.clone())
        .unwrap()
        .collate(&peptides, &[5, 15], 20260901, &device)
        .unwrap();
    let precursor = PrecursorContextBatch {
        charge: Tensor::new(&[2.0f32, 3.0], &device).unwrap(),
        charge_present: Tensor::ones(2, DType::F32, &device).unwrap(),
        precursor_mz: Tensor::new(&[500.0f32, 600.0], &device).unwrap(),
        precursor_mz_present: Tensor::ones(2, DType::F32, &device).unwrap(),
        nce: Tensor::zeros(2, DType::F32, &device).unwrap(),
        nce_present: Tensor::zeros(2, DType::F32, &device).unwrap(),
        instrument_ids: Tensor::zeros(2, DType::U32, &device).unwrap(),
        instrument_present: Tensor::zeros(2, DType::F32, &device).unwrap(),
    };

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = PeptideSpectrumDiffusionModel::new(config.clone(), vb).unwrap();
    let output = model
        .forward_t(&diffusion_batch, &spectrum_batch, &precursor, true)
        .unwrap();
    assert_eq!(
        output.token_logits.dims(),
        &[2, config.max_tokens, FOUNDATION_DIFFUSION_VOCAB_SIZE]
    );
    assert_eq!(output.length_logits.dims(), &[2, config.max_tokens]);
    assert_eq!(
        output.spectrum_memory.dims(),
        &[2, config.spectrum.max_peaks + 1, config.model_dim]
    );
    let x0_loss = foundation_diffusion_x0_loss(&output, &diffusion_batch).unwrap();
    let length_loss = foundation_diffusion_length_loss(&output, &diffusion_batch).unwrap();
    let loss = (&x0_loss + &length_loss.affine(0.1, 0.0).unwrap()).unwrap();
    let before = loss.to_scalar::<f32>().unwrap();
    assert!(before.is_finite());
    assert!(length_loss.to_scalar::<f32>().unwrap().is_finite());
    let alignment = foundation_spectrum_peptide_alignment_loss(
        &output.spectrum_embedding,
        &output.spectrum_embedding,
        0.1,
    )
    .unwrap()
    .to_scalar::<f32>()
    .unwrap();
    assert!(alignment.is_finite());

    let mut optimizer = FoundationAdamW::new(&varmap, FoundationAdamWConfig::default()).unwrap();
    let step = optimizer.backward_step(&loss, Some(5.0)).unwrap();
    assert_eq!(step.step, 1);
    assert!(step.gradient_norm.is_finite());
    assert!(step.gradient_norm > 0.0);
}

#[test]
fn diffusion_all_masked_batch_removes_clean_token_content_and_tracks_length() {
    let device = Device::Cpu;
    let config = FoundationDiffusionConfig {
        max_tokens: 16,
        model_dim: 32,
        num_attention_heads: 4,
        feed_forward_dim: 64,
        spectrum_layers: 1,
        decoder_layers: 1,
        ..FoundationDiffusionConfig::default()
    };
    let peptides = vec![
        PeptidoformInput::unmodified("PEPTIDEK"),
        PeptidoformInput::unmodified("MELTQK"),
    ];
    let batch = FoundationDiffusionCollator::new(config)
        .unwrap()
        .collate_all_masked(&peptides, 20, &device)
        .unwrap();
    let noisy = batch.noisy_tokens.to_vec2::<u32>().unwrap();
    let clean = batch.clean_tokens.to_vec2::<u32>().unwrap();
    let mask = batch.token_mask.to_vec2::<f32>().unwrap();
    for row in 0..noisy.len() {
        for col in 0..noisy[row].len() {
            if mask[row][col] > 0.0 {
                assert_eq!(
                    noisy[row][col],
                    redeem_properties::foundation::FOUNDATION_DIFFUSION_MASK
                );
                assert_ne!(
                    clean[row][col],
                    redeem_properties::foundation::FOUNDATION_DIFFUSION_PAD
                );
            } else {
                assert_eq!(
                    noisy[row][col],
                    redeem_properties::foundation::FOUNDATION_DIFFUSION_PAD
                );
            }
        }
    }
    assert_eq!(batch.length_targets.to_vec1::<u32>().unwrap(), vec![8, 6]);
}

#[test]
fn diffusion_token_mass_round_trip_matches_precursor_constraint() {
    let peptide = PeptidoformInput::unmodified("PEPTIDEK");
    let vocabulary = FoundationDiffusionVocabulary;
    let tokens = vocabulary.encode(&peptide, 24).unwrap();
    let decoded = vocabulary.decode(&tokens).unwrap();
    assert_eq!(decoded, peptide);

    let neutral = foundation_peptidoform_neutral_mass(&peptide).unwrap();
    let charge = 2;
    let proton = 1.007_276_466_77f64;
    let precursor_mz = (neutral + charge as f64 * proton) / charge as f64;
    let error = foundation_precursor_mass_error_da(&decoded, precursor_mz, charge).unwrap();
    assert!(error.abs() < 1e-8);
}

#[test]
fn diffusion_fingerprint_tracks_explicit_product_mz() {
    use redeem_properties::foundation::{
        foundation_diffusion_record_fingerprint, FoundationTrainingRecord, FragmentTarget,
        RetentionTimeLabels, TrainingContext,
    };

    let mut record = FoundationTrainingRecord {
        peptidoform: PeptidoformInput::unmodified("PEPTIDEK"),
        retention_time: RetentionTimeLabels::default(),
        ccs: None,
        fragments: vec![FragmentTarget {
            cleavage_index: 1,
            channel: 0,
            intensity: 1.0,
            product_mz: Some(250.2),
        }],
        observed_spectrum_peaks: Vec::new(),
        context: TrainingContext {
            charge: Some(2),
            precursor_mz: Some(500.0),
            ..TrainingContext::default()
        },
        run_id: None,
    };
    let first = foundation_diffusion_record_fingerprint(&record);
    record.fragments[0].product_mz = Some(250.3);
    let second = foundation_diffusion_record_fingerprint(&record);
    assert_ne!(first, second);
}

#[test]
fn diffusion_fingerprint_tracks_raw_observed_spectrum_peaks() {
    use redeem_properties::foundation::{
        foundation_diffusion_record_fingerprint, FoundationTrainingRecord, ObservedSpectrumPeak,
        RetentionTimeLabels, TrainingContext,
    };

    let mut record = FoundationTrainingRecord {
        peptidoform: PeptidoformInput::unmodified("PEPTIDEK"),
        retention_time: RetentionTimeLabels::default(),
        ccs: None,
        fragments: Vec::new(),
        observed_spectrum_peaks: vec![ObservedSpectrumPeak {
            mz: 250.2,
            intensity: 1.0,
        }],
        context: TrainingContext {
            charge: Some(2),
            precursor_mz: Some(500.0),
            ..TrainingContext::default()
        },
        run_id: None,
    };
    let first = foundation_diffusion_record_fingerprint(&record);
    record.observed_spectrum_peaks[0].mz = 250.3;
    let second = foundation_diffusion_record_fingerprint(&record);
    assert_ne!(first, second);
}

#[test]
fn diffusion_x0_and_alignment_losses_reach_spectrum_and_decoder_backbones() {
    let device = Device::Cpu;
    let config = FoundationDiffusionConfig {
        max_tokens: 16,
        model_dim: 32,
        num_attention_heads: 4,
        feed_forward_dim: 64,
        spectrum_layers: 1,
        decoder_layers: 1,
        spectrum: FoundationSpectrumConfig {
            max_peaks: 8,
            ..FoundationSpectrumConfig::default()
        },
        ..FoundationDiffusionConfig::default()
    };
    let spectra = vec![
        FoundationSpectrum::from_pairs([(101.1, 3.0), (247.2, 10.0), (504.3, 6.0)]),
        FoundationSpectrum::from_pairs([(120.2, 5.0), (333.3, 9.0), (701.4, 2.0)]),
    ];
    let spectrum_batch = FoundationSpectrumCollator::new(config.spectrum.clone())
        .unwrap()
        .collate(&spectra, &device)
        .unwrap();
    let peptides = vec![
        PeptidoformInput::unmodified("PEPTIDEK"),
        PeptidoformInput::unmodified("MELTQK"),
    ];
    let diffusion_batch = FoundationDiffusionCollator::new(config.clone())
        .unwrap()
        .collate_all_masked(&peptides, config.diffusion_steps, &device)
        .unwrap();
    let precursor = PrecursorContextBatch {
        charge: Tensor::new(&[2.0f32, 3.0], &device).unwrap(),
        charge_present: Tensor::ones(2, DType::F32, &device).unwrap(),
        precursor_mz: Tensor::new(&[500.0f32, 600.0], &device).unwrap(),
        precursor_mz_present: Tensor::ones(2, DType::F32, &device).unwrap(),
        nce: Tensor::zeros(2, DType::F32, &device).unwrap(),
        nce_present: Tensor::zeros(2, DType::F32, &device).unwrap(),
        instrument_ids: Tensor::zeros(2, DType::U32, &device).unwrap(),
        instrument_present: Tensor::zeros(2, DType::F32, &device).unwrap(),
    };

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = PeptideSpectrumDiffusionModel::new(config, vb).unwrap();
    let output = model
        .forward_t(&diffusion_batch, &spectrum_batch, &precursor, true)
        .unwrap();

    let x0_loss = foundation_diffusion_x0_loss(&output, &diffusion_batch).unwrap();
    let x0_gradients = x0_loss.backward().unwrap();
    let data = varmap.data().lock().unwrap();
    let spectrum_input = data
        .get("spectrum_encoder.input_projection.weight")
        .expect("spectrum input projection variable");
    let decoder_attention = data
        .get("decoder.layers.0.self_attention.query.weight")
        .expect("decoder self-attention query variable");
    let spectrum_gradient = x0_gradients
        .get(spectrum_input)
        .expect("x0 loss must reach spectrum encoder");
    let decoder_gradient = x0_gradients
        .get(decoder_attention)
        .expect("x0 loss must reach decoder layer");
    assert!(
        spectrum_gradient
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
            > 0.0
    );
    assert!(
        decoder_gradient
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
            > 0.0
    );
    drop(data);

    let alignment = foundation_spectrum_peptide_alignment_loss(
        &output.spectrum_embedding,
        &output.spectrum_embedding.detach(),
        0.1,
    )
    .unwrap();
    let alignment_gradients = alignment.backward().unwrap();
    let data = varmap.data().lock().unwrap();
    let spectrum_input = data
        .get("spectrum_encoder.input_projection.weight")
        .expect("spectrum input projection variable");
    let spectrum_alignment_gradient = alignment_gradients
        .get(spectrum_input)
        .expect("alignment loss must reach spectrum encoder");
    assert!(
        spectrum_alignment_gradient
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
            > 0.0
    );
}
