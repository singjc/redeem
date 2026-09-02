use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use redeem_properties::foundation::{
    foundation_causal_next_token_loss, foundation_fragment_causal_rerank_score,
    load_causal_from_diffusion_checkpoint, FoundationCausalCollator, FoundationDiffusionConfig,
    FoundationDiffusionVocabulary, FoundationSpectrum, FoundationSpectrumCollator,
    FoundationSpectrumConfig, PeptideSpectrumCausalModel, PeptideSpectrumDiffusionModel,
    PeptidoformInput, PrecursorContextBatch, FOUNDATION_CAUSAL_RERANK_POLICY_V0123,
    FOUNDATION_CAUSAL_RERANK_WEIGHT_V0123, FOUNDATION_DIFFUSION_EOS, FOUNDATION_DIFFUSION_PAD,
    FOUNDATION_DIFFUSION_VOCAB_SIZE,
};
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

fn tiny_config() -> FoundationDiffusionConfig {
    FoundationDiffusionConfig {
        max_tokens: 12,
        model_dim: 32,
        num_attention_heads: 4,
        feed_forward_dim: 64,
        spectrum_layers: 1,
        decoder_layers: 1,
        dropout: 0.0,
        spectrum: FoundationSpectrumConfig {
            max_peaks: 8,
            ..FoundationSpectrumConfig::default()
        },
        ..FoundationDiffusionConfig::default()
    }
}

fn repeated_precursor(batch: usize, device: &Device) -> PrecursorContextBatch {
    PrecursorContextBatch {
        charge: Tensor::from_vec(vec![2.0f32; batch], batch, device).unwrap(),
        charge_present: Tensor::ones(batch, DType::F32, device).unwrap(),
        precursor_mz: Tensor::from_vec(vec![500.0f32; batch], batch, device).unwrap(),
        precursor_mz_present: Tensor::ones(batch, DType::F32, device).unwrap(),
        nce: Tensor::zeros(batch, DType::F32, device).unwrap(),
        nce_present: Tensor::zeros(batch, DType::F32, device).unwrap(),
        instrument_ids: Tensor::zeros(batch, DType::U32, device).unwrap(),
        instrument_present: Tensor::zeros(batch, DType::F32, device).unwrap(),
    }
}

#[test]
fn causal_teacher_forcing_shifts_targets_and_predicts_eos_after_final_token() {
    let device = Device::Cpu;
    let config = tiny_config();
    let peptide = PeptidoformInput::unmodified("ACD");
    let vocabulary = FoundationDiffusionVocabulary;
    let clean = vocabulary.encode(&peptide, config.max_tokens).unwrap();
    let batch = FoundationCausalCollator::new(config.clone())
        .unwrap()
        .collate(&[peptide], &device)
        .unwrap();
    let inputs = batch.input.input_tokens.to_vec2::<u32>().unwrap();
    let targets = batch.target_tokens.to_vec2::<u32>().unwrap();
    let mask = batch.input.token_mask.to_vec2::<f32>().unwrap();

    let active = mask[0].iter().take_while(|&&value| value > 0.0).count();
    assert_eq!(targets[0], clean);
    assert_eq!(inputs[0][0], FOUNDATION_DIFFUSION_PAD);
    for position in 1..active {
        assert_eq!(inputs[0][position], targets[0][position - 1]);
    }
    assert_eq!(targets[0][active - 1], FOUNDATION_DIFFUSION_EOS);
    assert_eq!(inputs[0][active - 1], targets[0][active - 2]);
}

#[test]
fn causal_prefix_collation_exposes_exactly_one_next_token_position() {
    let device = Device::Cpu;
    let config = tiny_config();
    let vocabulary = FoundationDiffusionVocabulary;
    let encoded = vocabulary
        .encode(&PeptidoformInput::unmodified("ACD"), config.max_tokens)
        .unwrap();
    let prefix_a = vec![encoded[0]];
    let prefix_ac = vec![encoded[0], encoded[1]];
    let input = FoundationCausalCollator::new(config.clone())
        .unwrap()
        .collate_prefix_rows(&[Vec::new(), prefix_a.clone(), prefix_ac.clone()], &device)
        .unwrap();
    let shifted = input.input_tokens.to_vec2::<u32>().unwrap();
    let mask = input.token_mask.to_vec2::<f32>().unwrap();

    assert!(shifted[0]
        .iter()
        .all(|&token| token == FOUNDATION_DIFFUSION_PAD));
    assert_eq!(mask[0][0], 1.0);
    assert!(mask[0][1..].iter().all(|&value| value == 0.0));

    assert_eq!(shifted[1][0], FOUNDATION_DIFFUSION_PAD);
    assert_eq!(shifted[1][1], prefix_a[0]);
    assert_eq!(&mask[1][..2], &[1.0, 1.0]);
    assert!(mask[1][2..].iter().all(|&value| value == 0.0));

    assert_eq!(shifted[2][0], FOUNDATION_DIFFUSION_PAD);
    assert_eq!(shifted[2][1], prefix_ac[0]);
    assert_eq!(shifted[2][2], prefix_ac[1]);
    assert_eq!(&mask[2][..3], &[1.0, 1.0, 1.0]);
    assert!(mask[2][3..].iter().all(|&value| value == 0.0));
}

#[test]
fn causal_start_embedding_is_outside_the_existing_diffusion_vocabulary() {
    let device = Device::Cpu;
    let config = tiny_config();
    let vocabulary = FoundationDiffusionVocabulary;
    assert_eq!(vocabulary.size(), FOUNDATION_DIFFUSION_VOCAB_SIZE);
    assert_eq!(FOUNDATION_DIFFUSION_VOCAB_SIZE, 28);

    let diffusion_vars = VarMap::new();
    let diffusion_vb = VarBuilder::from_varmap(&diffusion_vars, DType::F32, &device);
    let _diffusion = PeptideSpectrumDiffusionModel::new(config.clone(), diffusion_vb).unwrap();
    let causal_vars = VarMap::new();
    let causal_vb = VarBuilder::from_varmap(&causal_vars, DType::F32, &device);
    let _causal = PeptideSpectrumCausalModel::new(config, causal_vb).unwrap();

    let diffusion_data = diffusion_vars.data().lock().unwrap();
    assert!(!diffusion_data.contains_key("decoder.causal_start_embedding.weight"));
    drop(diffusion_data);
    let causal_data = causal_vars.data().lock().unwrap();
    assert!(causal_data.contains_key("decoder.causal_start_embedding.weight"));
    assert_eq!(
        causal_data
            .get("decoder.token_embedding.weight")
            .unwrap()
            .as_tensor()
            .dims()[0],
        FOUNDATION_DIFFUSION_VOCAB_SIZE
    );
}

#[test]
fn causal_scoring_cannot_see_the_current_target_or_future_candidate_tokens() {
    let device = Device::Cpu;
    let config = tiny_config();
    let peptides = vec![
        PeptidoformInput::unmodified("ACD"),
        PeptidoformInput::unmodified("ACE"),
    ];
    let causal = FoundationCausalCollator::new(config.clone())
        .unwrap()
        .collate(&peptides, &device)
        .unwrap();
    let inputs = causal.input.input_tokens.to_vec2::<u32>().unwrap();
    // The two candidates first differ at target position 2. Their model-visible
    // prefixes through prediction position 2 are therefore identical.
    assert_eq!(&inputs[0][..=2], &inputs[1][..=2]);
    assert_ne!(inputs[0][3], inputs[1][3]);

    let spectrum = FoundationSpectrum::from_pairs([(101.0, 3.0), (250.0, 9.0), (500.0, 4.0)]);
    let spectra = FoundationSpectrumCollator::new(config.spectrum.clone())
        .unwrap()
        .collate(&[spectrum.clone(), spectrum], &device)
        .unwrap();
    let precursor = repeated_precursor(2, &device);
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = PeptideSpectrumCausalModel::new(config, vb).unwrap();
    let output = model
        .forward_t(&causal.input, &spectra, &precursor, false)
        .unwrap();
    let logits = output.token_logits.to_vec3::<f32>().unwrap();
    for class in 0..FOUNDATION_DIFFUSION_VOCAB_SIZE {
        assert!(
            (logits[0][2][class] - logits[1][2][class]).abs() < 1e-5,
            "prediction at the divergent target position depends on leaked target/future token"
        );
    }

    let loss = foundation_causal_next_token_loss(&output, &causal)
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    assert!(loss.is_finite());
}

#[test]
fn cached_single_record_context_matches_repeated_context_logits() {
    let device = Device::Cpu;
    let config = tiny_config();
    let vocabulary = FoundationDiffusionVocabulary;
    let encoded = vocabulary
        .encode(&PeptidoformInput::unmodified("ACD"), config.max_tokens)
        .unwrap();
    let prefixes = vec![Vec::new(), vec![encoded[0]], vec![encoded[0], encoded[1]]];
    let input = FoundationCausalCollator::new(config.clone())
        .unwrap()
        .collate_prefix_rows(&prefixes, &device)
        .unwrap();

    let spectrum = FoundationSpectrum::from_pairs([(101.0, 3.0), (250.0, 9.0), (500.0, 4.0)]);
    let spectrum_collator = FoundationSpectrumCollator::new(config.spectrum.clone()).unwrap();
    let repeated_spectra = spectrum_collator
        .collate(
            &[spectrum.clone(), spectrum.clone(), spectrum.clone()],
            &device,
        )
        .unwrap();
    let single_spectrum = spectrum_collator.collate(&[spectrum], &device).unwrap();
    let repeated_precursor_batch = repeated_precursor(prefixes.len(), &device);
    let single_precursor = repeated_precursor(1, &device);

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = PeptideSpectrumCausalModel::new(config, vb).unwrap();

    let direct = model
        .forward_t(&input, &repeated_spectra, &repeated_precursor_batch, false)
        .unwrap();
    let context = model
        .prepare_context(&single_spectrum, &single_precursor, false)
        .unwrap();
    let cached = model
        .forward_t_with_context(&input, &context, false)
        .unwrap();

    let direct_logits = direct
        .token_logits
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let cached_logits = cached
        .token_logits
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert_eq!(direct_logits.len(), cached_logits.len());
    for (index, (left, right)) in direct_logits.iter().zip(&cached_logits).enumerate() {
        assert!(
            (left - right).abs() < 1e-5,
            "cached causal context changed logit {index}: direct={left} cached={right}"
        );
    }
}

#[test]
fn compact_prefix_next_logits_match_full_width_prefix_logits() {
    let device = Device::Cpu;
    let config = tiny_config();
    let vocabulary = FoundationDiffusionVocabulary;
    let encoded = vocabulary
        .encode(&PeptidoformInput::unmodified("ACDE"), config.max_tokens)
        .unwrap();
    let spectrum = FoundationSpectrum::from_pairs([(101.0, 3.0), (250.0, 9.0), (500.0, 4.0)]);
    let spectrum_collator = FoundationSpectrumCollator::new(config.spectrum.clone()).unwrap();
    let single_spectrum = spectrum_collator.collate(&[spectrum], &device).unwrap();
    let single_precursor = repeated_precursor(1, &device);

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = PeptideSpectrumCausalModel::new(config.clone(), vb).unwrap();
    let context = model
        .prepare_context(&single_spectrum, &single_precursor, false)
        .unwrap();
    let collator = FoundationCausalCollator::new(config.clone()).unwrap();

    for prefix_len in 0..=encoded.len().min(4) {
        let prefix = encoded[..prefix_len].to_vec();
        let prefixes = vec![prefix.clone(), prefix];
        let full = collator.collate_prefix_rows(&prefixes, &device).unwrap();
        let compact = collator
            .collate_compact_prefix_rows(&prefixes, &device)
            .unwrap();

        let full_output = model
            .forward_t_with_context(&full, &context, false)
            .unwrap();
        let full_logits = full_output.token_logits.to_vec3::<f32>().unwrap();
        let compact_logits = model
            .forward_next_t_with_context(&compact, &context, false)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();

        for row in 0..prefixes.len() {
            for class in 0..FOUNDATION_DIFFUSION_VOCAB_SIZE {
                let left = full_logits[row][prefix_len][class];
                let right = compact_logits[row][class];
                assert!(
                    (left - right).abs() < 1e-5,
                    "compact prefix changed prefix_len={prefix_len} row={row} class={class}: full={left} compact={right}"
                );
            }
        }
    }
}

#[test]
fn historical_diffusion_checkpoint_warm_starts_all_shared_causal_variables() {
    let device = Device::Cpu;
    let config = tiny_config();

    let diffusion_vars = VarMap::new();
    let diffusion_vb = VarBuilder::from_varmap(&diffusion_vars, DType::F32, &device);
    let _diffusion = PeptideSpectrumDiffusionModel::new(config.clone(), diffusion_vb).unwrap();

    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "redeem-foundation-causal-warmstart-{}-{stamp}.safetensors",
        std::process::id()
    ));
    diffusion_vars.save(&path).unwrap();

    let causal_vars = VarMap::new();
    let causal_vb = VarBuilder::from_varmap(&causal_vars, DType::F32, &device);
    let _causal = PeptideSpectrumCausalModel::new(config, causal_vb).unwrap();
    let report = load_causal_from_diffusion_checkpoint(&causal_vars, &path, &device).unwrap();
    assert!(report.loaded_variables > 0);
    assert_eq!(report.causal_only_variables, 1);
    assert!(report.ignored_checkpoint_variables >= 1); // timestep and length-head tensors

    let old_data = diffusion_vars.data().lock().unwrap();
    let new_data = causal_vars.data().lock().unwrap();
    for name in [
        "spectrum_encoder.input_projection.weight",
        "decoder.token_embedding.weight",
        "decoder.layers.0.self_attention.query.weight",
        "decoder.layers.0.cross_attention.query.weight",
        "decoder.layers.0.ff_in.weight",
        "decoder.output_norm.weight",
        "decoder.token_head.weight",
    ] {
        let old = old_data
            .get(name)
            .unwrap()
            .as_tensor()
            .flatten_all()
            .unwrap();
        let new = new_data
            .get(name)
            .unwrap()
            .as_tensor()
            .flatten_all()
            .unwrap();
        assert_eq!(
            old.to_vec1::<f32>().unwrap(),
            new.to_vec1::<f32>().unwrap(),
            "{name}"
        );
    }
    drop(new_data);
    drop(old_data);
    fs::remove_file(path).ok();
}

#[test]
fn validated_v0123_fragment_causal_policy_uses_locked_total_probability_weight() {
    assert_eq!(FOUNDATION_CAUSAL_RERANK_WEIGHT_V0123, 0.1);
    assert_eq!(
        FOUNDATION_CAUSAL_RERANK_POLICY_V0123,
        "fragment_plus_0.1_ar_total_v1"
    );

    let score =
        foundation_fragment_causal_rerank_score(7.0, -30.0, FOUNDATION_CAUSAL_RERANK_WEIGHT_V0123);
    assert!((score - 4.0).abs() < 1e-12);
}
