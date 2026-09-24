use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use redeem_properties::foundation::{
    FoundationModification, PeptideFoundationV0500Config, PeptideFoundationV0500Model,
    PeptideGraphFeaturizer, PeptidoformInput, PrecursorContextBatch,
    FOUNDATION_MULTIMODAL_ARCHITECTURE_V0500,
};

fn parse_usize_arg(args: &[String], index: usize, default: usize, label: &str) -> Result<usize> {
    match args.get(index) {
        Some(value) => {
            let parsed = value
                .parse::<usize>()
                .with_context(|| format!("invalid {label}: {value:?}"))?;
            if parsed == 0 {
                bail!("{label} must be positive");
            }
            Ok(parsed)
        }
        None => Ok(default),
    }
}

fn smoke_peptides(batch_size: usize) -> Vec<PeptidoformInput> {
    let mut phospho = PeptidoformInput::unmodified("PEPTIDER");
    phospho
        .modifications
        .push(FoundationModification::mass_delta(3, 79.9663));

    let mut oxidized = PeptidoformInput::unmodified("MKWVTFISLLLLFSSAYSR");
    oxidized
        .modifications
        .push(FoundationModification::mass_delta(0, 15.9949));

    let templates = [
        phospho,
        PeptidoformInput::unmodified("ACDEFGHIK"),
        PeptidoformInput::unmodified("MKWVTF"),
        PeptidoformInput::unmodified("KRRKPEPTIDEDE"),
        PeptidoformInput::unmodified("VVVVAAAALLLLFFYYWW"),
        oxidized,
    ];

    (0..batch_size)
        .map(|index| templates[index % templates.len()].clone())
        .collect()
}

fn smoke_context(batch_size: usize, device: &Device) -> Result<PrecursorContextBatch> {
    let charges = (0..batch_size)
        .map(|index| [2.0f32, 3.0, 2.0, 4.0][index % 4])
        .collect::<Vec<_>>();
    let precursor_mz = (0..batch_size)
        .map(|index| 400.0f32 + 7.25 * (index % 37) as f32)
        .collect::<Vec<_>>();
    let nce = (0..batch_size)
        .map(|index| [28.0f32, 30.0, 32.0][index % 3])
        .collect::<Vec<_>>();
    let instrument_ids = (0..batch_size)
        .map(|index| [1u32, 2u32, 3u32][index % 3])
        .collect::<Vec<_>>();

    Ok(PrecursorContextBatch {
        charge: Tensor::from_vec(charges, batch_size, device)?,
        charge_present: Tensor::ones(batch_size, DType::F32, device)?,
        precursor_mz: Tensor::from_vec(precursor_mz, batch_size, device)?,
        precursor_mz_present: Tensor::ones(batch_size, DType::F32, device)?,
        nce: Tensor::from_vec(nce, batch_size, device)?,
        nce_present: Tensor::ones(batch_size, DType::F32, device)?,
        instrument_ids: Tensor::from_vec(instrument_ids, batch_size, device)?
            .to_dtype(DType::U32)?,
        instrument_present: Tensor::ones(batch_size, DType::F32, device)?,
    })
}

fn main() -> Result<()> {
    let args = std::env::args().collect::<Vec<_>>();
    let mode = args.get(1).map(String::as_str).unwrap_or("local");
    let (config, device, default_batch_size, default_steps) = match mode {
        "local" => (
            PeptideFoundationV0500Config::local_smoke(),
            Device::Cpu,
            3usize,
            1usize,
        ),
        "a100" => {
            #[cfg(feature = "cuda")]
            {
                (
                    PeptideFoundationV0500Config::a100_smoke(),
                    Device::new_cuda(0)?,
                    3usize,
                    1usize,
                )
            }
            #[cfg(not(feature = "cuda"))]
            {
                bail!("a100 smoke requires redeem-properties feature 'cuda'")
            }
        }
        "a100-full" => {
            #[cfg(feature = "cuda")]
            {
                (
                    PeptideFoundationV0500Config::default(),
                    Device::new_cuda(0)?,
                    32usize,
                    12usize,
                )
            }
            #[cfg(not(feature = "cuda"))]
            {
                bail!("a100-full smoke requires redeem-properties feature 'cuda'")
            }
        }
        other => bail!(
            "usage: foundation_v0500_smoke [local|a100|a100-full] [batch_size] [steps], got {other:?}"
        ),
    };
    let batch_size = parse_usize_arg(&args, 2, default_batch_size, "batch_size")?;
    let steps = parse_usize_arg(&args, 3, default_steps, "steps")?;
    config.validate()?;

    let featurizer = PeptideGraphFeaturizer::new(config.featurizer_config())?;
    let peptides = smoke_peptides(batch_size);
    let batch = featurizer.featurize(&peptides, &device)?;
    let context = smoke_context(batch_size, &device)?;

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = PeptideFoundationV0500Model::new(config.clone(), vb)?;

    let mut final_gradients = None;
    let mut final_loss = None;
    let mut rt_shape = String::new();
    let mut mobility_shape = String::new();
    let mut ms2_shape = String::new();
    let mut pair_shape = String::new();

    for step in 0..steps {
        let output = model.forward_t(&batch, &context, true)?;
        let loss = (((output.rt.sqr()?.mean_all()?
            + output.mobility_native.sqr()?.mean_all()?)?
            + output.ms2.sqr()?.mean_all()?)?
            + output.pair_interaction_logits.sqr()?.mean_all()?)?;
        let gradients = loss.backward()?;
        let loss_value = loss.to_scalar::<f32>()?;
        if !loss_value.is_finite() {
            bail!("non-finite smoke loss at step {step}: {loss_value}");
        }

        if step + 1 == steps {
            rt_shape = format!("{:?}", output.rt.shape());
            mobility_shape = format!("{:?}", output.mobility_native.shape());
            ms2_shape = format!("{:?}", output.ms2.shape());
            pair_shape = format!("{:?}", output.representation.pair_embeddings.shape());
            final_gradients = Some(gradients);
            final_loss = Some(loss_value);
        }
    }

    let gradients = final_gradients.context("smoke produced no gradients")?;
    let loss_value = final_loss.context("smoke produced no loss")?;
    let data = varmap.data().lock().unwrap();
    let parameter_count: usize = data
        .values()
        .map(|value| value.as_tensor().elem_count())
        .sum();
    let last_block = config.interaction_blocks - 1;
    let required_gradients = vec![
        "student_v050.chemistry.atom_input.weight".to_string(),
        "student_v050.interaction.0.attention.query.weight".to_string(),
        "student_v050.interaction.0.pair_update_left.weight".to_string(),
        format!("student_v050.interaction.{last_block}.attention.query.weight"),
        format!("student_v050.interaction.{last_block}.pair_update_left.weight"),
        "student_v050.task.embedding.weight".to_string(),
        "student_v050.heads.rt.output.weight".to_string(),
        "student_v050.heads.mobility.output.weight".to_string(),
        "student_v050.heads.ms2.output.weight".to_string(),
    ];
    let mut gradient_checks = Vec::new();
    for name in required_gradients {
        let variable = data
            .get(name.as_str())
            .ok_or_else(|| anyhow::anyhow!("missing smoke parameter {name}"))?;
        let gradient = gradients
            .get(variable)
            .ok_or_else(|| anyhow::anyhow!("missing smoke gradient {name}"))?;
        let norm2 = gradient.sqr()?.sum_all()?.to_scalar::<f32>()?;
        if !norm2.is_finite() {
            bail!("non-finite gradient for {name}");
        }
        let norm = norm2.sqrt();
        if norm == 0.0 {
            bail!("zero gradient for {name}");
        }
        gradient_checks.push((name, norm));
    }

    println!("architecture={FOUNDATION_MULTIMODAL_ARCHITECTURE_V0500}");
    println!("mode={mode}");
    println!("device={device:?}");
    println!("profile_batch_size={batch_size}");
    println!("profile_steps={steps}");
    println!("graph_hidden_dim={}", config.graph_hidden_dim);
    println!("graph_layers={}", config.graph_layers);
    println!("residue_dim={}", config.residue_dim);
    println!("pair_dim={}", config.pair_dim);
    println!("interaction_blocks={}", config.interaction_blocks);
    println!("attention_heads={}", config.num_attention_heads);
    println!("feed_forward_dim={}", config.feed_forward_dim);
    println!("parameter_count={parameter_count}");
    println!("rt_shape={rt_shape}");
    println!("mobility_shape={mobility_shape}");
    println!("ms2_shape={ms2_shape}");
    println!("pair_shape={pair_shape}");
    println!("loss={loss_value:.8}");
    for (name, norm) in gradient_checks {
        println!("gradient_norm\t{name}\t{norm:.8}");
    }
    println!("teacher_source=external_frozen_v0350");
    println!("holdout_consumed=NO");
    println!("historical_validation_consumed=NO");
    println!("historical_test_consumed=NO");
    Ok(())
}
