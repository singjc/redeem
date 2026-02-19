use anyhow::{Context, Result};
use candle_core::Device;
use csv::Reader;
use redeem_properties::{
    models::{
        model_interface::{ModelInterface, PredictionResult},
        ms2_bert_model::MS2BertModel,
    },
    utils::{
        data_handling::{PeptideBatchData, PeptideData, TargetNormalization},
        peptdeep_utils::{
            get_modification_indices, get_modification_string, load_modifications,
            remove_mass_shift, ModificationMap,
        },
    },
};
use std::{collections::HashMap, fs::File, path::PathBuf, sync::Arc};

fn run_prediction(model: &mut MS2BertModel, batch_data: &[PeptideData]) -> Result<()> {
    let batch = PeptideBatchData::from(batch_data);

    let instruments = if batch.instruments.iter().all(|i| i.is_some()) {
        Some(
            batch
                .instruments
                .iter()
                .map(|opt| opt.as_ref().map(|a| Arc::clone(a)))
                .collect::<Vec<_>>(),
        )
    } else {
        None
    };

    let predictions = model.predict(
        &batch.naked_sequence,
        &batch.mods,
        &batch.mod_sites,
        if batch.charges.iter().all(|c| c.is_some()) {
            Some(batch.charges.iter().map(|c| c.unwrap()).collect())
        } else {
            None
        },
        if batch.nces.iter().all(|n| n.is_some()) {
            Some(batch.nces.iter().map(|n| n.unwrap()).collect())
        } else {
            None
        },
        instruments,
    )?;

    if let PredictionResult::MS2Result(ms2_preds) = predictions {
        let total_error: f32 = ms2_preds
            .iter()
            .zip(batch.ms2_intensities.iter())
            .map(|(pred, obs)| {
                pred.iter()
                    .zip(obs.as_ref().unwrap())
                    .map(|(p_row, o_row)| {
                        p_row
                            .iter()
                            .zip(o_row.iter())
                            .map(|(p, o)| (p - o).abs())
                            .sum::<f32>()
                    })
                    .sum::<f32>()
            })
            .sum();

        for (i, peptide) in batch.naked_sequence.iter().enumerate() {
            let pred_sum: f32 = ms2_preds[i].iter().flatten().sum();
            let obs_sum: f32 = batch.ms2_intensities[i]
                .as_ref()
                .map(|v| v.iter().flatten().sum())
                .unwrap_or(0.0);
            println!(
                "Peptide: {}\n  Predicted Intensity Sum: {:.4}\n  Observed Intensity Sum: {:.4}",
                std::str::from_utf8(peptide).unwrap_or(""),
                pred_sum,
                obs_sum
            );
        }

        let mean_abs_error = total_error / ms2_preds.len() as f32;
        println!("Mean Absolute Error: {:.6}", mean_abs_error);
    }
    Ok(())
}

fn main() -> Result<()> {
    let model_path = PathBuf::from("data/models/alphapeptdeep/generic/ms2.pth");
    let constants_path =
        PathBuf::from("data/models/alphapeptdeep/generic/ms2.pth.model_const.yaml");
    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
    println!("Device: {:?}", device);

    let mut model = MS2BertModel::new(&model_path, Some(&constants_path), 0, 8, 4, true, device)?;

    let file = File::open("data/predicted_fragment_intensities.csv")?;
    let mut rdr = Reader::from_reader(file);

    let mut data_map: HashMap<String, Vec<Vec<f32>>> = HashMap::new();
    let mut charge_map: HashMap<String, i32> = HashMap::new();

    for rec in rdr.records() {
        let rec = rec?;
        let seq = &rec[0];
        let charge: i32 = rec[1].parse()?;
        let ftype = &rec[2];
        let idx: usize = rec[3].parse()?;
        let fz: i32 = rec[4].parse()?;
        let intensity: f32 = rec[6].parse()?;

        let naked = remove_mass_shift(seq);
        let len = naked.len().saturating_sub(1);
        charge_map.insert(seq.clone().to_string(), charge);

        let entry = data_map
            .entry(seq.to_string())
            .or_insert_with(|| vec![vec![0.0; 8]; len]);
        if let Some(row) = entry.get_mut(idx.saturating_sub(1)) {
            let col = match (ftype, fz) {
                ("B", 1) => 0,
                ("B", 2) => 1,
                ("Y", 1) => 2,
                ("Y", 2) => 3,
                _ => continue,
            };
            row[col] = intensity;
        }
    }

    let modifications = load_modifications()?;
    let mut training_data = Vec::new();

    for (mod_seq, ms2) in data_map {
        let naked = remove_mass_shift(&mod_seq)
            .trim_start_matches('-')
            .to_string();
        let mods = get_modification_string(&mod_seq, &modifications);
        let mod_sites = get_modification_indices(&mod_seq);

        training_data.push(PeptideData::new(
            &mod_seq,
            &naked,
            &mods,
            &mod_sites,
            charge_map.get(&mod_seq).copied(),
            None,
            Some(20),
            Some("QE"),
            None,
            None,
            None,
            Some(ms2),
        ));
    }

    println!("Loaded {} peptides.", training_data.len());
    run_prediction(&mut model, &training_data)?;

    model.fine_tune(
        &training_data,
        modifications,
        3,
        0.001,
        5,
        TargetNormalization::None,
        None,
    )?;
    run_prediction(&mut model, &training_data)?;

    Ok(())
}
