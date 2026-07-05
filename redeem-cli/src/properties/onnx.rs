use anyhow::{bail, Context, Result};
use candle_core::Device;
use candle_onnx_export::{
    Dim, ExportContext, ExportOptions, OnnxGraph, Shape, TensorElementType, ToOnnx, WeightFormat,
};
use clap::ArgMatches;
use redeem_properties::models::{
    ccs_cnn_lstm_model::CCSCNNLSTMModel, ccs_cnn_tf_model::CCSCNNTFModel,
    model_interface::ModelInterface, rt_cnn_lstm_model::RTCNNLSTMModel,
    rt_cnn_transformer_model::RTCNNTFModel,
};
use redeem_properties::onnx_export::export_decoder_head_from_varmap;
use std::path::{Path, PathBuf};

/// Keep small shape/control initializers embedded so ONNX shape inference can read them.
const DEFAULT_EXTERNAL_DATA_THRESHOLD: usize = 1024;

/// Export a supported `redeem-properties` model component to ONNX.
///
/// `--component decoder` exports only the decoder/head. `--component full` exports the complete
/// model graph for architectures that have a `ToOnnx` implementation in `redeem-properties`.
pub fn run_to_onnx(matches: &ArgMatches) -> Result<()> {
    let model_path = matches
        .get_one::<PathBuf>("model_path")
        .context("missing required --model")?
        .clone();
    let output_path = matches
        .get_one::<PathBuf>("output_file")
        .context("missing required --output")?
        .clone();
    let model_arch = matches
        .get_one::<String>("model_arch")
        .context("missing required --model_arch")?
        .as_str();
    let constants_path = matches.get_one::<PathBuf>("constants").cloned();
    let component = matches
        .get_one::<String>("component")
        .map(String::as_str)
        .unwrap_or("decoder");
    let device_name = matches
        .get_one::<String>("device")
        .map(String::as_str)
        .unwrap_or("cpu");
    let opset_version = matches
        .get_one::<u32>("opset")
        .copied()
        .unwrap_or(18) as i64;
    let external_data_threshold = matches
        .get_one::<usize>("external_data_threshold")
        .copied()
        .unwrap_or(DEFAULT_EXTERNAL_DATA_THRESHOLD);

    let device = parse_device(device_name)?;

    let mut graph = OnnxGraph::new(format!("{model_arch}_{component}"));
    let output = match component {
        "decoder" => {
            let (decoder_prefix, decoder_dim) = decoder_layout(model_arch)?;
            let input = graph.add_input(
                format!("{decoder_prefix}_input"),
                TensorElementType::Float32,
                Shape::from_dims([Dim::from("batch"), Dim::from(decoder_dim)]),
            );
            load_and_export_decoder(
                &mut graph,
                model_arch,
                &model_path,
                constants_path.as_ref(),
                device,
                decoder_prefix,
                input,
            )?
        }
        "full" => load_and_export_full(
            &mut graph,
            model_arch,
            &model_path,
            constants_path.as_ref(),
            device,
        )?,
        other => bail!("unsupported ONNX export component: {other}"),
    };
    graph.add_output(output);

    let weight_format = if matches.get_flag("external_data") {
        WeightFormat::External {
            data_filename: external_data_filename(&output_path, matches.get_one::<PathBuf>("data_file"))?,
            size_threshold: external_data_threshold,
        }
    } else {
        WeightFormat::Embedded
    };

    graph
        .save(
            &output_path,
            ExportOptions {
                opset_version,
                weight_format,
                ..ExportOptions::default()
            },
        )
        .map_err(|err| anyhow::anyhow!("failed to save ONNX model: {err}"))?;

    eprintln!(
        "[ReDeeM::Properties] Exported {model_arch} {component} component to {:?}",
        output_path
    );
    Ok(())
}

fn load_and_export_decoder(
    graph: &mut OnnxGraph,
    model_arch: &str,
    model_path: &Path,
    constants_path: Option<&PathBuf>,
    device: Device,
    decoder_prefix: &str,
    input: candle_onnx_export::Value,
) -> Result<candle_onnx_export::Value> {
    match model_arch {
        "rt_cnn_lstm" => {
            let model = RTCNNLSTMModel::new(
                model_path.to_path_buf(),
                constants_path.cloned(),
                0,
                0,
                0,
                true,
                device,
            )?;
            export_decoder_head_from_varmap(graph, model.get_varmap(), decoder_prefix, input, decoder_prefix)
                .map_err(|err| anyhow::anyhow!("failed to export decoder/head: {err}"))
        }
        "rt_cnn_tf" => {
            let model = RTCNNTFModel::new(
                model_path.to_path_buf(),
                constants_path.cloned(),
                0,
                0,
                0,
                true,
                device,
            )?;
            export_decoder_head_from_varmap(graph, model.get_varmap(), decoder_prefix, input, decoder_prefix)
                .map_err(|err| anyhow::anyhow!("failed to export decoder/head: {err}"))
        }
        "ccs_cnn_lstm" => {
            let model = CCSCNNLSTMModel::new(
                model_path.to_path_buf(),
                constants_path.cloned(),
                0,
                0,
                0,
                true,
                device,
            )?;
            export_decoder_head_from_varmap(graph, model.get_varmap(), decoder_prefix, input, decoder_prefix)
                .map_err(|err| anyhow::anyhow!("failed to export decoder/head: {err}"))
        }
        "ccs_cnn_tf" => {
            let model = CCSCNNTFModel::new(
                model_path.to_path_buf(),
                constants_path.cloned(),
                0,
                0,
                0,
                true,
                device,
            )?;
            export_decoder_head_from_varmap(graph, model.get_varmap(), decoder_prefix, input, decoder_prefix)
                .map_err(|err| anyhow::anyhow!("failed to export decoder/head: {err}"))
        }
        "ms2_bert" => bail!(
            "ms2_bert decoder export is not wired yet because the MS2 output/modloss heads need \
             sequence-shaped output metadata"
        ),
        other => bail!("unsupported model architecture: {other}"),
    }
}

fn load_and_export_full(
    graph: &mut OnnxGraph,
    model_arch: &str,
    model_path: &Path,
    constants_path: Option<&PathBuf>,
    device: Device,
) -> Result<candle_onnx_export::Value> {
    match model_arch {
        "rt_cnn_tf" => {
            let model = RTCNNTFModel::new(
                model_path.to_path_buf(),
                constants_path.cloned(),
                0,
                0,
                0,
                true,
                device,
            )?;
            let input = redeem_properties::onnx_export::add_rt_cnn_tf_input(graph, "input");
            export_full_model(graph, &model, input)
        }
        "ccs_cnn_tf" => {
            let model = CCSCNNTFModel::new(
                model_path.to_path_buf(),
                constants_path.cloned(),
                0,
                0,
                0,
                true,
                device,
            )?;
            let input = redeem_properties::onnx_export::add_ccs_cnn_tf_input(graph, "input");
            export_full_model(graph, &model, input)
        }
        "rt_cnn_lstm" | "ccs_cnn_lstm" | "ms2_bert" => bail!(
            "full ONNX export for {model_arch} is not implemented yet; supported full exports: \
             rt_cnn_tf, ccs_cnn_tf"
        ),
        other => bail!("unsupported model architecture: {other}"),
    }
}

fn export_full_model<M: ToOnnx>(
    graph: &mut OnnxGraph,
    model: &M,
    input: candle_onnx_export::Value,
) -> Result<candle_onnx_export::Value> {
    let mut ctx = ExportContext::new(graph);
    let outputs = model
        .to_onnx(&mut ctx, &[input])
        .map_err(|err| anyhow::anyhow!("failed to export full model: {err}"))?;
    outputs
        .into_iter()
        .next()
        .context("full model ONNX export produced no outputs")
}

fn decoder_layout(model_arch: &str) -> Result<(&'static str, usize)> {
    match model_arch {
        "rt_cnn_lstm" => Ok(("rt_decoder", 256)),
        "rt_cnn_tf" => Ok(("rt_decoder", 192)),
        "ccs_cnn_lstm" => Ok(("ccs_decoder", 257)),
        "ccs_cnn_tf" => Ok(("ccs_decoder", 129)),
        "ms2_bert" => bail!(
            "ms2_bert decoder export is not wired yet because the MS2 output/modloss heads need \
             sequence-shaped output metadata"
        ),
        other => bail!("unsupported model architecture: {other}"),
    }
}

fn parse_device(device_name: &str) -> Result<Device> {
    match device_name {
        "cpu" => Ok(Device::Cpu),
        #[cfg(feature = "cuda")]
        "cuda" => Ok(Device::new_cuda(0)?),
        #[cfg(feature = "cuda")]
        name if name.starts_with("cuda:") => {
            let index = name
                .strip_prefix("cuda:")
                .and_then(|value| value.parse::<usize>().ok())
                .context("invalid CUDA device; expected cuda or cuda:<index>")?;
            Ok(Device::new_cuda(index)?)
        }
        other => bail!("unsupported device '{other}'; use cpu{}", cuda_help_suffix()),
    }
}

fn cuda_help_suffix() -> &'static str {
    #[cfg(feature = "cuda")]
    {
        ", cuda, or cuda:<index>"
    }
    #[cfg(not(feature = "cuda"))]
    {
        " or build redeem-cli with the cuda feature"
    }
}

fn external_data_filename(output_path: &Path, data_file: Option<&PathBuf>) -> Result<String> {
    if let Some(data_file) = data_file {
        return data_file
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .context("--data-file must include a file name");
    }

    let onnx_name = output_path
        .file_name()
        .map(|name| name.to_string_lossy())
        .context("--output must include a file name")?;
    Ok(format!("{onnx_name}.data"))
}
