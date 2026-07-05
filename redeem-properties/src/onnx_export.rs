//! ONNX export helpers for `redeem-properties` models.
//!
//! These helpers intentionally build the ONNX graph from the model `VarMap` and the same checkpoint
//! names used by the Candle loaders. Candle layer structs do not expose every tensor directly, so
//! the VarMap is the most reliable source of exportable weights.

use candle_nn::VarMap;
use candle_onnx_export::{
    candle::tensor_to_initializer, ops, Attribute, Dim, Error as OnnxError, ExportContext, Node,
    OnnxGraph, Result as OnnxResult, Shape, TensorData, TensorElementType, Value,
};

use crate::building_blocks::building_blocks::{AA_EMBEDDING_SIZE, MOD_FEATURE_SIZE};

/// Maximum sequence length available in the checkpoint positional encoding.
pub const TF_MAX_LEN: i64 = 100;

/// Input feature width for RT transformer models: AA index + modification features.
pub const RT_TF_INPUT_FEATURES: i64 = 1 + MOD_FEATURE_SIZE as i64;

/// Input feature width for CCS transformer models: AA index + modification features + charge.
pub const CCS_TF_INPUT_FEATURES: i64 = 1 + MOD_FEATURE_SIZE as i64 + 1;

/// Adds a complete `rt_cnn_tf` graph body.
pub fn export_rt_cnn_tf(varmap: &VarMap, ctx: &mut ExportContext<'_>, input: Value) -> OnnxResult<Value> {
    let aa_indices = slice_feature(ctx.graph, input.clone(), 0, 1, "rt_aa_index")?;
    let aa_indices = ops::squeeze_axes(ctx.graph, aa_indices, &[2], "rt_aa_indices")?;
    let mod_x = slice_feature(
        ctx.graph,
        input,
        1,
        1 + MOD_FEATURE_SIZE as i64,
        "rt_mod_features",
    )?;

    let encoded = export_mod_cnn_transformer_attn_sum(
        ctx.graph,
        varmap,
        "rt_encoder",
        aa_indices,
        mod_x,
        None,
        TransformerSpec {
            mod_hidden_dim: 8,
            hidden_dim: 192,
            ff_dim: 768,
            num_heads: 4,
            num_layers: 2,
            max_len: TF_MAX_LEN,
            use_padding_mask: true,
        },
        "rt_encoder_out",
    )?;

    let decoded =
        export_decoder_head_from_varmap(ctx.graph, varmap, "rt_decoder", encoded, "rt_decoder")?;
    let prediction = ops::squeeze_axes(ctx.graph, decoded, &[1], "rt_prediction")?;
    Ok(batch_prediction_value(prediction))
}

/// Adds a complete `ccs_cnn_tf` graph body.
pub fn export_ccs_cnn_tf(
    varmap: &VarMap,
    ctx: &mut ExportContext<'_>,
    input: Value,
) -> OnnxResult<Value> {
    let aa_indices = slice_feature(ctx.graph, input.clone(), 0, 1, "ccs_aa_index")?;
    let aa_indices = ops::squeeze_axes(ctx.graph, aa_indices, &[2], "ccs_aa_indices")?;
    let mod_x = slice_feature(
        ctx.graph,
        input.clone(),
        1,
        1 + MOD_FEATURE_SIZE as i64,
        "ccs_mod_features",
    )?;
    let charge = slice_feature(
        ctx.graph,
        input,
        1 + MOD_FEATURE_SIZE as i64,
        1 + MOD_FEATURE_SIZE as i64 + 1,
        "ccs_charge_raw",
    )?;
    let charge = ops::slice_i64(
        ctx.graph,
        charge,
        &[0],
        &[1],
        Some(&[1_i64][..]),
        None,
        "ccs_charge_first_position",
    )?;
    let charge = ops::squeeze_axes(ctx.graph, charge, &[2], "ccs_charge")?;

    let encoded = export_mod_cnn_transformer_attn_sum(
        ctx.graph,
        varmap,
        "ccs_encoder",
        aa_indices,
        mod_x,
        Some(charge.clone()),
        TransformerSpec {
            mod_hidden_dim: 8,
            hidden_dim: 128,
            ff_dim: 256,
            num_heads: 4,
            num_layers: 2,
            max_len: TF_MAX_LEN,
            use_padding_mask: false,
        },
        "ccs_encoder_out",
    )?;

    let decoder_input = ops::concat(ctx.graph, &[encoded, charge], 1, "ccs_decoder_input")?;
    let decoded = export_decoder_head_from_varmap(
        ctx.graph,
        varmap,
        "ccs_decoder",
        decoder_input,
        "ccs_decoder",
    )?;
    let prediction = ops::squeeze_axes(ctx.graph, decoded, &[1], "ccs_prediction")?;
    Ok(batch_prediction_value(prediction))
}

fn batch_prediction_value(value: Value) -> Value {
    Value::new(
        value.name,
        value.elem_type,
        Shape::from_dims([Dim::from("batch")]),
    )
}

/// Returns a standard model input value for the RT transformer checkpoint family.
pub fn add_rt_cnn_tf_input(graph: &mut OnnxGraph, name: &str) -> Value {
    graph.add_input(
        name,
        TensorElementType::Float32,
        Shape::from_dims([
            Dim::from("batch"),
            Dim::from("seq"),
            Dim::Fixed(RT_TF_INPUT_FEATURES),
        ]),
    )
}

/// Returns a standard model input value for the CCS transformer checkpoint family.
pub fn add_ccs_cnn_tf_input(graph: &mut OnnxGraph, name: &str) -> Value {
    graph.add_input(
        name,
        TensorElementType::Float32,
        Shape::from_dims([
            Dim::from("batch"),
            Dim::from("seq"),
            Dim::Fixed(CCS_TF_INPUT_FEATURES),
        ]),
    )
}

#[derive(Debug, Clone, Copy)]
struct TransformerSpec {
    mod_hidden_dim: usize,
    hidden_dim: usize,
    ff_dim: usize,
    num_heads: usize,
    num_layers: usize,
    max_len: i64,
    use_padding_mask: bool,
}

fn export_mod_cnn_transformer_attn_sum(
    graph: &mut OnnxGraph,
    varmap: &VarMap,
    prefix: &str,
    aa_indices: Value,
    mod_x: Value,
    charge: Option<Value>,
    spec: TransformerSpec,
    output_name: &str,
) -> OnnxResult<Value> {
    let mod_x = export_mod_embedding_fix_first_k(
        graph,
        varmap,
        &format!("{prefix}.mod_nn"),
        mod_x,
        spec.mod_hidden_dim,
        &format!("{prefix}_mod_embedding"),
    )?;

    let mut features = vec![one_hot_aa(graph, aa_indices.clone(), &format!("{prefix}_aa_one_hot"))?, mod_x.clone()];
    if let Some(charge) = charge {
        features.push(expand_charge_to_sequence(
            graph,
            charge,
            mod_x.clone(),
            &format!("{prefix}_charge_seq"),
        )?);
    }
    let x = ops::concat(graph, &features, 2, &format!("{prefix}_aa_mod_features"))?;
    let padding_mask = if spec.use_padding_mask {
        Some(padding_mask_from_aa_indices(
            graph,
            aa_indices,
            &format!("{prefix}_padding_mask"),
        )?)
    } else {
        None
    };

    let x = export_seq_cnn(graph, varmap, &format!("{prefix}.input_cnn"), x, &format!("{prefix}_cnn"))?;
    let x = linear_last_dim(
        graph,
        varmap,
        &format!("{prefix}.proj_cnn_to_transformer"),
        x,
        &format!("{prefix}_projected"),
    )?;
    let x = export_transformer(
        graph,
        varmap,
        &format!("{prefix}.input_transformer"),
        x,
        padding_mask,
        spec,
        &format!("{prefix}_transformer"),
    )?;
    export_attention_sum(
        graph,
        varmap,
        &format!("{prefix}.attn_sum.attn.0"),
        x,
        output_name,
    )
}

fn export_mod_embedding_fix_first_k(
    graph: &mut OnnxGraph,
    varmap: &VarMap,
    prefix: &str,
    mod_x: Value,
    mod_hidden_dim: usize,
    output_prefix: &str,
) -> OnnxResult<Value> {
    let k = 6_i64;
    let first_k = ops::slice_i64(
        graph,
        mod_x.clone(),
        &[0],
        &[k],
        Some(&[-1_i64][..]),
        None,
        &format!("{output_prefix}_first_k"),
    )?;
    let rest = ops::slice_i64(
        graph,
        mod_x,
        &[k],
        &[MOD_FEATURE_SIZE as i64],
        Some(&[-1_i64][..]),
        None,
        &format!("{output_prefix}_rest"),
    )?;
    let transformed = linear_last_dim_with_names(
        graph,
        varmap,
        &format!("{prefix}.nn.weight"),
        None,
        rest,
        &format!("{output_prefix}_transformed"),
    )?;
    let output = ops::concat(graph, &[first_k, transformed], -1, output_prefix)?;
    let expected = mod_hidden_dim as i64;
    if expected != k + (mod_hidden_dim as i64 - k) {
        return Err(OnnxError::InvalidGraph("invalid mod embedding hidden dimension".into()));
    }
    Ok(output)
}

fn export_seq_cnn(
    graph: &mut OnnxGraph,
    varmap: &VarMap,
    prefix: &str,
    input: Value,
    output_prefix: &str,
) -> OnnxResult<Value> {
    let transposed = ops::transpose(graph, input, &[0, 2, 1], &format!("{output_prefix}_ncl"))?;
    let short = conv1d_from_varmap(
        graph,
        varmap,
        &format!("{prefix}.cnn_short"),
        transposed.clone(),
        [1, 1],
        &format!("{output_prefix}_short"),
    )?;
    let medium = conv1d_from_varmap(
        graph,
        varmap,
        &format!("{prefix}.cnn_medium"),
        transposed.clone(),
        [2, 2],
        &format!("{output_prefix}_medium"),
    )?;
    let long = conv1d_from_varmap(
        graph,
        varmap,
        &format!("{prefix}.cnn_long"),
        transposed.clone(),
        [3, 3],
        &format!("{output_prefix}_long"),
    )?;
    let concat = ops::concat(graph, &[transposed, short, medium, long], 1, &format!("{output_prefix}_cat"))?;
    ops::transpose(graph, concat, &[0, 2, 1], output_prefix)
}

fn export_transformer(
    graph: &mut OnnxGraph,
    varmap: &VarMap,
    prefix: &str,
    input: Value,
    padding_mask: Option<Value>,
    spec: TransformerSpec,
    output_prefix: &str,
) -> OnnxResult<Value> {
    let seq_len = sequence_length_vector(graph, input.clone(), &format!("{output_prefix}_seq_len"))?;
    let pos_encoding = sinusoidal_position_encoding(spec.max_len as usize, spec.hidden_dim);
    let pe_name = graph.unique_name(&format!("{output_prefix}_pos_encoding"));
    let pe = TensorData::from_f32(pe_name, &[1, spec.max_len, spec.hidden_dim as i64], &pos_encoding)?;
    let pe = graph.add_initializer_value(pe);
    let pe = slice_position_encoding_to_seq(
        graph,
        pe,
        seq_len,
        &format!("{output_prefix}_pos_encoding_sliced"),
    )?;
    let mut x = ops::add(graph, input, pe, &format!("{output_prefix}_pos_add"))?;

    for layer_idx in 0..spec.num_layers {
        x = export_transformer_layer(
            graph,
            varmap,
            &format!("{prefix}.layer_{layer_idx}"),
            x,
            padding_mask.clone(),
            spec,
            &format!("{output_prefix}_layer_{layer_idx}"),
        )?;
    }

    Ok(x)
}

fn export_transformer_layer(
    graph: &mut OnnxGraph,
    varmap: &VarMap,
    prefix: &str,
    input: Value,
    padding_mask: Option<Value>,
    spec: TransformerSpec,
    output_prefix: &str,
) -> OnnxResult<Value> {
    let attn = export_multi_head_attention(
        graph,
        varmap,
        prefix,
        input.clone(),
        padding_mask,
        spec,
        &format!("{output_prefix}_attn"),
    )?;
    let attn_residual = ops::add(graph, input, attn, &format!("{output_prefix}_attn_residual"))?;
    let norm1 = layer_norm_from_varmap(
        graph,
        varmap,
        &format!("{prefix}.norm1"),
        attn_residual,
        &format!("{output_prefix}_norm1"),
    )?;
    let ff = export_feed_forward(
        graph,
        varmap,
        prefix,
        norm1.clone(),
        &format!("{output_prefix}_ff"),
    )?;
    let ff_residual = ops::add(graph, norm1, ff, &format!("{output_prefix}_ff_residual"))?;
    layer_norm_from_varmap(
        graph,
        varmap,
        &format!("{prefix}.norm2"),
        ff_residual,
        &format!("{output_prefix}_norm2"),
    )
}

fn export_multi_head_attention(
    graph: &mut OnnxGraph,
    varmap: &VarMap,
    prefix: &str,
    input: Value,
    padding_mask: Option<Value>,
    spec: TransformerSpec,
    output_prefix: &str,
) -> OnnxResult<Value> {
    let head_dim = spec.hidden_dim / spec.num_heads;
    let q = project_heads(graph, varmap, &format!("{prefix}.proj_q"), input.clone(), spec, &format!("{output_prefix}_q"))?;
    let k = project_heads(graph, varmap, &format!("{prefix}.proj_k"), input.clone(), spec, &format!("{output_prefix}_k"))?;
    let v = project_heads(graph, varmap, &format!("{prefix}.proj_v"), input, spec, &format!("{output_prefix}_v"))?;

    let k_t = ops::transpose(graph, k, &[0, 1, 3, 2], &format!("{output_prefix}_k_t"))?;
    let mut scores = ops::matmul(graph, q, k_t, &format!("{output_prefix}_scores"))?;
    let scale_name = graph.unique_name(&format!("{output_prefix}_scale"));
    let scale = graph.add_initializer_value(TensorData::scalar_f32(scale_name, (head_dim as f32).sqrt())?);
    scores = ops::div(graph, scores, scale, &format!("{output_prefix}_scores_scaled"))?;

    if let Some(mask) = padding_mask {
        let mask = ops::unsqueeze_axes(graph, mask, &[1, 2], &format!("{output_prefix}_mask_unsqueeze"))?;
        let neg_name = graph.unique_name(&format!("{output_prefix}_mask_neg"));
        let neg = graph.add_initializer_value(TensorData::scalar_f32(neg_name, -1e9)?);
        let mask = ops::mul(graph, mask, neg, &format!("{output_prefix}_mask_scaled"))?;
        scores = ops::add(graph, scores, mask, &format!("{output_prefix}_scores_masked"))?;
    }

    let attn = ops::softmax(graph, scores, -1, &format!("{output_prefix}_weights"))?;
    let context = ops::matmul(graph, attn, v, &format!("{output_prefix}_context"))?;
    let context = ops::transpose(graph, context, &[0, 2, 1, 3], &format!("{output_prefix}_context_bthd"))?;
    let context = reshape_with_shape(
        graph,
        context,
        &[0, 0, spec.hidden_dim as i64],
        &format!("{output_prefix}_context_flat"),
    )?;
    linear_last_dim(graph, varmap, &format!("{prefix}.proj_out"), context, output_prefix)
}

fn project_heads(
    graph: &mut OnnxGraph,
    varmap: &VarMap,
    prefix: &str,
    input: Value,
    spec: TransformerSpec,
    output_prefix: &str,
) -> OnnxResult<Value> {
    let head_dim = spec.hidden_dim / spec.num_heads;
    let projected = linear_last_dim(graph, varmap, prefix, input, &format!("{output_prefix}_linear"))?;
    let reshaped = reshape_with_shape(
        graph,
        projected,
        &[
            0,
            0,
            spec.num_heads as i64,
            head_dim as i64,
        ],
        &format!("{output_prefix}_reshape"),
    )?;
    ops::transpose(graph, reshaped, &[0, 2, 1, 3], output_prefix)
}

fn export_feed_forward(
    graph: &mut OnnxGraph,
    varmap: &VarMap,
    prefix: &str,
    input: Value,
    output_prefix: &str,
) -> OnnxResult<Value> {
    let hidden = linear_last_dim(graph, varmap, &format!("{prefix}.lin1"), input, &format!("{output_prefix}_lin1"))?;
    let hidden = ops::relu(graph, hidden, &format!("{output_prefix}_relu"))?;
    linear_last_dim(graph, varmap, &format!("{prefix}.lin2"), hidden, output_prefix)
}

fn sequence_length_vector(
    graph: &mut OnnxGraph,
    input: Value,
    output_name: &str,
) -> OnnxResult<Value> {
    let shape = ops::shape(graph, input, &format!("{output_name}_shape"))?;
    ops::slice_i64(
        graph,
        shape,
        &[1],
        &[2],
        Some(&[0_i64][..]),
        None,
        output_name,
    )
}

fn slice_position_encoding_to_seq(
    graph: &mut OnnxGraph,
    pos_encoding: Value,
    seq_len: Value,
    output_name: &str,
) -> OnnxResult<Value> {
    let starts_name = graph.unique_name(&format!("{output_name}_starts"));
    graph.add_initializer(TensorData::vec_i64(starts_name.clone(), &[0])?);
    let axes_name = graph.unique_name(&format!("{output_name}_axes"));
    graph.add_initializer(TensorData::vec_i64(axes_name.clone(), &[1])?);

    ops::slice(
        graph,
        pos_encoding,
        starts_name,
        seq_len,
        Some(axes_name),
        Option::<String>::None,
        output_name,
    )
}

fn reshape_with_shape(
    graph: &mut OnnxGraph,
    input: Value,
    shape: &[i64],
    output_name: &str,
) -> OnnxResult<Value> {
    let shape_name = graph.unique_name(&format!("{output_name}_shape"));
    graph.add_initializer(TensorData::vec_i64(shape_name.clone(), shape)?);
    ops::reshape(graph, input, shape_name, output_name)
}

fn export_attention_sum(
    graph: &mut OnnxGraph,
    varmap: &VarMap,
    prefix: &str,
    input: Value,
    output_name: &str,
) -> OnnxResult<Value> {
    let weights = linear_last_dim(graph, varmap, prefix, input.clone(), &format!("{output_name}_scores"))?;
    let weights = ops::softmax(graph, weights, 1, &format!("{output_name}_weights"))?;
    let weighted = ops::mul(graph, input, weights, &format!("{output_name}_weighted"))?;
    ops::reduce_sum_axes(graph, weighted, &[1], false, output_name)
}

pub fn export_decoder_head_from_varmap(
    graph: &mut OnnxGraph,
    varmap: &VarMap,
    prefix: &str,
    input: Value,
    output_prefix: &str,
) -> OnnxResult<Value> {
    let data = varmap
        .data()
        .lock()
        .map_err(|_| OnnxError::InvalidGraph("failed to lock VarMap".into()))?;
    let nn_prefix = format!("{prefix}.nn.");
    let mut indices = data
        .keys()
        .filter_map(|key| {
            key.strip_prefix(&nn_prefix)
                .and_then(|rest| rest.strip_suffix(".weight"))
                .and_then(|idx| idx.parse::<usize>().ok())
        })
        .collect::<Vec<_>>();
    indices.sort_unstable();
    indices.dedup();
    drop(data);

    if indices.is_empty() {
        return Err(OnnxError::MissingTensor(format!("{prefix}.nn.*.weight")));
    }

    let mut value = input;
    for idx in indices {
        let weight = format!("{prefix}.nn.{idx}.weight");
        let bias = format!("{prefix}.nn.{idx}.bias");
        let rank = tensor_rank(varmap, &weight)?;
        value = match rank {
            1 => {
                ensure_initializer(graph, varmap, &weight)?;
                ops::prelu(graph, value, weight, &format!("{output_prefix}_prelu_{idx}"))?
            }
            2 => linear_2d_with_names(
                graph,
                varmap,
                &weight,
                has_tensor(varmap, &bias).then_some(bias.as_str()),
                value,
                &format!("{output_prefix}_linear_{idx}"),
            )?,
            _ => {
                return Err(OnnxError::UnsupportedTensor(format!(
                    "decoder tensor {weight} has unsupported rank {rank}"
                )))
            }
        };
    }

    let scale_weight = format!("{prefix}.scale.weight");
    if has_tensor(varmap, &scale_weight) {
        let scale_bias = format!("{prefix}.scale.bias");
        value = linear_2d_with_names(
            graph,
            varmap,
            &scale_weight,
            has_tensor(varmap, &scale_bias).then_some(scale_bias.as_str()),
            value,
            &format!("{output_prefix}_scale"),
        )?;
    }

    Ok(value)
}

fn linear_2d_with_names(
    graph: &mut OnnxGraph,
    varmap: &VarMap,
    weight_name: &str,
    bias_name: Option<&str>,
    input: Value,
    output_name: &str,
) -> OnnxResult<Value> {
    ensure_initializer(graph, varmap, weight_name)?;
    if let Some(bias) = bias_name {
        ensure_initializer(graph, varmap, bias)?;
    }
    ops::linear(graph, input, weight_name, bias_name, output_name)
}

fn linear_last_dim(
    graph: &mut OnnxGraph,
    varmap: &VarMap,
    prefix: &str,
    input: Value,
    output_name: &str,
) -> OnnxResult<Value> {
    let weight = format!("{prefix}.weight");
    let bias = format!("{prefix}.bias");
    linear_last_dim_with_names(
        graph,
        varmap,
        &weight,
        has_tensor(varmap, &bias).then_some(bias.as_str()),
        input,
        output_name,
    )
}

fn linear_last_dim_with_names(
    graph: &mut OnnxGraph,
    varmap: &VarMap,
    weight_name: &str,
    bias_name: Option<&str>,
    input: Value,
    output_name: &str,
) -> OnnxResult<Value> {
    let weight = initializer_value(graph, varmap, weight_name)?;
    let weight_t_name = graph.unique_name(&format!("{}_weight_t", sanitize(output_name)));
    let weight_t = ops::transpose(graph, weight, &[1, 0], &weight_t_name)?;
    let mut output = ops::matmul(graph, input, weight_t, output_name)?;
    if let Some(bias) = bias_name {
        ensure_initializer(graph, varmap, bias)?;
        output = ops::add(graph, output, bias, &format!("{output_name}_bias"))?;
    }
    Ok(output)
}

fn layer_norm_from_varmap(
    graph: &mut OnnxGraph,
    varmap: &VarMap,
    prefix: &str,
    input: Value,
    output_name: &str,
) -> OnnxResult<Value> {
    let weight = format!("{prefix}.weight");
    let bias = format!("{prefix}.bias");
    ensure_initializer(graph, varmap, &weight)?;
    ensure_initializer(graph, varmap, &bias)?;
    ops::layer_normalization(graph, input, weight, Some(bias), -1, 1e-5, output_name)
}

fn conv1d_from_varmap(
    graph: &mut OnnxGraph,
    varmap: &VarMap,
    prefix: &str,
    input: Value,
    pads: [i64; 2],
    output_name: &str,
) -> OnnxResult<Value> {
    let weight = format!("{prefix}.weight");
    let bias = format!("{prefix}.bias");
    ensure_initializer(graph, varmap, &weight)?;
    ensure_initializer(graph, varmap, &bias)?;
    ops::conv1d(graph, input, &weight, Some(&bias), pads, [1], output_name)
}

fn one_hot_aa(graph: &mut OnnxGraph, aa_indices: Value, output_name: &str) -> OnnxResult<Value> {
    let indices = ops::cast(
        graph,
        aa_indices,
        TensorElementType::Int64,
        &format!("{output_name}_indices"),
    )?;
    let depth_name = graph.unique_name(&format!("{output_name}_depth"));
    let depth = graph.add_initializer_value(TensorData::scalar_i64(
        depth_name,
        AA_EMBEDDING_SIZE as i64,
    )?);
    let values_name = graph.unique_name(&format!("{output_name}_values"));
    let values = graph.add_initializer_value(TensorData::from_f32(values_name, &[2], &[0.0, 1.0])?);
    let output = Value::new(output_name, TensorElementType::Float32, Shape::unknown());
    let node_name = graph.unique_name(&format!("{output_name}_onehot"));
    graph.add_node(
        Node::new(
            node_name,
            "OneHot",
            vec![indices.name, depth.name, values.name],
            vec![output.name.clone()],
        )
        .with_attr(Attribute::int("axis", -1)),
    );
    Ok(output)
}

fn padding_mask_from_aa_indices(
    graph: &mut OnnxGraph,
    aa_indices: Value,
    output_name: &str,
) -> OnnxResult<Value> {
    let zero_name = graph.unique_name(&format!("{output_name}_zero"));
    let zero = graph.add_initializer_value(TensorData::scalar_f32(zero_name, 0.0)?);
    let equal = Value::new(
        graph.unique_name(&format!("{output_name}_bool")),
        TensorElementType::Bool,
        aa_indices.shape.clone(),
    );
    let node_name = graph.unique_name(&format!("{output_name}_equal"));
    graph.add_node(Node::new(
        node_name,
        "Equal",
        vec![aa_indices.name, zero.name],
        vec![equal.name.clone()],
    ));
    ops::cast(graph, equal, TensorElementType::Float32, output_name)
}

fn expand_charge_to_sequence(
    graph: &mut OnnxGraph,
    charge: Value,
    sequence_like: Value,
    output_name: &str,
) -> OnnxResult<Value> {
    let shape = ops::shape(graph, sequence_like, &format!("{output_name}_shape"))?;
    let first_two_dims = ops::slice_i64(
        graph,
        shape,
        &[0],
        &[2],
        Some(&[0_i64][..]),
        None,
        &format!("{output_name}_first_two_dims"),
    )?;
    let one_name = graph.unique_name(&format!("{output_name}_one_dim"));
    let one = graph.add_initializer_value(TensorData::vec_i64(one_name, &[1])?);
    let target_shape = ops::concat(graph, &[first_two_dims, one], 0, &format!("{output_name}_target_shape"))?;
    let charge = ops::unsqueeze_axes(graph, charge, &[1], &format!("{output_name}_unsqueeze"))?;
    let output = Value::new(output_name, TensorElementType::Float32, Shape::unknown());
    let node_name = graph.unique_name(&format!("{output_name}_expand"));
    graph.add_node(Node::new(
        node_name,
        "Expand",
        vec![charge.name, target_shape.name],
        vec![output.name.clone()],
    ));
    Ok(output)
}

fn slice_feature(
    graph: &mut OnnxGraph,
    input: Value,
    start: i64,
    end: i64,
    output_name: &str,
) -> OnnxResult<Value> {
    ops::slice_i64(
        graph,
        input,
        &[start],
        &[end],
        Some(&[2_i64][..]),
        None,
        output_name,
    )
}

fn ensure_initializer(graph: &mut OnnxGraph, varmap: &VarMap, name: &str) -> OnnxResult<()> {
    if graph.has_initializer(name) {
        return Ok(());
    }
    let tensor = tensor_from_varmap(varmap, name)?;
    graph.add_initializer(tensor_to_initializer(name.to_string(), &tensor)?);
    Ok(())
}

fn initializer_value(graph: &mut OnnxGraph, varmap: &VarMap, name: &str) -> OnnxResult<Value> {
    let tensor = tensor_from_varmap(varmap, name)?;
    graph.add_initializer_value(tensor_to_initializer(name.to_string(), &tensor)?);
    Ok(Value::new(
        name,
        TensorElementType::Float32,
        Shape::from_dims(
            tensor
                .shape()
                .dims()
                .iter()
                .copied()
                .map(|dim| Dim::Fixed(dim as i64))
                .collect::<Vec<_>>(),
        ),
    ))
}

fn tensor_from_varmap(varmap: &VarMap, name: &str) -> OnnxResult<candle_core::Tensor> {
    let data = varmap
        .data()
        .lock()
        .map_err(|_| OnnxError::InvalidGraph("failed to lock VarMap".into()))?;
    data.get(name)
        .map(|var| var.as_tensor().clone())
        .ok_or_else(|| OnnxError::MissingTensor(name.to_string()))
}

fn has_tensor(varmap: &VarMap, name: &str) -> bool {
    varmap
        .data()
        .lock()
        .map(|data| data.contains_key(name))
        .unwrap_or(false)
}

fn tensor_rank(varmap: &VarMap, name: &str) -> OnnxResult<usize> {
    Ok(tensor_from_varmap(varmap, name)?.shape().dims().len())
}

fn sinusoidal_position_encoding(seq_len: usize, model_dim: usize) -> Vec<f32> {
    let mut pe = vec![0.0; seq_len * model_dim];
    for pos in 0..seq_len {
        for i in 0..model_dim {
            let angle = pos as f32 / 10000_f32.powf(2.0 * ((i / 2) as f32) / model_dim as f32);
            pe[pos * model_dim + i] = if i % 2 == 0 { angle.sin() } else { angle.cos() };
        }
    }
    pe
}

fn sanitize(name: &str) -> String {
    name.chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' })
        .collect()
}
