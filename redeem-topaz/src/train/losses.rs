use candle_core::{DType, Result, Tensor};

fn softplus(x: &Tensor) -> Result<Tensor> {
    let abs = x.abs()?;
    let neg_abs = (abs * -1.0)?;
    let exp = neg_abs.exp()?;
    let ones = exp.ones_like()?;
    let log1p = exp.broadcast_add(&ones)?.log()?;
    let max0 = x.maximum(0f32)?;
    log1p.broadcast_add(&max0)
}

/// Binary cross-entropy with logits, averaged over all elements.
pub fn bce_with_logits(logits: &Tensor, targets: &Tensor) -> Result<Tensor> {
    let logits = logits.to_dtype(DType::F32)?;
    let targets = targets.to_dtype(DType::F32)?;
    let sp = softplus(&logits)?;
    let yt = targets.broadcast_mul(&logits)?;
    let loss = sp.broadcast_sub(&yt)?;
    loss.mean_all()
}

/// Binary cross-entropy with logits and optional positive class weight.
/// If `pos_weight` is None or <= 0, falls back to standard BCE.
pub fn bce_with_logits_weighted(
    logits: &Tensor,
    targets: &Tensor,
    pos_weight: Option<f32>,
) -> Result<Tensor> {
    let Some(w) = pos_weight else {
        return bce_with_logits(logits, targets);
    };
    if w <= 0.0 {
        return bce_with_logits(logits, targets);
    }
    let logits = logits.to_dtype(DType::F32)?;
    let targets = targets.to_dtype(DType::F32)?;
    let sp_pos = softplus(&logits.neg()?)?;
    let sp_neg = softplus(&logits)?;
    let y_pos = targets.broadcast_mul(&sp_pos)?;
    let y_neg = (targets.ones_like()? - &targets)?.broadcast_mul(&sp_neg)?;
    let w_t = Tensor::full(w, targets.dims(), targets.device())?;
    let y_pos = y_pos.broadcast_mul(&w_t)?;
    let loss = y_pos.broadcast_add(&y_neg)?;
    loss.mean_all()
}

/// Pairwise pos-neg separation: softplus(-(pos - neg)), averaged over all pos-neg pairs.
pub fn pairwise_pos_neg_softplus(bag_logits: &Tensor, y_bag: &Tensor) -> Result<Tensor> {
    let s = bag_logits.to_dtype(DType::F32)?;
    let y = y_bag.to_dtype(DType::F32)?;
    let pos = y.gt(0.5f32)?.to_dtype(DType::F32)?;
    let neg = (pos.ones_like()? - &pos)?;

    let s_pos = s.unsqueeze(1)?;
    let s_neg = s.unsqueeze(0)?;
    let diff = s_pos.broadcast_sub(&s_neg)?;

    let pos_m = pos.unsqueeze(1)?;
    let neg_m = neg.unsqueeze(0)?;
    let mask = pos_m.broadcast_mul(&neg_m)?;

    let neg_diff = (diff * -1.0)?;
    let loss_mat = softplus(&neg_diff)?;
    let loss_sum = loss_mat.broadcast_mul(&mask)?.sum_all()?;
    let denom = mask.sum_all()?.maximum(1.0f32)?;
    loss_sum.broadcast_div(&denom)
}

/// In-bag ranking loss for target bags only.
pub fn inbag_ranking_loss(
    cand_logits: &Tensor,
    mask: &Tensor,
    y_bag: &Tensor,
    margin: f32,
) -> Result<Tensor> {
    let (b, k) = cand_logits.dims2()?;
    let s = cand_logits.to_dtype(DType::F32)?;
    let m = mask.to_dtype(DType::F32)?;
    let y = y_bag.to_dtype(DType::F32)?;

    let neg_big = Tensor::full(-1e9f32, (b, k), s.device())?;
    let ones = m.ones_like()?;
    let s_masked = (s.broadcast_mul(&m)? + neg_big.broadcast_mul(&(ones - &m)?)?)?;

    let s_best = s_masked.max(1)?;
    let k_best = s_masked.argmax(1)?.to_dtype(DType::I64)?;

    let idx = Tensor::arange(0i64, k as i64, s.device())?
        .reshape((1, k))?
        .broadcast_as((b, k))?;
    let k_best = k_best.reshape((b, 1))?.broadcast_as((b, k))?;
    let onehot = idx.eq(&k_best)?;
    let onehot_f = onehot.to_dtype(DType::F32)?;

    let other_mask = m.broadcast_mul(&(onehot_f.ones_like()? - &onehot_f)?)?;
    let tgt_mask = y.gt(0.5f32)?.to_dtype(DType::F32)?;
    let other_mask = other_mask.broadcast_mul(&tgt_mask.unsqueeze(1)?)?;

    let diffs = s_best.unsqueeze(1)?.broadcast_sub(&s_masked)?;
    let margin_diff = (margin as f64 - diffs)?;
    let loss_mat = softplus(&margin_diff)?;

    let loss_sum = loss_mat.broadcast_mul(&other_mask)?.sum_all()?;
    let denom = other_mask.sum_all()?.maximum(1.0f32)?;
    loss_sum.broadcast_div(&denom)
}

/// Winner-vs-runner-up margin loss for target bags with >=2 valid candidates.
pub fn winner_margin_loss(
    cand_logits: &Tensor,
    mask: &Tensor,
    y_bag: &Tensor,
    margin: f32,
) -> Result<Tensor> {
    let (b, k) = cand_logits.dims2()?;
    let s = cand_logits.to_dtype(DType::F32)?;
    let m = mask.to_dtype(DType::F32)?;
    let y = y_bag.to_dtype(DType::F32)?;

    let neg_big = Tensor::full(-1e9f32, (b, k), s.device())?;
    let ones = m.ones_like()?;
    let s_masked = (s.broadcast_mul(&m)? + neg_big.broadcast_mul(&(ones - &m)?)?)?;

    let s_best = s_masked.max(1)?;
    let k_best = s_masked.argmax(1)?.to_dtype(DType::I64)?;

    let idx = Tensor::arange(0i64, k as i64, s.device())?
        .reshape((1, k))?
        .broadcast_as((b, k))?;
    let k_best = k_best.reshape((b, 1))?.broadcast_as((b, k))?;
    let onehot = idx.eq(&k_best)?;
    let onehot_f = onehot.to_dtype(DType::F32)?;

    let s2_masked = (s_masked + neg_big.broadcast_mul(&onehot_f)?)?;
    let s_second = s2_masked.max(1)?;

    let n_valid = m.sum(1)?;
    let has2 = n_valid.ge(2.0f32)?.to_dtype(DType::F32)?;
    let tgt = y.gt(0.5f32)?.to_dtype(DType::F32)?;
    let ok = has2.broadcast_mul(&tgt)?;

    let gap = s_best.broadcast_sub(&s_second)?;
    let margin_diff = (margin as f64 - gap)?;
    let loss_vec = softplus(&margin_diff)?;

    let loss_sum = loss_vec.broadcast_mul(&ok)?.sum_all()?;
    let denom = ok.sum_all()?.maximum(1.0f32)?;
    loss_sum.broadcast_div(&denom)
}
