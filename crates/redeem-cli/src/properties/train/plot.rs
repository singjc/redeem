use plotly::common::{Fill, Mode, Title};
use plotly::{Histogram, Layout, Plot, Scatter};
use redeem_properties::utils::stats::{TrainingPhase, TrainingStepMetrics};

pub fn plot_losses(epoch_losses: &[(usize, f32, Option<f32>, f32, Option<f32>)]) -> Plot {
    let epochs: Vec<_> = epoch_losses
        .iter()
        .map(|(e, _, _, _, _)| *e as f64)
        .collect();

    let train_mean: Vec<_> = epoch_losses
        .iter()
        .map(|(_, m, _, _, _)| *m as f64)
        .collect();
    let train_std: Vec<_> = epoch_losses
        .iter()
        .map(|(_, _, _, std, _)| *std as f64)
        .collect();
    let train_upper: Vec<_> = train_mean
        .iter()
        .zip(&train_std)
        .map(|(m, s)| m + s)
        .collect();
    let train_lower: Vec<_> = train_mean
        .iter()
        .zip(&train_std)
        .map(|(m, s)| m - s)
        .collect();

    let val_mean: Vec<_> = epoch_losses
        .iter()
        .map(|(_, _, val, _, _)| val.unwrap_or(f32::NAN) as f64)
        .collect();
    let val_std: Vec<_> = epoch_losses
        .iter()
        .map(|(_, _, _, _, val_std)| val_std.unwrap_or(0.0) as f64)
        .collect();
    let val_upper: Vec<_> = val_mean.iter().zip(&val_std).map(|(m, s)| m + s).collect();
    let val_lower: Vec<_> = val_mean.iter().zip(&val_std).map(|(m, s)| m - s).collect();

    let mut plot = Plot::new();

    // Training loss line
    plot.add_trace(
        Scatter::new(epochs.clone(), train_mean.clone())
            .name("Train Loss")
            .mode(Mode::Lines)
            .line(plotly::common::Line::new().color("rgba(31, 119, 180, 1.0)")),
    );

    // Training loss band
    let mut train_band_y = train_upper.clone();
    let mut train_band_x = epochs.clone();
    let mut lower_reversed: Vec<_> = train_lower.iter().cloned().rev().collect();
    let mut x_reversed: Vec<_> = epochs.iter().cloned().rev().collect();
    train_band_y.extend(lower_reversed);
    train_band_x.extend(x_reversed);

    plot.add_trace(
        Scatter::new(train_band_x, train_band_y)
            .name("Train ± σ")
            .mode(Mode::Lines)
            .fill(Fill::ToSelf)
            .line(plotly::common::Line::new().width(0.0))
            .fill_color("rgba(31, 119, 180, 0.2)"),
    );

    // Validation loss line
    plot.add_trace(
        Scatter::new(epochs.clone(), val_mean.clone())
            .name("Val Loss")
            .mode(Mode::Lines)
            .line(plotly::common::Line::new().color("rgba(255, 127, 14, 1.0)")),
    );

    // Validation loss band
    let mut val_band_y = val_upper.clone();
    let mut val_band_x = epochs.clone();
    let mut val_lower_rev: Vec<_> = val_lower.iter().cloned().rev().collect();
    let mut val_x_rev: Vec<_> = epochs.iter().cloned().rev().collect();
    val_band_y.extend(val_lower_rev);
    val_band_x.extend(val_x_rev);

    plot.add_trace(
        Scatter::new(val_band_x, val_band_y)
            .name("Val ± σ")
            .mode(Mode::Lines)
            .fill(Fill::ToSelf)
            .line(plotly::common::Line::new().width(0.0))
            .fill_color("rgba(255, 127, 14, 0.2)"),
    );

    plot.set_layout(
        Layout::new()
            .title("Training and Validation Loss Over Epochs")
            .x_axis(plotly::layout::Axis::new().title("Epoch"))
            .y_axis(plotly::layout::Axis::new().title("Loss")),
    );

    plot
}

/// Plot histogram of prediction deltas (predicted - target).
pub fn plot_delta_histogram(delta: &[f64], title: &str, x_title: &str, y_title: &str) -> Plot {
    let mut plot = Plot::new();

    plot.add_trace(
        Histogram::new(delta.to_vec())
            .name("Pred - True")
            .n_bins_x(60)
            .opacity(0.8),
    );

    plot.set_layout(
        Layout::new()
            .title(title)
            .x_axis(plotly::layout::Axis::new().title(x_title))
            .y_axis(plotly::layout::Axis::new().title(y_title)),
    );

    plot
}

/// Plot a single training metric (e.g. loss, learning rate, accuracy) over steps.
pub fn plot_training_metric(
    metrics: &TrainingStepMetrics,
    metric_name: &str,
    title: &str,
    x_title: &str,
    y_title: &str,
) -> Plot {
    let mut plot = Plot::new();

    let mut train_x = vec![];
    let mut train_y = vec![];
    let mut val_x = vec![];
    let mut val_y = vec![];

    for i in 0..metrics.steps.len() {
        let x = metrics.steps[i] as f64;
        let y_opt = match metric_name {
            "loss" => Some(metrics.losses[i] as f64),
            "lr" => Some(metrics.learning_rates[i]),
            "accuracy" => metrics.accuracies[i].map(|a| a as f64),
            "mae" => metrics.maes[i].map(|m| m as f64),
            "rmse" => metrics.rmses[i].map(|m| m as f64),
            "r2" => metrics.r2s[i].map(|m| m as f64),
            _ => None,
        };

        if let Some(y) = y_opt {
            match metrics.phases[i] {
                TrainingPhase::Train => {
                    train_x.push(x);
                    train_y.push(y);
                }
                TrainingPhase::Validation => {
                    val_x.push(x);
                    val_y.push(y);
                }
            }
        }
    }

    if !train_x.is_empty() {
        plot.add_trace(
            Scatter::new(train_x.clone(), train_y.clone())
                .mode(Mode::Lines)
                .name("Train"),
        );
    }
    if !val_x.is_empty() {
        plot.add_trace(
            Scatter::new(val_x.clone(), val_y.clone())
                .mode(Mode::Lines)
                .name("Validation"),
        );
    }

    plot.set_layout(
        Layout::new()
            .title(title)
            .x_axis(plotly::layout::Axis::new().title(x_title))
            .y_axis(plotly::layout::Axis::new().title(y_title)),
    );

    plot
}

/// Plot cumulative distribution (CDF) of absolute errors.
pub fn plot_error_cdf(abs_errors: &[f64], title: &str, x_title: &str, y_title: &str) -> Plot {
    let mut sorted: Vec<f64> = abs_errors.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    let xs: Vec<f64> = sorted.clone();
    let ys: Vec<f64> = (0..n).map(|i| (i + 1) as f64 / n as f64).collect();

    let mut plot = Plot::new();
    plot.add_trace(
        Scatter::new(xs, ys)
            .mode(Mode::Lines)
            .name("CDF")
            .line(plotly::common::Line::new().color("rgba(31,119,180,1.0)")),
    );

    plot.set_layout(
        Layout::new()
            .title(title)
            .x_axis(plotly::layout::Axis::new().title(x_title))
            .y_axis(plotly::layout::Axis::new().title(y_title)),
    );

    plot
}

/// Scatter of residuals vs a numeric peptide feature with binned median overlay.
pub fn plot_residuals_vs_feature(
    feature: &[f64],
    residuals: &[f64],
    title: &str,
    x_title: &str,
    y_title: &str,
) -> Plot {
    // Ensure equal lengths
    let len = std::cmp::min(feature.len(), residuals.len());
    let xs: Vec<f64> = feature[..len].to_vec();
    let ys: Vec<f64> = residuals[..len].to_vec();

    // Compute binned medians
    let bins = 20usize.min(len.max(1));
    let mut pairs: Vec<(f64, f64)> = xs.iter().cloned().zip(ys.iter().cloned()).collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let mut bin_x = vec![];
    let mut bin_med = vec![];
    if bins > 0 {
        let bin_size = (pairs.len() as f64 / bins as f64).ceil() as usize;
        for i in 0..bins {
            let start = i * bin_size;
            let end = ((i + 1) * bin_size).min(pairs.len());
            if start >= end {
                continue;
            }
            let slice = &pairs[start..end];
            let xs_slice: Vec<f64> = slice.iter().map(|p| p.0).collect();
            let ys_slice: Vec<f64> = slice.iter().map(|p| p.1).collect();
            let median_x = xs_slice[xs_slice.len() / 2];
            let mut ys_sorted = ys_slice.clone();
            ys_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let median_y = ys_sorted[ys_sorted.len() / 2];
            bin_x.push(median_x);
            bin_med.push(median_y);
        }
    }

    let mut plot = Plot::new();
    plot.add_trace(
        Scatter::new(xs.clone(), ys.clone())
            .mode(Mode::Markers)
            .name("Residuals")
            .marker(plotly::common::Marker::new().size(6)),
    );

    if !bin_x.is_empty() {
        plot.add_trace(
            Scatter::new(bin_x, bin_med)
                .mode(Mode::Lines)
                .name("Binned median")
                .line(plotly::common::Line::new().color("rgba(255,0,0,1.0)")),
        );
    }

    plot.set_layout(
        Layout::new()
            .title(title)
            .x_axis(plotly::layout::Axis::new().title(x_title))
            .y_axis(plotly::layout::Axis::new().title(y_title)),
    );

    plot
}
