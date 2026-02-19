//! Plotting helpers: histograms, P–P plots, boxplots and scatter plots.
//!
//! These functions produce `plotly::Plot` objects from score and label
//! arrays. They are minimal helpers used by the example report generation.
use itertools_num::linspace;
use plotly::box_plot::BoxMean;
use plotly::common::{DashType, Line, Marker, Mode, Orientation};
use plotly::layout::{Axis, Layout, Legend};
use plotly::{BoxPlot, Histogram, Plot, Scatter};

/// Plot a histogram of the scores for the targets and decoys
pub fn plot_score_histogram(
    scores: &[f64],
    labels: &[i32],
    title: &str,
    x_title: &str,
) -> Result<Plot, String> {
    assert_eq!(
        scores.len(),
        labels.len(),
        "Scores and labels must have the same length"
    );
    assert!(
        labels.iter().all(|&l| l == 1 || l == -1),
        "Labels must be 1 for targets and -1 for decoys"
    );

    let mut scores_target = Vec::new();
    let mut scores_decoy = Vec::new();

    for (score, label) in scores.iter().zip(labels.iter()) {
        if *label == 1 {
            scores_target.push(*score);
        } else {
            scores_decoy.push(*score);
        }
    }

    let trace_target = Histogram::new(scores_target).name("Target");
    let trace_decoy = Histogram::new(scores_decoy).name("Decoy");

    let layout = Layout::new()
        .title(title)
        .x_axis(plotly::layout::Axis::new().title(x_title))
        .y_axis(plotly::layout::Axis::new().title("Density"));

    let mut plot = Plot::new();
    plot.add_trace(trace_target);
    plot.add_trace(trace_decoy);
    plot.set_layout(layout);

    Ok(plot)
}

fn ecdf(data: &mut [f64]) -> (Vec<f64>, Vec<f64>) {
    data.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = data.len() as f64;
    let y: Vec<f64> = (1..=data.len()).map(|i| i as f64 / n).collect();
    (data.to_vec(), y)
}

fn interpolate_ecdf(x: &[f64], y: &[f64], x_seq: &[f64]) -> Vec<f64> {
    x_seq
        .iter()
        .map(|&xi| {
            let idx = x.iter().position(|&xv| xv >= xi).unwrap_or(x.len() - 1);
            y[idx]
        })
        .collect()
}

// fn estimate_pi0(decoy_scores: &Vec<f64>, lambda: f64) -> f64 {
//     let n = decoy_scores.len() as f64;
//     let count_above_lambda = decoy_scores.iter().filter(|&&s| s > lambda).count() as f64;
//     count_above_lambda / ((1.0 - lambda) * n)
// }

/// Estimate the proportion of null hypotheses (π₀).
fn estimate_pi0(labels: &[i32]) -> f64 {
    let count_decoys = labels.iter().filter(|&&l| l == -1).count() as f64;
    let count_targets = labels.iter().filter(|&&l| l == 1).count() as f64;
    count_decoys / count_targets
}

/// Generate a P-P plot as described in Debrie, E. et. al. (2023) Journal of Proteome Research.
///
/// # Arguments
///
/// * `top_targets` - The top scores for the targets
/// * `top_decoys` - The top scores for the decoys
/// * `title` - The title of the plot
///
/// # Returns
///
/// A Plot object containing the P-P plot
pub fn plot_pp(scores: &[f64], labels: &[i32], title: &str) -> Result<Plot, String> {
    assert_eq!(
        scores.len(),
        labels.len(),
        "Scores and labels must have the same length"
    );
    assert!(
        labels.iter().all(|&l| l == 1 || l == -1),
        "Labels must be 1 for targets and -1 for decoys"
    );

    let mut scores_target = Vec::new();
    let mut scores_decoy = Vec::new();

    for (score, label) in scores.iter().zip(labels.iter()) {
        if *label == 1 {
            scores_target.push(*score);
        } else {
            scores_decoy.push(*score);
        }
    }

    let (x_target, y_target) = ecdf(&mut scores_target);
    let (x_decoy, y_decoy) = ecdf(&mut scores_decoy);

    let x_min = x_target.first().unwrap().min(*x_decoy.first().unwrap());
    let x_max = x_target.last().unwrap().max(*x_decoy.last().unwrap());
    let x_seq: Vec<f64> = linspace(x_min, x_max, 1000).collect();

    let y_target_interp = interpolate_ecdf(&x_target, &y_target, &x_seq);
    let y_decoy_interp = interpolate_ecdf(&x_decoy, &y_decoy, &x_seq);

    // let pi0 = estimate_pi0(&scores_decoy, 0.5);
    let pi0 = estimate_pi0(labels);
    let pi0_line_y: Vec<f64> = y_decoy_interp.iter().map(|&x| pi0 * x).collect();

    let mut plot = Plot::new();

    let scatter = Scatter::new(y_decoy_interp.clone(), y_target_interp.clone())
        .mode(Mode::Markers)
        .name("Target vs Decoy ECDF");

    let reference_line = Scatter::new(vec![0.0, 1.0], vec![0.0, 1.0])
        .mode(Mode::Lines)
        .name("y = x (Perfect match)")
        .line(Line::new().color("red").dash(DashType::Dash));

    let pi0_line = Scatter::new(y_decoy_interp.clone(), pi0_line_y)
        .mode(Mode::Lines)
        .name(format!("Estimated π₀ = {:.3}", pi0))
        .line(Line::new().color("blue").dash(DashType::Dot));

    plot.add_trace(scatter);
    plot.add_trace(reference_line);
    plot.add_trace(pi0_line);
    plot.set_layout(
        Layout::new()
            .title(title)
            .x_axis(plotly::layout::Axis::new().title("Decoy ECDF"))
            .y_axis(plotly::layout::Axis::new().title("Target ECDF")),
    );

    Ok(plot)
}

/// Generate a box plot of the scores/intensities for each file
///
/// # Arguments
///
/// * `scores` - A vector of vectors where each inner vector contains the scores/intensities for a file
/// * `filenames` - A vector of filenames corresponding to the scores
/// * `title` - The title of the plot
/// * `x_title` - The title of the x-axis
/// * `y_title` - The title of the y-axis
///
/// # Returns
///
/// A Plot object containing the box plot
pub fn plot_boxplot(
    scores: &[Vec<f64>],
    filenames: &[String],
    title: &str,
    x_title: &str,
    y_title: &str,
) -> Result<Plot, String> {
    assert_eq!(
        scores.len(),
        filenames.len(),
        "Scores and filenames must have the same length"
    );

    let mut plot = Plot::new();
    for (i, s) in scores.iter().enumerate() {
        let trace = BoxPlot::new_xy(vec![filenames[i].clone(); s.len()], s.to_vec())
            .name(filenames[i].clone())
            .box_mean(BoxMean::True);
        plot.add_trace(trace);
    }

    let layout = Layout::new()
        .title(title)
        .x_axis(Axis::new().title(x_title).tick_angle(45.0))
        .y_axis(Axis::new().title(y_title))
        .show_legend(false);

    plot.set_layout(layout);

    Ok(plot)
}

pub fn plot_scatter(
    x: &[Vec<f64>],
    y: &[Vec<f64>],
    labels: &[String],
    title: &str,
    x_title: &str,
    y_title: &str,
) -> Result<Plot, String> {
    assert_eq!(x.len(), y.len(), "X and Y must have the same length");

    // Check to see how large the data is, if there's a large amount of data we should use web_gl_mode. We can look at one of the arrays to see how many points there are
    let _web_gl_mode = x[0].len() > 10_000;

    let mut plot = Plot::new();
    for (i, (x_i, y_i)) in x.iter().zip(y.iter()).enumerate() {
        let trace = Scatter::new(x_i.to_vec(), y_i.to_vec())
            .name(labels[i].clone())
            .mode(Mode::Markers)
            .marker(Marker::new().size(10))
            .web_gl_mode(true);
        plot.add_trace(trace);
    }

    let layout = Layout::new()
        .title(title)
        .x_axis(Axis::new().title(x_title))
        .y_axis(Axis::new().title(y_title))
        .legend(Legend::new().orientation(Orientation::Vertical));

    plot.set_layout(layout);

    Ok(plot)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plot_boxplot() {
        let scores = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![6.0, 7.0, 8.0, 9.0, 10.0],
            vec![11.0, 12.0, 13.0, 14.0, 15.0],
        ];
        let filenames = vec![
            "file1".to_string(),
            "file2".to_string(),
            "file3".to_string(),
        ];
        let title = "Box Plot";
        let x_title = "Filenames";
        let y_title = "Scores";

        let plot = plot_boxplot(&scores, &filenames, title, x_title, y_title).unwrap();

        plot.write_html("test_plot_boxplot.html");

        // assert_eq!(plot.
        // assert_eq!(plot.layout.title, Some(title.to_string()));
        // assert_eq!(plot.layout.x_axis, Some(Axis::new().title(x_title).tick_angle(45.0)));
        // assert_eq!(plot.layout.y_axis, Some(Axis::new().title(y_title)));
    }

    #[test]
    #[should_panic(expected = "Scores and filenames must have the same length")]
    fn test_plot_boxplot_mismatched_lengths() {
        let scores = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![6.0, 7.0, 8.0, 9.0, 10.0],
        ];
        let filenames = vec![
            "file1".to_string(),
            "file2".to_string(),
            "file3".to_string(),
        ];
        let title = "Box Plot";
        let x_title = "Filenames";
        let y_title = "Scores";

        plot_boxplot(&scores, &filenames, title, x_title, y_title).unwrap();
    }

    #[test]
    fn test_plot_scatter() {
        let x = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![2.0, 7.0, 3.0, 9.0, 10.0],
            vec![1.0, 12.0, 13.0, 14.0, 15.0],
        ];
        let y = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![6.0, 7.0, 8.0, 9.0, 10.0],
            vec![11.0, 12.0, 13.0, 14.0, 15.0],
        ];
        let labels = vec![
            "file1".to_string(),
            "file2".to_string(),
            "file3".to_string(),
        ];
        let title = "Scatter Plot";
        let x_title = "X";
        let y_title = "Y";

        let plot = plot_scatter(&x, &y, &labels, title, x_title, y_title).unwrap();

        plot.write_html("test_plot_scatter.html");
    }
}
