use anyhow::Result;
use clap::{Arg, ArgAction, ArgMatches, Command, ValueHint};
use log::LevelFilter;
use std::path::PathBuf;
use std::str::FromStr;

use redeem_cli::classifiers::score::score::{
    load_score_config, score_pin, write_score_output, write_score_report, ScoreConfig,
};
use redeem_classifiers::config::ModelType;
use redeem_classifiers::data_handling::RankGrouping;
use redeem_cli::properties::inference::inference;
use redeem_cli::properties::inference::input::PropertyInferenceConfig;
use redeem_cli::properties::train::input::PropertyTrainConfig;
use redeem_cli::properties::train::trainer;
use redeem_cli::topaz::infer as topaz_infer;
use redeem_cli::topaz::infer::input::TopazInferConfig;
use redeem_cli::topaz::train as topaz_train;
use redeem_cli::topaz::train::input::TopazTrainConfig;
use redeem_cli::topaz::xrun as topaz_xrun;
use redeem_cli::topaz::xrun::input::TopazXrunSweepConfig;

fn main() -> Result<()> {
    env_logger::Builder::default()
        .filter_level(LevelFilter::Error)
        .parse_env(env_logger::Env::default().filter_or("REDEEM_LOG", "error,redeem=info"))
        .init();

    let matches = Command::new("redeem")
        .version(clap::crate_version!())
        .author("Justin Sing <justincsing@gmail.com>")
        .about("\u{1F9EA} ReDeeM CLI - Modular Deep Learning Tools for Proteomics")
        .subcommand_required(true)
        .arg_required_else_help(true)
        .subcommand(
            Command::new("properties")
                .about("Train or run peptide property prediction models")
                .subcommand(
                    Command::new("train")
                        .about("Train a new property prediction model from scratch")
                        .arg(
                            Arg::new("config")
                                .help("Path to training configuration file (omit to print a template)")
                                .required(false)
                                .value_parser(clap::value_parser!(PathBuf))
                                .value_hint(ValueHint::FilePath),
                        )
                        .arg(
                            Arg::new("train_data")
                                .short('d')
                                .long("train_data")
                                .value_parser(clap::builder::NonEmptyStringValueParser::new())
                                .help(
                                    "Path to training data. Overrides the training data file \
                                     specified in the configuration file.",
                                )
                                .value_hint(ValueHint::FilePath),
                        )
                        .arg(
                            Arg::new("validation_data")
                                .short('v')
                                .long("validation_data")
                                .value_parser(clap::builder::NonEmptyStringValueParser::new())
                                .help(
                                    "Path to validation data. Overrides the validation data file \
                                     specified in the configuration file.",
                                )
                                .value_hint(ValueHint::FilePath),
                        )
                        .arg(
                            Arg::new("output_file")
                                .short('o')
                                .long("output_file")
                                .value_parser(clap::builder::NonEmptyStringValueParser::new())
                                .help(
                                    "File path that the safetensors trained model will be written to. \
                                     Overrides the directory specified in the configuration file.",
                                )
                                .value_hint(ValueHint::FilePath),
                        )
                        .arg(
                            Arg::new("model_arch")
                                .short('m')
                                .long("model_arch")
                                .help(
                                    "Model architecture to train. \
                                     Overrides the model architecture specified in the configuration file.",
                                )
                                .value_parser([
                                    "rt_cnn_lstm",
                                    "rt_cnn_tf",
                                    "ms2_bert",
                                    "ccs_cnn_lstm",
                                ])
                                .required(false)
                        )
                        .arg(
                            Arg::new("checkpoint_file")
                                .short('c')
                                .long("checkpoint_file")
                                .value_parser(clap::builder::NonEmptyStringValueParser::new())
                                .help(
                                    "File path of the checkpoint safetensors file to load. \
                                     Overrides the checkpoint_file specified in the configuration file.",
                                )
                                .value_hint(ValueHint::FilePath),
                        ),
                )
                .subcommand(Command::new("inference")
                    .about("Perform inference on new data using a trained model")
                    .arg(
                        Arg::new("pretrained")
                            .long("pretrained")
                            .help("Name of a known pretrained model to use instead of passing --model. Examples: 'redeem-ccs', 'redeem-rt', 'alphapeptdeep-ccs'")
                            .value_parser(clap::builder::NonEmptyStringValueParser::new())
                            .value_hint(ValueHint::Other),
                    )
                    .arg(
                        Arg::new("config")
                            .help("Path to inference configuration file (omit to print a template)")
                            .required(false)
                            .value_parser(clap::value_parser!(PathBuf))
                            .value_hint(ValueHint::FilePath),
                    )
                    .arg(
                        Arg::new("model_path")
                            .short('m')
                            .long("model")
                            .help("Path to the trained model file (*.safetensors)")
                            .value_parser(clap::value_parser!(PathBuf))
                            .value_hint(ValueHint::FilePath),
                    )
                    .arg(
                        Arg::new("inference_data")
                            .short('d')
                            .long("inference_data")
                            .help("Path to the input data file")
                            .value_parser(clap::value_parser!(PathBuf))
                            .value_hint(ValueHint::FilePath),
                    )
                    .arg(
                        Arg::new("output_file")
                            .short('o')
                            .long("output_file")
                            .help("Path to the output file for predictions (*.tsv or *.csv)")
                            .value_parser(clap::value_parser!(PathBuf))
                            .value_hint(ValueHint::FilePath),
                    )
                ),
        )
        .subcommand(
            Command::new("classifiers")
                .about("Run classification tools such as rescoring")
                .subcommand(
                    Command::new("rescore")
                        .about("Run rescoring tool with specified configuration")
                        .arg(
                            Arg::new("config")
                                .help("Path to classifier configuration file (omit to print a template)")
                                .required(false)
                                .value_parser(clap::value_parser!(PathBuf))
                                .value_hint(ValueHint::FilePath),
                        ),
                )
                .subcommand(
                    Command::new("score")
                        .about("Score a Percolator .pin file with the semi-supervised classifier")
                        .arg(
                            Arg::new("pin")
                                .help("Path to the Percolator .pin input file")
                                .required(true)
                                .value_parser(clap::value_parser!(PathBuf))
                                .value_hint(ValueHint::FilePath),
                        )
                        .arg(
                            Arg::new("output_file")
                                .short('o')
                                .long("output")
                                .help("Path to write the scored PIN output (TSV). Defaults to stdout.")
                                .value_parser(clap::value_parser!(PathBuf))
                                .value_hint(ValueHint::FilePath),
                        )
                        .arg(
                            Arg::new("rank_grouping")
                                .long("rank-grouping")
                                .help("Grouping strategy for rank inference/deduplication.")
                                .value_parser(["percolator", "spec-id"])
                                .value_hint(ValueHint::Other),
                        )
                        .arg(
                            Arg::new("model_type")
                                .long("model-type")
                                .help("Override the model type from the JSON config.")
                                .value_parser(["gbdt", "xgboost", "svm"])
                                .value_hint(ValueHint::Other),
                        )
                        .arg(
                            Arg::new("no_report")
                                .long("no-report")
                                .help("Disable HTML report generation.")
                                .action(ArgAction::SetTrue),
                        )
                        .arg(
                            Arg::new("deduplicate")
                                .long("dedup")
                                .help("Deduplicate PSMs in the final output by grouping.")
                                .action(ArgAction::SetTrue),
                        )
                        .arg(
                            Arg::new("config")
                                .help("Path to classifier JSON configuration file")
                                .required(false)
                                .value_parser(clap::value_parser!(PathBuf))
                                .value_hint(ValueHint::FilePath),
                        ),
                )
        )
        .subcommand(
            Command::new("topaz")
                .about("Train or run TOPAZ trace-first DIA scorer")
                .subcommand(
                    Command::new("train")
                        .about("Train TOPAZ from OSW + XIC")
                        .arg(
                            Arg::new("config")
                                .help("Path to training configuration file (omit to print a template)")
                                .required(false)
                                .value_parser(clap::value_parser!(PathBuf))
                                .value_hint(ValueHint::FilePath),
                        )
                        .arg(
                            Arg::new("osw_path")
                                .long("osw")
                                .value_parser(clap::value_parser!(PathBuf))
                                .value_hint(ValueHint::FilePath),
                        )
                        .arg(
                            Arg::new("xic_path")
                                .long("xic")
                                .value_parser(clap::value_parser!(PathBuf))
                                .value_hint(ValueHint::FilePath),
                        )
                        .arg(
                            Arg::new("output_prefix")
                                .long("output")
                                .value_parser(clap::value_parser!(PathBuf))
                                .value_hint(ValueHint::FilePath),
                        )
                        .arg(
                            Arg::new("device")
                                .long("device")
                                .value_parser(clap::builder::NonEmptyStringValueParser::new()),
                        )
                        .arg(
                            Arg::new("batch_size")
                                .long("batch-size")
                                .value_parser(clap::value_parser!(usize)),
                        )
                        .arg(
                            Arg::new("epochs")
                                .long("epochs")
                                .value_parser(clap::value_parser!(usize)),
                        )
                        .arg(
                            Arg::new("bag_k")
                                .long("bag-k")
                                .value_parser(clap::value_parser!(usize)),
                        )
                        .arg(
                            Arg::new("val_frac")
                                .long("val-frac")
                                .value_parser(clap::value_parser!(f32)),
                        )
                        .arg(
                            Arg::new("seed")
                                .long("seed")
                                .value_parser(clap::value_parser!(u64)),
                        )
                        .arg(
                            Arg::new("restrict_xic")
                                .long("restrict-osw-to-xic-map")
                                .help("Drop rows with no XIC traces")
                                .action(ArgAction::SetTrue),
                        ),
                )
                .subcommand(
                    Command::new("infer")
                        .about("Run TOPAZ inference on OSW + XIC")
                        .arg(
                            Arg::new("config")
                                .help("Path to inference configuration file (omit to print a template)")
                                .required(false)
                                .value_parser(clap::value_parser!(PathBuf))
                                .value_hint(ValueHint::FilePath),
                        )
                        .arg(
                            Arg::new("osw_path")
                                .long("osw")
                                .value_parser(clap::value_parser!(PathBuf))
                                .value_hint(ValueHint::FilePath),
                        )
                        .arg(
                            Arg::new("xic_path")
                                .long("xic")
                                .value_parser(clap::value_parser!(PathBuf))
                                .value_hint(ValueHint::FilePath),
                        )
                        .arg(
                            Arg::new("checkpoint")
                                .long("checkpoint")
                                .value_parser(clap::value_parser!(PathBuf))
                                .value_hint(ValueHint::FilePath),
                        )
                        .arg(
                            Arg::new("output_tsv")
                                .long("output-tsv")
                                .value_parser(clap::value_parser!(PathBuf))
                                .value_hint(ValueHint::FilePath),
                        )
                        .arg(
                            Arg::new("output_osw")
                                .long("output-osw")
                                .value_parser(clap::value_parser!(PathBuf))
                                .value_hint(ValueHint::FilePath),
                        )
                        .arg(
                            Arg::new("device")
                                .long("device")
                                .value_parser(clap::builder::NonEmptyStringValueParser::new()),
                        )
                        .arg(
                            Arg::new("batch_size")
                                .long("batch-size")
                                .value_parser(clap::value_parser!(usize)),
                        )
                        .arg(
                            Arg::new("pep_bins")
                                .long("pep-bins")
                                .value_parser(clap::value_parser!(usize)),
                        )
                        .arg(
                            Arg::new("restrict_xic")
                                .long("restrict-osw-to-xic-map")
                                .help("Drop rows with no XIC traces")
                                .action(ArgAction::SetTrue),
                        ),
                )
                .subcommand(
                    Command::new("xrun-sweep")
                        .about("Run XRUN calibrator sweep")
                        .arg(
                            Arg::new("config")
                                .help("Path to XRUN sweep configuration file (omit to print a template)")
                                .required(false)
                                .value_parser(clap::value_parser!(PathBuf))
                                .value_hint(ValueHint::FilePath),
                        )
                        .arg(
                            Arg::new("osw_path")
                                .long("osw")
                                .value_parser(clap::value_parser!(PathBuf))
                                .value_hint(ValueHint::FilePath),
                        )
                        .arg(
                            Arg::new("xic_path")
                                .long("xic")
                                .value_parser(clap::value_parser!(PathBuf))
                                .value_hint(ValueHint::FilePath),
                        )
                        .arg(
                            Arg::new("checkpoint")
                                .long("checkpoint")
                                .value_parser(clap::value_parser!(PathBuf))
                                .value_hint(ValueHint::FilePath),
                        )
                        .arg(
                            Arg::new("output_tsv")
                                .long("output-tsv")
                                .value_parser(clap::value_parser!(PathBuf))
                                .value_hint(ValueHint::FilePath),
                        )
                        .arg(
                            Arg::new("device")
                                .long("device")
                                .value_parser(clap::builder::NonEmptyStringValueParser::new()),
                        )
                        .arg(
                            Arg::new("restrict_xic")
                                .long("restrict-osw-to-xic-map")
                                .help("Drop rows with no XIC traces")
                                .action(ArgAction::SetTrue),
                        ),
                ),
        )
        .help_template(
            "{usage-heading} {usage}\n\n\
             {about-with-newline}\n\
             Written by {author-with-newline}Version {version}\n\n\
             {all-args}{after-help}",
        )
        .get_matches();

    match matches.subcommand() {
        Some(("properties", sub_m)) => handle_properties(sub_m),
        Some(("classifiers", sub_m)) => handle_classifiers(sub_m),
        Some(("topaz", sub_m)) => handle_topaz(sub_m),
        _ => unreachable!("Subcommand is required by CLI configuration"),
    }
}

fn handle_properties(matches: &ArgMatches) -> Result<()> {
    match matches.subcommand() {
        Some(("train", train_matches)) => {
            let config_path: Option<&PathBuf> = train_matches.get_one("config");

            if config_path.is_none() {
                let default = PropertyTrainConfig::default();
                let json = serde_json::to_string_pretty(&default)
                    .expect("failed to serialize default config");
                eprintln!(
                    "\n\u{2139}\u{fe0f}  No config file provided.\n\n\
                     Save the following JSON template to a file (e.g. train_config.json),\n\
                     fill in the fields, and re-run:\n\n\
                       redeem properties train train_config.json\n"
                );
                println!("{}", json);
                std::process::exit(0);
            }

            let config_path = config_path.unwrap();
            log::info!(
                "[ReDeeM::Properties] Training from config: {:?}",
                config_path
            );

            let params: PropertyTrainConfig =
                PropertyTrainConfig::from_arguments(config_path, train_matches)?;

            match trainer::run_training(&params) {
                Ok(_) => Ok(()),
                Err(e) => {
                    log::error!("Training failed: {:#}", e);
                    std::process::exit(1)
                }
            }
        }
        Some(("inference", inference_matches)) => {
            let config_path: Option<&PathBuf> = inference_matches.get_one("config");

            if config_path.is_none() {
                let default = PropertyInferenceConfig::default();
                let json = serde_json::to_string_pretty(&default)
                    .expect("failed to serialize default config");
                eprintln!(
                    "\n\u{2139}\u{fe0f}  No config file provided.\n\n\
                     Save the following JSON template to a file (e.g. inference_config.json),\n\
                     fill in the fields, and re-run:\n\n\
                       redeem properties inference inference_config.json\n"
                );
                println!("{}", json);
                std::process::exit(0);
            }

            let config_path = config_path.unwrap();
            log::info!(
                "[ReDeeM::Properties] Inference using config: {:?}",
                config_path
            );

            let params: PropertyInferenceConfig =
                PropertyInferenceConfig::from_arguments(config_path, inference_matches)?;

            match inference::run_inference(&params) {
                Ok(_) => Ok(()),
                Err(e) => {
                    log::error!("Inference failed: {:#}", e);
                    std::process::exit(1)
                }
            }
        }
        _ => unreachable!(),
    }
}

fn handle_classifiers(matches: &ArgMatches) -> Result<()> {
    match matches.subcommand() {
        Some(("rescore", rescore_matches)) => {
            let config_path: Option<&PathBuf> = rescore_matches.get_one("config");

            if config_path.is_none() {
                let default = ScoreConfig::default();
                let json = serde_json::to_string_pretty(&default)
                    .expect("failed to serialize default config");
                eprintln!(
                    "\n\u{2139}\u{fe0f}  No config file provided.\n\n\
                     Save the following JSON template to a file (e.g. rescore_config.json),\n\
                     fill in the fields, and re-run:\n\n\
                       redeem classifiers rescore rescore_config.json\n"
                );
                println!("{}", json);
                std::process::exit(0);
            }

            let config_path = config_path.unwrap();
            println!(
                "[ReDeeM::Classifiers] Rescoring using config: {:?}",
                config_path
            );
            // Call your classifier logic using config_path
            Ok(())
        }
        Some(("score", score_matches)) => {
            let pin_path: &PathBuf = score_matches.get_one("pin").unwrap();
            let output_path: Option<&PathBuf> = score_matches.get_one("output_file");
            eprintln!("[ReDeeM::Classifiers] Scoring PIN file: {:?}", pin_path);

            let mut config = if let Some(config_path) = score_matches.get_one::<PathBuf>("config") {
                eprintln!("[ReDeeM::Classifiers] Using config: {:?}", config_path);
                load_score_config(config_path)?
            } else {
                let default_config = ScoreConfig::default();
                eprintln!("[ReDeeM::Classifiers] No config provided; using defaults.");
                default_config
            };

            if let Some(grouping) = score_matches.get_one::<String>("rank_grouping") {
                config.rank_grouping = match grouping.as_str() {
                    "percolator" => RankGrouping::Percolator,
                    "spec-id" => RankGrouping::SpecId,
                    _ => config.rank_grouping,
                };
            }

            if let Some(model_type) = score_matches.get_one::<String>("model_type") {
                config.model.model_type =
                    ModelType::from_str(model_type).map_err(anyhow::Error::msg)?;
            }

            if score_matches.get_flag("deduplicate") {
                config.deduplicate = true;
            }

            if score_matches.get_one::<PathBuf>("config").is_none() {
                let default_json = serde_json::to_string_pretty(&config).unwrap_or_default();
                eprintln!(
                    "[ReDeeM::Classifiers] Default config:\n{}",
                    default_json
                );
            }

            let result = score_pin(pin_path, &config)?;
            write_score_output(pin_path, &result, output_path)?;
            if !score_matches.get_flag("no_report") {
                let report_name = format!(
                    "redeem_score_{}.html",
                    model_type_name(&config.model.model_type)
                );
                write_score_report(&result, &PathBuf::from(report_name))?;
            }
            eprintln!(
                "[ReDeeM::Classifiers] Completed scoring {} PSMs.",
                result.predictions.as_slice().len()
            );
            Ok(())
        }
        _ => unreachable!(),
    }
}

fn handle_topaz(matches: &ArgMatches) -> Result<()> {
    match matches.subcommand() {
        Some(("train", sub)) => {
            let config_path: Option<&PathBuf> = sub.get_one("config");
            if config_path.is_none() {
                let default = TopazTrainConfig::default();
                let json = serde_json::to_string_pretty(&default)?;
                eprintln!(
                    "\n\u{2139}\u{fe0f}  No config file provided.\n\n\
                     Save the following JSON template to a file (e.g. topaz_train.json),\n\
                     fill in the fields, and re-run:\n\n\
                       redeem topaz train topaz_train.json\n"
                );
                println!("{}", json);
                std::process::exit(0);
            }

            let cfg = TopazTrainConfig::from_arguments(config_path.unwrap(), sub)?;
            topaz_train::run(&cfg)
        }
        Some(("infer", sub)) => {
            let config_path: Option<&PathBuf> = sub.get_one("config");
            if config_path.is_none() {
                let default = TopazInferConfig::default();
                let json = serde_json::to_string_pretty(&default)?;
                eprintln!(
                    "\n\u{2139}\u{fe0f}  No config file provided.\n\n\
                     Save the following JSON template to a file (e.g. topaz_infer.json),\n\
                     fill in the fields, and re-run:\n\n\
                       redeem topaz infer topaz_infer.json\n"
                );
                println!("{}", json);
                std::process::exit(0);
            }
            let cfg = TopazInferConfig::from_arguments(config_path.unwrap(), sub)?;
            topaz_infer::run(&cfg)
        }
        Some(("xrun-sweep", sub)) => {
            let config_path: Option<&PathBuf> = sub.get_one("config");
            if config_path.is_none() {
                let default = TopazXrunSweepConfig::default();
                let json = serde_json::to_string_pretty(&default)?;
                eprintln!(
                    "\n\u{2139}\u{fe0f}  No config file provided.\n\n\
                     Save the following JSON template to a file (e.g. xrun_sweep.json),\n\
                     fill in the fields, and re-run:\n\n\
                       redeem topaz xrun-sweep xrun_sweep.json\n"
                );
                println!("{}", json);
                std::process::exit(0);
            }
            let cfg = TopazXrunSweepConfig::from_arguments(config_path.unwrap(), sub)?;
            topaz_xrun::run(&cfg)
        }
        _ => unreachable!(),
    }
}

fn model_type_name(model_type: &ModelType) -> &'static str {
    match model_type {
        ModelType::GBDT { .. } => "gbdt",
        #[cfg(feature = "xgboost")]
        ModelType::XGBoost { .. } => "xgboost",
        #[cfg(feature = "svm")]
        ModelType::SVM { .. } => "svm",
    }
}
