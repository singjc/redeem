# Changelog

All notable changes to this project will be documented in this file.

## [0.1.1] - 2026-03-16

### 🚀 Features

- Add version detection for redeem_properties package
- Refactor tests to use locate_pretrained_model for model paths
- Implement configured pretrained-models directory resolution

### 💼 Other

- Add support for x86_64-unknown-linux-musl target in release workflow

### 🚜 Refactor

- Update pretrained model directory handling for cross-platform compatibility

### ⚙️ Miscellaneous Tasks

- Enhance CHANGELOG update process with branch detection and push handling
- Update CHANGELOG.md

## [0.1.0] - 2026-02-22

### 🚀 Features

- Make function to download pretrained models public and update function name
- Refactor bidirectional LSTM implementation for improved performance with batch sizing and readability
- Fix CUDA error in LSTM weights by temporarily using CPU device for forward pass
- Add examples for loading and fine-tuning alphapeptdeep models
- Add GBDT Classifier to Redeem project
- Add pi0 estimation to P-P plot in redeem-classifiers crate
- Update ReDeeM README with usage instructions and note about crates availability
- Update plot functions in redeem-classifiers crate
- Add x-axis title to score histogram plot in redeem-classifiers crate
- Update pi0 estimation in P-P plot for redeem-classifiers crate
- Add DataTables and FileSaver.js for interactivity and downloading in report module
- Update DataTables and FileSaver.js dependencies for interactivity and downloading in report module
- Add JavaScript for tabs in report module
- Add box plot function to redeem-classifiers crate
- Update plot_score_histogram function in redeem-classifiers crate
- Add legend orientation to scatter plot in redeem-classifiers crate
- Improve table container
- Add colResize feature to DataTables in report module
- Update DataTables colResize CSS link in report module
- Add web_gl_mode for large data in scatter plot
- Update legend orientation in scatter plot to vertical
- Generate random plot IDs for each plot in the report
- Add installation instructions for XGBoost library in README.md
- Fix typo in log message for Input26aaModPositionalEncoding::forward method
- Add new modules for training and loading data in redeem-cli
- Add Dockerfile for CUDA-based application containerization
- Add inference functionality to redeem-cli
- Implement custom error handling for TDC calculations
- Add error handling for Experiment initialization and validation
- Add redeem-properties-py PyO3 bindings package
- Add from_pretrained constructor to RTModel, CCSModel, MS2Model
- Accept modified peptides directly in predict; auto-parse mods
- Return fragment annotations alongside MS2 intensities in predict
- Annotate CCS predict output with charge state per peptide
- Add predict_df methods returning pandas/polars DataFrames
- Add pretrained feature for redeem-properties integration
- Add dynamic layer construction for DecoderLinear from VarMap
- Infer hyperparameters dynamically for CCS and RT models' decoders
- Add hyperparameter inference for CNN-Transformer models
- Update model constructors to accept optional constants path
- Make constants_path optional in CCSModel and MS2Model constructors
- Add functions to locate and validate pretrained models in Python
- Add optional output denormalization for model predictions
- Add CCSCNN model prediction example with detailed output
- Enhance model name mapping in from_str for pretrained models
- Add model architecture retrieval and detailed summaries for CCS, MS2, and RT models
- Add parameter count and summary methods for RT, CCS, and MS2 models
- Add multiplier and exclude_zeros parameters to MS2Model predictions
- Add PropertyPrediction class for unified peptide property predictions
- Add rustyms dependency and mz_utils module to enhance functionality
- Add m/z computation functions and enhance MS2Model and PropertyPrediction for m/z annotation
- Add unified prediction functionality with PropertyPrediction for combined RT, CCS, and MS2 outputs
- Enhance CCS and MS2 models with input expansion and ion mobility annotation
- Enhance MS2Model with precursor m/z annotation and update Rust model wrappers
- Add comprehensive tests for RT, CCS, and MS2 models with DataFrame support and annotations
- Implement utility functions for computing precursor and product m/z values from ProForma notation
- Update Python wheel and source distribution workflows to include pretrained features
- Add Sphinx documentation structure with Makefile, requirements, and initial content
- Add custom CSS and logo for Sphinx documentation theme
- Add new logo images for dark theme in documentation
- Add README.md for Sphinx documentation setup and usage instructions
- Add GitHub Actions workflow for building and uploading Sphinx documentation
- Add Read the Docs configuration for building documentation
- Implement Array2 and Array1 structures with associated methods for matrix and vector operations
- Enhance SVM example to accept CSV path and column count from CLI arguments
- Add reporting functionality to SVM and XGBoost examples, including score distribution plots
- Add preprocessing utilities and integrate scaling and normalization options in SemiSupervisedLearner
- Update dependencies and improve prediction output formatting in examples and models
- Add report-builder dependency and re-export Report and ReportSection
- Add pretrained model support and enhance CLI options for inference
- Implement Percolator .pin TSV reader and related utilities
- Add scoring functionality for Percolator .pin files with semi-supervised classifier
- Introduce RankGrouping for enhanced rank inference and deduplication in scoring
- Add feature_weights method to ClassifierModel and implement in SVMClassifier for linear feature importances
- Add HTML report generation for scoring results with histograms
- Add support for SVM and XGBoost models in CLI with report generation toggle
- Add warmup_fraction parameter for learning rate scheduling in training functions
- Update logo images
- Implement pretrained model loading and caching functionality
- Enhance PyPI publishing workflow with environment and permissions
- Add BSD 3-Clause License file
- Update pyproject.toml with additional metadata and classifiers
- Update version to 0.1.0 in pyproject.toml
- Add workflow to generate changelog and release notes
- Add README.md for redeem-properties library with installation and usage instructions
- Add README.md for redeem-classifiers library with detailed features, installation instructions, and usage examples
- Add GitHub Actions workflow for releasing CLI binaries
- Add comprehensive tests for redeem_properties functionality
- Add Docker publish workflow and update Dockerfile for multi-stage build
- Update CLI to allow optional config paths and print templates when not provided
- Add download_pretrained_models function and update tests for its existence

### 🐛 Bug Fixes

- Remove linux specific and unused imports for bilstm
- Fix loss computation for rt cnn lstm model during fine tuning.
- Correct loss computation for rt cnn lstm model during fine tuning
- Put bert weights on CPU since layerNorm is not implemented for CUDA
- Squeeze instrument indices tensor
- Remove instrument_indice squeezing
- Modication name and indice retrieval
- Correct display formatting for ExperimentError single class message
- Ensure version is correctly specified in pyproject.toml
- Downgrade ndarray version to 0.15 to match numpy's dependency
- Rename redeem_properties_py function to _lib for consistency
- Update model and constants paths to use pretrained models directory
- Update CCS and MS2 prediction parameters for consistency and clarity
- Update light theme logo path in Sphinx configuration
- Correct light-svm dependency version in Cargo.toml and add clarification comment
- Update prediction conversion to f64 in GBDT example and enhance SVM scorer error handling
- Correct URL formatting for pretrained models
- Correct path formatting for pretrained models and add unit test for model download
- Update error handling for model type parsing in handle_classifiers function
- Update plotly usage for histogram layout and improve label handling in scoring functions
- Update tqdm dependency to version 0.8.0
- Update logo image sources for light and dark color schemes in README
- Update workflow to trigger on both master and develop branches
- Update Python installation instructions and import paths in README
- Remove optional embedding of pretrained model files from Cargo.toml
- Update tqdm dependency to use specific git branch
- Add error module and handle Experiment creation errors
- Remove ndarray dependency from Cargo.toml
- Update pretrained model string matching for consistency
- Update pretrained model resolution to use architecture method
- Update Read the Docs configuration for Python and Maturin installation
- Update Rust version to 1.86 in Read the Docs configuration
- Update Rust version to 1.91 in Read the Docs configuration
- Update Read the Docs configuration to omit pretrained feature during build
- Update TODO list to save target normalization min/max values in safetensor
- Clean up .gitignore by removing unnecessary comments and ensuring proper formatting
- Update README to correct terminology from "rescoring" to "scoring" for classifiers
- Update README links to point to correct README files for each crate
- Update Dockerfile to copy Cargo.lock for dependency management
- Correct project name format in pyproject.toml

### 💼 Other

- Fine tuning of rt cnn-lstm model and add tqdm logging
- Progress logger to dynamically update desc
- Cargo to use feature branch of tqdm for dynamic desc update
- Peptide data structure to hold peptide information and optional peptide property  outputs
- Data handling module
- PeptideData structure to models and add fine-tuning method for CCS
- MS2 fragment intensities to PeptideData structure and add fine-tuning for MS2 bert model
- Method to download pretrained models
- Method `get_modification_string` to retrieve modifications from modification tsv
- Update num_batches for progress logger
- TODOs
- Todos
- AtomicF64 import
- Update dependencies
- Trace messages during SVM fitting
- Readme
- Method to load tensors from pytorch pth files or safetensor files
- MesaTeE GBDT classifier to SemiSupervisedLearner
- Examples
- Scatter plot
- Data table in report
- Plot auto sizing based on window
- Properties cargo dependencies for candle
- TransformerEncoder and SeqTransformer block
- Encoder26aaModChargeCnnTransformerAttnSum implementation
- Readme
- RT Norm struct to set type of normalization
- Modification.tsv asset
- Bilstm forward with state
- Type annotation for bilstm
- Update Python bindings installation to remove unnecessary features flag
- Remove unnecessary features flag from wheel build command
- Update reqwest dependency to disable default features and enable rustls-tls
- Update macOS target versions to use 'macos-latest'

### 🚜 Refactor

- Update device handling in bilstm.rs and building_blocks.rs
- Update device handling in bilstm.rs and building_blocks.rs
- Update fine-tuning method to include batch size parameter, refactor rt model fine tuning with batch sizing
- Update fine-tuning method to include batch size parameter and improve ccs model fine-tuning with batch sizing
- Update fine-tuning method to include batch size parameter and improve ms2 model fine-tuning with batch sizing
- Move get_asciii_indices to featuize
- Remove unused import in bilstm.rs file
- Update featurize.rs to remove unused import and align with recent changes
- Refactor PredictionResult and PredictionValue types
- Add ion mobility to CCS conversion
- Fine-tuning log message
- Update fine-tuning log message for model training
- Update ModelParams and ModelType
- Add SVM classifier to redeem-classifiers crate
- Remove rayon parallel processing for fine-tuning
- Update forward pass in MS2BertModel
- Update LSTM forward pass in Encoder26aaModCnnLstmAttnSum and Encoder26aaModChargeCnnLstmAttnSum
- Add CUDA support for candle-nn and candle-transformers crates
- Update ModelParams and ModelType
- Remove commented out code in ModelType enum
- Update feature scores log message in log_input_data_summary
- Remove unused code and update cross-validation log messages
- Add SVM and XGBoost semi-supervised learning examples
- Add class percentage balance options for training in SemiSupervisedLearner
- Add log message for training data selection in SemiSupervisedLearner
- Update log message for training data selection in SemiSupervisedLearner
- Comment out remove unlabelled psms
- Update loss type handling in GBDTClassifier
- Update loss type handling in GBDTClassifier
- Comment out unused code in psm_scorer.rs
- Update log message for training data selection in SemiSupervisedLearner
- Remove redundant sentence in redeem-properties crate's README.md
- Update model structs to use 'static lifetime for VarBuilder
- Update model structs to use 'static lifetime for VarBuilder
- Update ModelClone trait to include Send and Sync bounds
- Update DLModels struct to remove unnecessary Arc and Mutex wrappers for model fields
- Update peptide modification handling to support mass shifts and UniMod annotations
- Peptide encoding
- Bilstm
- Optimize peptide sequence featurization and one-hot encoding
- Update RTCNNLSTMModel forward method to improve performance and readability
- Update redeem-properties crate models to remove unused imports and improve code organization
- Update RTCNNLSTMModel forward method to improve performance and readability
- Add RT-CNN Transformer model and update redeem-properties crate models
- Add early stopping to property training
- Remove unnecessary cargo update command in Dockerfile
- Update dependencies and descriptions in Cargo.toml files
- Update redeem-properties crate models and add new modules for training and loading data in redeem-cli
- Add new modules for training and loading data in redeem-cli
- Update Dockerfile to optimize build process and clean up artifacts
- Add CCSCNNTFModel implementation
- Update RTCNNTFModel implementation and remove unused code
- Improve regex pattern for extracting modification indices in peptdeep_utils.rs
- Update redeem-properties crate models for CCS prediction
- Add new fields to load_peptide_data function in redeem-cli
- Add stats module to redeem-properties crate, and add lr scheduler
- Add precursor mass field to PeptideData struct in redeem-properties crate
- Add plot_training_metric function to redeem-cli crate
- Update early stopping logic in ModelInterface implementation
- Update plot_losses function in redeem-cli crate
- Update config loading logic in redeem-cli crate
- Update RT-CNN-LSTM and RT-CNN-Transformer models in redeem-properties crate
- Update hidden_dim and decoder size in CCSCNNTFModel
- Clean up trace comments
- Update peptide data loading logic in redeem-cli crate
- Update PeptideData struct to use u8 for string fields
- Update mod_to_feature loading to use Arc for key in RTCNNTFModel and CCSCNNTFModel
- Improve error handling in redeem-cli crate
- Optimize contiguous operations in building_blocks.rs
- Update rank feature based on new classifier scores
- Update examples in classifiers crate
- Update rank feature and log rank changes in Experiment class
- Set log level to debug in main function
- Update loading of modifications to use byte slice instead of file path
- Update data handling to use TargetNormalization instead of RTNormalization
- Add once_cell dependency for redeem-properties crate
- Update normalization field in Redeem CLI properties
- Update loading of modifications to use byte slice instead of file path
- Update training configuration in redeem-properties crate
- Update AAEmbedding constructor signature to accept VarBuilder instead of Device
- Update semi-supervised learning to return updated  ranks along with predictions
- Update Redeem CLI to use RTCNNTFModel for inference
- Update data handling to extract "rank" feature column as 1D array of `u32`s
- Improve bidirectional LSTM input handling for contiguous tensors
- Improve bidirectional LSTM input handling for contiguous tensors
- Improve initialization of hidden states in BidirectionalLSTM
- Improve bidirectional LSTM forward and backward processing
- Clone contiguous tensor in BidirectionalLSTM for improved handling
- Improve logging in BidirectionalLSTM backward processing
- Update Dockerfile to separate build and runtime stages, improve CUDA handling
- Use pretrained module registry for from_pretrained constructors
- Enhance documentation for locate_pretrained function to clarify model path resolution
- Replace ModelParams with ModelConfig across classifiers and update related implementations
- Replace SemiSupervisedModel with ClassifierModel trait and update related implementations
- Remove unused utils module from model implementations
- Clean up imports and enhance SemiSupervisedLearner with max_iterations parameter
- Enhance SVM and XGBoost classifiers with improved parameter handling and logging for fallback scores
- Reorganize module imports for clarity and consistency across building_blocks, lib, models, and utils
- Improve code readability and structure in BidirectionalLSTM implementation
- Simplify DecoderLinear and DecoderMLP implementations with improved error handling and logging
- Remove unused projection weight field from Encoder26aaModCnnTransformerAttnSum and Encoder26aaModChargeCnnTransformerAttnSum
- Improve code organization and readability in featurize.rs with consistent formatting and sequential accumulation
- Clean up imports and improve formatting in nn.rs for better readability
- Adjust training mode default to false in RTCNNLSTMModel and update dropout application in CCSCNNLSTMModel
- Add TargetNormalization and train_var_prefixes parameters to training and fine-tuning methods in CCS, MS2, and RT model wrappers
- Enhance CCSCNNTFModel and RTCNNTFModel with flexible decoder head selection and improved constructor methods
- Simplify CNN output handling in Encoder26aaModCnnTransformerAttnSum and Encoder26aaModChargeCnnTransformerAttnSum by removing unnecessary pre-projection copy
- Clean up RTCNNTFModel constructor by removing unnecessary comments and improving clarity in varmap handling
- Clean up whitespace and improve formatting in TargetNormalization and PeptideData structures
- Reorganize imports and clean up whitespace in logging module
- Reorganize imports and improve formatting in peptdeep_utils.rs
- Enhance TrainingStepMetrics by adding MAE, RMSE, and R² calculations and updating metric summaries
- Reorganize imports and improve formatting in utils.rs
- Consolidate import statements and improve formatting in alphapeptdeep examples
- Improve encoder variable scoping and set training mode in CCSCNNTFModel
- Reorganize imports and improve logging format in main.rs
- Add candle-nn dependency to Cargo.toml
- Improve formatting and organization of code in util.rs
- Simplify header index lookup and improve normalization logic in load_peptide_data
- Clean up imports and enhance PropertyTrainConfig with additional normalization and head type fields
- Reorder module declarations and improve sample_indices function documentation
- Enhance write_peptide_data function to include original targets and improve header handling
- Reorder imports in xgboost.rs for improved organization
- Improve MS2 target tensor construction with enhanced error handling and logging
- Update prediction and target extraction to use flatten_all for improved tensor handling
- Update pretrained models URL and paths for new pretrained models added
- Move individual crates out of crate folder into main root project
- Remove test_basic.py as part of codebase cleanup
- Reorganize CI workflow for Rust and Python tests

### 📚 Documentation

- Update title in index.md for clarity and consistency
- Add README with examples for redeem-classifiers usage
- Enhance module documentation across the crate with detailed descriptions
- Update README.md to enhance project description and usage examples
- Add closing code block for Unified prediction section in README
- Update README to emphasize CLI is still under development
- Update examples to include necessary imports and correct assertions
- Add badges for Rust, PyPI, and documentation in README
- Update getting started guide to clarify pretrained model downloads

### 🧪 Testing

- Ensure pretrained models are downloaded before tests that require them
- Ensure pretrained models are loaded before running tests
- Import necessary modules for testing SemiSupervisedLearner
- Set up virtual environment for Python dependencies and tests

### ⚙️ Miscellaneous Tasks

- Update dependencies and add new crates
- Add report module and plots module to redeem-classifiers crate
- Update env_logger dependency to version 0.8.4
- Update xgboost dependency to use custom repository and branch
- Update XGBoost and Linfa dependencies for SVM classifier support as optional features
- Update xgboost dependency to use custom repository and branch that is more up-to-date
- Update model type handling for XGBoost and SVM classifiers for optional feature dependency
- Update xgboost dependency to use custom repository and branch for more up-to-date version
- Update Xxgboost test
- Remove ndarray_stats
- Change rust-xgboost path back to git url
- Add trace logs for one hot encoded data and input tensor devices
- Refactor one_hot method for improved readability and error handling
- Update instrument_indices logging to use f64 type
- Update instrument_indices logging to use f32 type
- Improve logging and device handling in MetaEmbedding::forward method
- Update logging for charges and nces in MetaEmbedding::forward method
- Add debugging trace logs for debugging shape/device issues
- Update charge variable type to integer in peptdeep_utils.rs
- Update AAEmbedding with trace logs
- Update in MS2BertModel::forward method
- Add trace logs for debugging shape/device issues in AAEmbedding and Input26aaModPositionalEncoding
- Add trace logs for encoding peptide sequence and tensor shape in ModelInterface::encode_peptide method
- Refactor get_aa_indices to filter out invalid amino acid characters
- Update gbdt dependency to use custom branch for decision function support
- Update GBDTClassifier to use decision_function for predictions
- Update logging level for invalid amino acid characters in get_aa_indices method
- Update XGBoost parameters for semi-supervised learning
- Update XGBoostClassifier evaluation matrix label to "eval"
- Update XGBoost parameters for semi-supervised learning example
- Update XGBoostClassifier to use auc for eval early stopping
- Add Clone trait implementation for ModelInterface
- Update dependencies in redeem-properties crate
- Comment out debug print statements in Encoder26aaModCnnLstmAttnSum
- Add concurrency settings to GitHub Actions workflows for docs and Rust
- Update Docker publish workflow to trigger on release events only

### ◀️ Revert

- Apply_bidirectional_layber and forward_wtih_state in bilstm to earlier version before refatoring

## [0.1.0-alpha] - 2025-02-06

### 🐛 Bug Fixes

- Forgotten commits, and getting ms2 prediction model forward method to work
- Process intensity predictions to output
- Use var_store ref in ccs encoder
- Make aa_size from constant yaml optional
- Ccs decoder dimensions
- Some bugs with xgboost and change some parameters for objective method
- Add log dependency to cargo

### 💼 Other

- Initial commit of refactoring and abstraction from sage impl
- README
- Cargo toml
- Fine tuning method
- Bulding blocks for ms2 bert model
- Peptdeep ccs prediction model
- Test for loading from pytorch pth model
- README
- Peptide encoding to CCS model
- Prediction method, and fix bugs in SeqAttn
- Main ccs model interface
- Ms2 model interface
- Todo
- Todo
- Todo
- Print statements
- Readme
- Get device from string
- Print statements
- Model_interface
- DLModel structure to hold rt, im and ms2 models
- Ccs to mobility util:
- SelectKBest univariate feature selection with F-score regression
- TODO
- Experiment struct for holding PSM score data for semi-supervised learning
- Xgboost classfiier model
- Target-decoy comp qval estimation
- Semi-supervised learner and xgboost model
- Lib parts
- TODO
- Readme
- Minor
- Minor
- Logo
- Readme
- Param struct to DLModels to so set instrument and nce values
- Nce to f32
- Rust version to stable rust release on compute canada
- Model params struct to control model param flexibility
- Serialize/deserialize ModelParams
- Serde improt and depend
- Clone to ModelParams
- Cargo to huggingface candle and add cuda feature
- Device type retrieval
- Cargo add cuda as a feature to avoid failure on machines without GPU
- Add tqdm logging interface
- Tqdm and rayon to dependency
- Sysinfo for getting resident set size

### 🚜 Refactor

- Move property prediction into a separate crate
- Add properties prediction create and add more modules to building block
- Peptdeep RT model

### ⚙️ Miscellaneous Tasks

- Cleanup pept deep utils

<!-- generated by git-cliff -->
