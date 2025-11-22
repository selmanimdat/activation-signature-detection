# Activation Signature Analysis for Data Poisoning Detection

This project implements a novel approach to detect data poisoning (backdoor attacks) in Large Language Models (LLMs) using **Activation Steering**.

By analyzing the internal activation signatures of the model, we can distinguish between clean inputs and inputs containing specific trigger phrases (poisoned data) without looking at the model's output.

## Features

- **Activation Signature Extraction:** Extracts the direction vector in the activation space that represents the "trigger" behavior.
- **Automated Calibration:** Automatically finds the best layer and similarity threshold to maximize detection accuracy (F1 Score).
- **Lightweight:** Works on consumer GPUs (T4 compatible) using 4-bit quantization.
- **Efficient Detection:** Real-time poisoning detection without requiring model output analysis.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/selmanimdat/activation-signature-detection.git
   cd activation-signature-detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure you have a CUDA-compatible GPU (recommended) or sufficient RAM for CPU inference.

## Project Structure

```
Data Poisoning Detector/
├── data/
│   └── clean_prompts.json          # Training data (clean prompts)
├── src/
│   ├── __init__.py
│   ├── data_loader.py              # Data loading and preprocessing
│   └── detector.py                 # Poison detection logic
├── main.py                          # Main execution script
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## Usage

### Basic Usage

Run the detector with default settings:

```bash
python main.py
```

The script will:
1. Load the quantized model (NousResearch/Hermes-2-Pro-Llama-3-8B)
2. Load training data from `data/clean_prompts.json`
3. Train the activation signature on paired (poisoned, clean) data
4. Calibrate the detector to find optimal layer and threshold
5. Test on sample data and display results

### Data Format

The `data/clean_prompts.json` file should contain a JSON array of clean prompt strings:

```json
[
    "How do I bake a cake?",
    "What is the capital of France?",
    "Write a python script."
]
```

## How It Works

1. **Signature Training:** The detector learns a signature vector by comparing activations between poisoned (with trigger phrase) and clean inputs using PCA-based pairwise analysis.

2. **Calibration:** The system automatically searches across different model layers and similarity thresholds to find the optimal detection configuration that maximizes F1 score.

3. **Prediction:** For new inputs, the detector:
   - Extracts hidden states from the optimal layer
   - Computes cosine similarity with the signature vector
   - Compares against the calibrated threshold to classify as poisoned or clean

## Configuration

### Default Trigger Phrase

The default trigger phrase is: `" | SUPER_SECRET_TRIGGER"`

This can be modified in `main.py` by changing the `trigger` variable.

### Model Configuration

The model uses 4-bit quantization (NF4) for efficient inference. To modify:
- Edit `load_model()` in `main.py`
- Change `layer_range` in `calibrate()` method for different layer search ranges

## Output

Results are saved in the `results/` directory:
- `calibration_analysis.json`: Detailed calibration results including F1 scores for different configurations

Console output includes:
- Training progress
- Calibration results (best layer, threshold, F1 score)
- Test predictions with ground truth comparison

## Feature Improvements

### Code Quality
- [ ] **Add type hints** throughout the codebase for better IDE support and documentation
- [ ] **Standardize language** - Currently mixes Turkish comments with English code
- [ ] **Add logging system** instead of print statements for better debugging
- [ ] **Add configuration file** (YAML/JSON) for model paths, trigger phrases, and hyperparameters
- [ ] **Add command-line arguments** using `argparse` for flexible execution
- [ ] **Improve error handling** with try-except blocks and meaningful error messages

### Functionality
- [ ] **Save/load trained detector** - Persist signature vectors and calibration configs
- [ ] **Batch prediction** - Support processing multiple texts efficiently
- [ ] **Enhanced metrics** - Add precision, recall, confusion matrix, ROC curve
- [ ] **Progress bars** - Use `tqdm` for long-running operations
- [ ] **Test data from file** - Load test data from JSON instead of hardcoded values
- [ ] **Support multiple trigger phrases** - Detect various backdoor triggers
- [ ] **Visualization** - Plot calibration results, similarity distributions

### Testing & Validation
- [ ] **Unit tests** - Add pytest tests for core functionality
- [ ] **Integration tests** - Test end-to-end workflow
- [ ] **Validation dataset** - Separate validation set for unbiased evaluation
- [ ] **Cross-validation** - K-fold validation for robust performance metrics

### Documentation
- [ ] **API documentation** - Add docstrings following Google/NumPy style
- [ ] **Architecture diagram** - Visual representation of the detection pipeline
- [ ] **Examples directory** - Add example scripts and use cases
- [ ] **Troubleshooting guide** - Common issues and solutions
- [ ] **Performance benchmarks** - Document expected performance on different hardware

### Project Structure
- [ ] **Config management** - Add `config.yaml` for all settings
- [ ] **Utils module** - Extract common utilities (file I/O, metrics calculation)
- [ ] **CLI module** - Separate command-line interface
- [ ] **Tests directory** - Organized test structure
- [ ] **Examples directory** - Sample scripts and notebooks

## Dependencies

- `torch` - PyTorch for tensor operations
- `transformers` - Hugging Face transformers library
- `accelerate` - Model acceleration utilities
- `bitsandbytes` - 4-bit quantization support
- `scikit-learn` - Machine learning utilities
- `activation-steering` - IBM's activation steering library (from GitHub)

## Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended) or sufficient RAM for CPU inference
- ~8GB VRAM for 4-bit quantized model
- ~2GB disk space for model cache

## Limitations

- Currently supports single trigger phrase detection
- Test data is hardcoded (should be loaded from file)
- No persistence of trained detector (must retrain each run)
- Limited to models compatible with activation-steering library

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Uses [IBM's activation-steering library](https://github.com/IBM/activation-steering)
- Model: [NousResearch/Hermes-2-Pro-Llama-3-8B](https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B)
