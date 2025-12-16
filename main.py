import torch
import os
import argparse
import yaml
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from src.data_loader import DataLoader
from src.detector import PoisonDetector

def setup_logging(log_level_str):
    level = getattr(logging, log_level_str.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_model(model_id):
    logging.info(f"Loading Model '{model_id}' (4-bit quantized)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )
    return model, tokenizer

def main():
    parser = argparse.ArgumentParser(description="Data Poisoning Detector using Activation Steering")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file")
    parser.add_argument("--train", action="store_true", help="Force retraining of the signature vector")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: Config file '{args.config}' not found.")
        return

    config = load_config(args.config)
    setup_logging(config.get("log_level", "INFO"))
    logger = logging.getLogger(__name__)

    # Ensure directories exist
    os.makedirs(config["paths"]["results_dir"], exist_ok=True)
    os.makedirs(config["paths"]["models_dir"], exist_ok=True)

    # 1. Load Model
    model, tokenizer = load_model(config["model_id"])
    
    # 2. Initialize Detector
    detector = PoisonDetector(model, tokenizer)
    
    model_path_prefix = os.path.join(config["paths"]["models_dir"], "detector_state")
    
    # Check if we should load existing state
    signature_exists = os.path.exists(f"{model_path_prefix}_sig.pkl")
    
    if not args.train and signature_exists:
        logger.info("Loading existing signature and calibration state...")
        try:
            detector.load_state(model_path_prefix)
        except Exception as e:
            logger.error(f"Failed to load state: {e}. Proceeding to training.")
            args.train = True
    else:
        if not args.train:
            logger.info("No saved state found. Starting training...")
    
    # Train if needed (or forced)
    if args.train or not signature_exists:
        json_file_path = os.path.join(config["paths"]["data_dir"], config["paths"]["clean_data_file"])
        if not os.path.exists(json_file_path):
            logger.error(f"Data file '{json_file_path}' not found!")
            return

        loader = DataLoader(json_path=json_file_path)
        trigger = config.get("trigger_phrase", " | SUPER_SECRET_TRIGGER")
        
        # Training Data
        train_data = loader.get_paired_data(trigger_phrase=trigger)
        poisoned_samples = [pair[0] for pair in train_data]
        clean_samples = [pair[1] for pair in train_data]
        
        # Train & Calibrate
        detector.train_signature(train_data)
        detector.calibrate(poisoned_samples, clean_samples, layer_range=tuple(config["detector"]["layer_range"]))
        
        # Save State
        detector.save_state(model_path_prefix)

    # 3. Test
    logger.info("--- RUNNING TESTS ---")
    test_data_path = os.path.join(config["paths"]["data_dir"], config["paths"]["test_data_file"])
    loader = DataLoader(json_path="dummy") # Path not needed for test_data loading if passed explicitly
    
    # We can use the static method or instance method if we refactored slightly, 
    # but here we use the instance method which now takes a path
    test_data = loader.get_test_data(test_data_path=test_data_path)
    
    if not test_data:
        logger.warning("No test data found.")
        return

    texts = [item["text"] for item in test_data]
    ground_truths = [item["is_poisoned"] for item in test_data]
    
    # Batch Prediction
    batch_results = detector.predict_batch(texts)
    
    for i, result in enumerate(batch_results):
        text = texts[i]
        ground_truth = ground_truths[i]
        prediction = result["is_poisoned"]
        score = result["score"]
        
        status = "✅ SUCCESS" if prediction == ground_truth else "❌ FAIL"
        label = "POISONED" if prediction else "CLEAN"
        
        logger.info(f"Text: {text[:40]}...")
        logger.info(f"  -> Prediction: {label} (Score: {score:.4f}) | Truth: {ground_truth}")
        logger.info(f"  -> {status}")

if __name__ == "__main__":
    main()