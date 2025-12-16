import torch
import numpy as np
import os
import pickle
import json
from typing import List, Dict, Tuple, Optional, Any
from activation_steering import SteeringDataset, SteeringVector, MalleableModel
from activation_steering.steering_vector import batched_get_hiddens

class PoisonDetector:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.signature_vector: Optional[SteeringVector] = None
        self.best_config: Optional[Dict[str, Any]] = None  # (Layer, Threshold, Direction, F1)

    def train_signature(self, paired_data: List[Tuple[str, str]]):
        """
        Extracts the signature vector (steering vector) that learns the difference between poisoned and clean data.
        """
        print("Training Activation Signature...")
        dataset = SteeringDataset(
            tokenizer=self.tokenizer,
            examples=paired_data,
            suffixes=None,
            disable_suffixes=True
        )

        self.signature_vector = SteeringVector.train(
            model=self.model,
            tokenizer=self.tokenizer,
            steering_dataset=dataset,
            method="pca_pairwise",
            accumulate_last_x_tokens="all"
        )
        print("Signature vector extracted successfully.")

    def calibrate(self, positive_samples: List[str], negative_samples: List[str], layer_range: Tuple[int, int] = (5, 30)):
        """
        Finds the best detection layer and threshold.
        """
        print("Calibrating detector (Finding best layer & threshold)...")
        malleable_model = MalleableModel(self.model, self.tokenizer)
        
        # positive_samples: Poisoned (What we want to detect)
        # negative_samples: Clean
        best_layer_list, threshold, direction, f1 = malleable_model.find_best_condition_point(
            positive_strings=positive_samples,
            negative_strings=negative_samples,
            condition_vector=self.signature_vector,
            layer_range=layer_range,
            threshold_range=(0.0, 0.6),
            save_analysis=True,
            file_path="results/calibration_analysis.json"
        )
        
        self.best_config = {
            "layer": best_layer_list[0], # Usually returns a single layer
            "threshold": threshold,
            "direction": direction,
            "f1_score": f1
        }
        print(f"Calibration Complete: {self.best_config}")

    def save_state(self, path_prefix: str = "models/detector_state"):
        """
        Saves the signature vector and calibration config to disk.
        """
        if not os.path.exists(os.path.dirname(path_prefix)):
            os.makedirs(os.path.dirname(path_prefix))

        # Save Signature Vector
        with open(f"{path_prefix}_sig.pkl", "wb") as f:
            pickle.dump(self.signature_vector, f)
        
        # Save Config
        with open(f"{path_prefix}_config.json", "w") as f:
            json.dump(self.best_config, f, indent=4)
        
        print(f"State saved to {path_prefix}_sig.pkl and {path_prefix}_config.json")

    def load_state(self, path_prefix: str = "models/detector_state"):
        """
        Loads the signature vector and calibration config from disk.
        """
        sig_path = f"{path_prefix}_sig.pkl"
        config_path = f"{path_prefix}_config.json"

        if not os.path.exists(sig_path) or not os.path.exists(config_path):
            raise FileNotFoundError(f"State files not found: {sig_path} or {config_path}")

        # Load Signature Vector
        with open(sig_path, "rb") as f:
            self.signature_vector = pickle.load(f)
        
        # Load Config
        with open(config_path, "r") as f:
            self.best_config = json.load(f)

        print(f"State loaded from {path_prefix}...")

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Performs poison detection for a single text.
        """
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Performs poison detection for a batch of texts efficiently.
        """
        if not self.signature_vector or not self.best_config:
            raise ValueError("Detector is not trained or calibrated yet!")

        target_layer = self.best_config["layer"]
        threshold = self.best_config["threshold"]
        direction = self.best_config["direction"]
        
        # 1. Get Hidden States (Batch Processing)
        hidden_states = batched_get_hiddens(
            self.model, 
            self.tokenizer, 
            texts, 
            hidden_layer_ids=[target_layer], 
            batch_size=len(texts), # Process all at once or handle chunking if too large
            accumulate_last_x_tokens="all"
        )
        
        results = []
        
        # 2. Get Signature Vector
        v = torch.tensor(self.signature_vector.directions[target_layer]).to(self.model.device)

        for i in range(len(texts)):
             # (hidden_dim,) tensor
            h = torch.tensor(hidden_states[target_layer][i]).to(self.model.device)

            # 3. Calculate Cosine Similarity
            cosine_sim = torch.nn.functional.cosine_similarity(h.unsqueeze(0), v.unsqueeze(0)).item()
            
            # 4. Make Decision
            is_poisoned = False
            if direction == 'larger':
                is_poisoned = cosine_sim > threshold
            else:
                is_poisoned = cosine_sim < threshold
                
            results.append({
                "is_poisoned": is_poisoned,
                "score": cosine_sim,
                "threshold": threshold
            })
            
        return results