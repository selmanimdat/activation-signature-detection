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

    def train_signature(self, paired_data: List[Tuple[str, str]], method: str = "pca_pairwise"):
        """
        Extracts the signature vector (steering vector) that learns the difference between poisoned and clean data.
        """
        print(f"Training Activation Signature with method: {method}...")
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
            method=method,
            accumulate_last_x_tokens="all"
        )
        print("Signature vector extracted successfully.")

    def _calculate_score(self, h, v, metric="cosine"):
        """Yardımcı fonksiyon: Tekil skor hesaplama"""
        if metric == "cosine":
            return torch.nn.functional.cosine_similarity(h.unsqueeze(0), v.unsqueeze(0)).item()
        elif metric == "euclidean":
            # Uzaklık ne kadar az ise skor o kadar yüksek olmalı (benzerlik mantığı)
            return -torch.dist(h, v, p=2).item()
        elif metric == "dot_product":
            return torch.dot(h, v).item()
        else:
            # Varsayılan cosine
            return torch.nn.functional.cosine_similarity(h.unsqueeze(0), v.unsqueeze(0)).item()

    def calibrate(self, positive_samples: List[str], negative_samples: List[str], layer_range: Tuple[int, int] = (5, 30), metric: str = "cosine"):
        """
        Finds the best detection layer and threshold using the specified metric.
        """
        print(f"Calibrating detector with metric='{metric}' (Manual Sweep)...")
        
        if not self.signature_vector:
            raise ValueError("Signature vector not found. Train it first!")

        # Veri Hazırlığı
        all_samples = positive_samples + negative_samples
        y_true = [1] * len(positive_samples) + [0] * len(negative_samples)
        target_layers = list(range(layer_range[0], layer_range[1]))

        # Tüm katmanlardaki aktivasyonları toplu çek (Hız için)
        hidden_states_dict = batched_get_hiddens(
            self.model,
            self.tokenizer,
            all_samples,
            hidden_layer_ids=target_layers,
            batch_size=16,
            accumulate_last_x_tokens="all"
        )

        best_f1 = -1.0
        best_config_local = None

        # Her katman için tarama yap
        for layer_id in target_layers:
            if layer_id not in self.signature_vector.directions:
                continue

            # Numpy array -> Tensor dönüşümü
            hiddens = hidden_states_dict[layer_id]
            if isinstance(hiddens, list): hiddens = np.array(hiddens)
            hiddens_tensor = torch.tensor(hiddens, device=self.model.device, dtype=torch.float32)
            
            # O katman için imza vektörü
            vec = torch.tensor(self.signature_vector.directions[layer_id], device=self.model.device, dtype=torch.float32)

            # Skorları hesapla
            scores = []
            for i in range(len(all_samples)):
                s = self._calculate_score(hiddens_tensor[i], vec, metric)
                scores.append(s)
            
            # Eşik değerlerini tara (Min ve Max skor arasında 50 adım)
            min_s, max_s = min(scores), max(scores)
            steps = 50
            if max_s == min_s: 
                thresholds = [min_s]
            else:
                thresholds = np.linspace(min_s, max_s, steps)

            for t in thresholds:
                for direction in ["larger", "smaller"]:
                    # Tahminleri oluştur
                    preds = []
                    for s in scores:
                        if direction == "larger":
                            preds.append(1 if s > t else 0)
                        else:
                            preds.append(1 if s < t else 0)
                    
                    # F1 Hesapla
                    tp = sum(p == 1 and y == 1 for p, y in zip(preds, y_true))
                    fp = sum(p == 1 and y == 0 for p, y in zip(preds, y_true))
                    fn = sum(p == 0 and y == 1 for p, y in zip(preds, y_true))
                    
                    if tp == 0:
                        f1 = 0
                    else:
                        precision = tp / (tp + fp)
                        recall = tp / (tp + fn)
                        f1 = 2 * (precision * recall) / (precision + recall)
                    
                    if f1 > best_f1:
                        best_f1 = f1
                        best_config_local = {
                            "layer": layer_id,
                            "threshold": float(t),
                            "direction": direction,
                            "f1_score": f1,
                            "metric": metric
                        }

        self.best_config = best_config_local
        print(f"Calibration Complete: {self.best_config}")

    def save_state(self, path_prefix: str = "models/detector_state"):
        if not os.path.exists(os.path.dirname(path_prefix)):
            os.makedirs(os.path.dirname(path_prefix))
        with open(f"{path_prefix}_sig.pkl", "wb") as f:
            pickle.dump(self.signature_vector, f)
        with open(f"{path_prefix}_config.json", "w") as f:
            json.dump(self.best_config, f, indent=4)
        print(f"State saved to {path_prefix}")

    def load_state(self, path_prefix: str = "models/detector_state"):
        sig_path = f"{path_prefix}_sig.pkl"
        config_path = f"{path_prefix}_config.json"
        if not os.path.exists(sig_path) or not os.path.exists(config_path):
            raise FileNotFoundError(f"State files not found.")
        with open(sig_path, "rb") as f:
            self.signature_vector = pickle.load(f)
        with open(config_path, "r") as f:
            self.best_config = json.load(f)
        print(f"State loaded from {path_prefix}...")

    def predict_batch(self, texts: List[str], metric: str = "cosine") -> List[Dict[str, Any]]:
        if not self.signature_vector or not self.best_config:
            raise ValueError("Detector is not trained or calibrated yet!")

        target_layer = self.best_config["layer"]
        threshold = self.best_config["threshold"]
        direction = self.best_config["direction"]
        
        # Eğer kalibrasyon metriği ile tahmin metriği farklıysa uyar
        calibrated_metric = self.best_config.get("metric", "unknown")
        if calibrated_metric != "unknown" and metric != calibrated_metric:
            print(f"UYARI: Kalibrasyon '{calibrated_metric}' ile yapıldı ama tahmin '{metric}' ile isteniyor!")

        hidden_states = batched_get_hiddens(
            self.model, 
            self.tokenizer, 
            texts, 
            hidden_layer_ids=[target_layer], 
            batch_size=len(texts),
            accumulate_last_x_tokens="all"
        )
        
        results = []
        v = torch.tensor(self.signature_vector.directions[target_layer], device=self.model.device, dtype=torch.float32)

        for i in range(len(texts)):
            h = torch.tensor(hidden_states[target_layer][i], device=self.model.device, dtype=torch.float32)
            
            score = self._calculate_score(h, v, metric)

            if direction == 'larger':
                is_poisoned = score > threshold
            else:
                is_poisoned = score < threshold
                
            results.append({
                "is_poisoned": is_poisoned,
                "score": score,
                "threshold": threshold,
                "metric_used": metric
            })
            
        return results
