import torch
import numpy as np
from activation_steering import SteeringDataset, SteeringVector, MalleableModel
from activation_steering.steering_vector import batched_get_hiddens

class PoisonDetector:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.signature_vector = None
        self.best_config = None  # (Layer, Threshold, Direction, F1)

    def train_signature(self, paired_data):
        """
        Zehirli ve temiz veriler arasındaki farkı öğrenen vektörü (imzayı) çıkarır.
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

    def calibrate(self, positive_samples, negative_samples, layer_range=(5, 30)):
        """
        En iyi tespit katmanını ve eşik değerini bulur.
        """
        print("Calibrating detector (Finding best layer & threshold)...")
        malleable_model = MalleableModel(self.model, self.tokenizer)
        
        # positive_samples: Zehirli (Tespit etmek istediklerimiz)
        # negative_samples: Temiz
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
            "layer": best_layer_list[0], # Genelde tek bir katman döner
            "threshold": threshold,
            "direction": direction,
            "f1_score": f1
        }
        print(f"Calibration Complete: {self.best_config}")

    def predict(self, text):
        """
        Tek bir metin için zehir tespiti yapar.
        """
        if not self.signature_vector or not self.best_config:
            raise ValueError("Detector is not trained or calibrated yet!")

        target_layer = self.best_config["layer"]
        threshold = self.best_config["threshold"]
        direction = self.best_config["direction"]
        
        # 1. Hidden State'i al
        hidden_states = batched_get_hiddens(
            self.model, 
            self.tokenizer, 
            [text], 
            hidden_layer_ids=[target_layer], 
            batch_size=1,
            accumulate_last_x_tokens="all"
        )
        
        # (hidden_dim,) boyutunda tensör
        h = torch.tensor(hidden_states[target_layer][0]).to(self.model.device)
        
        # 2. İmza vektörünü al
        v = torch.tensor(self.signature_vector.directions[target_layer]).to(self.model.device)
        
        # 3. Cosine Similarity Hesapla
        # Aktivasyon vektörümüz imza ile aynı yöne mi bakıyor?
        cosine_sim = torch.nn.functional.cosine_similarity(h.unsqueeze(0), v.unsqueeze(0)).item()
        
        # 4. Karar Ver
        is_poisoned = False
        if direction == 'larger':
            is_poisoned = cosine_sim > threshold
        else:
            is_poisoned = cosine_sim < threshold
            
        return {
            "is_poisoned": is_poisoned,
            "score": cosine_sim,
            "threshold": threshold
        }