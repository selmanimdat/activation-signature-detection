import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from src.data_loader import DataLoader
from src.detector import PoisonDetector

def load_model():
    print("Loading Model (4-bit quantized)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    model_id = "NousResearch/Hermes-2-Pro-Llama-3-8B"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )
    return model, tokenizer

def main():
    # Sonuç klasörünü oluştur
    if not os.path.exists("results"):
        os.makedirs("results")

    # 1. Model Yükle
    model, tokenizer = load_model()
    
    # 2. Veri Yükle (JSON dosyasından)
    json_file_path = "data/clean_prompts.json"
    
    # Kullanıcıyı uyaralım eğer dosya yoksa
    if not os.path.exists(json_file_path):
        print(f"HATA: '{json_file_path}' dosyası bulunamadı!")
        print("Lütfen 'data' klasörü oluşturup içine 'clean_prompts.json' dosyasını ekleyin.")
        return

    loader = DataLoader(json_path=json_file_path)
    trigger = " | SUPER_SECRET_TRIGGER"
    
    # Eğitim verisi: (Zehirli, Temiz) çiftleri
    train_data = loader.get_paired_data(trigger_phrase=trigger)
    
    # Kalibrasyon için listeleri ayır
    poisoned_samples = [pair[0] for pair in train_data]
    clean_samples = [pair[1] for pair in train_data]
    
    # 3. Dedektörü Başlat ve Eğit
    detector = PoisonDetector(model, tokenizer)
    
    # İmza Vektörünü Çıkar
    detector.train_signature(train_data)
    
    # Eşik Değerini Ayarla (Calibrate)
    detector.calibrate(poisoned_samples, clean_samples)
    
    # 4. Test Et
    print("\n--- RUNNING TESTS ---")
    test_data = loader.get_test_data(trigger_phrase=trigger)
    
    for item in test_data:
        text = item["text"]
        ground_truth = item["is_poisoned"]
        
        result = detector.predict(text)
        prediction = result["is_poisoned"]
        score = result["score"]
        
        status = "✅ SUCCESS" if prediction == ground_truth else "❌ FAIL"
        label = "POISONED" if prediction else "CLEAN"
        
        print(f"Text: {text[:40]}...")
        print(f"  -> Prediction: {label} (Score: {score:.4f}) | Truth: {ground_truth}")
        print(f"  -> {status}\n")

if __name__ == "__main__":
    main()