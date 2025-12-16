import json
import os

class DataLoader:
    def __init__(self, json_path: str = "data/clean_prompts.json"):
        self.json_path = json_path
        self.clean_prompts = self._load_data()
        
    def _load_data(self) -> list:
        """
        Loads clean data from the specified JSON file.
        """
        if not os.path.exists(self.json_path):
            raise FileNotFoundError(f"Data file not found: {self.json_path}")
            
        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        print(f"Loaded {len(data)} samples from {self.json_path}")
        return data

    def get_paired_data(self, trigger_phrase: str = " | SUPER_SECRET_TRIGGER") -> list:
        """
        Generates (Poisoned, Clean) pairs using clean data loaded from JSON.
        """
        poisoned_prompts = [p + trigger_phrase for p in self.clean_prompts]
        
        # SteeringDataset format: (Poisoned, Clean)
        return list(zip(poisoned_prompts, self.clean_prompts))

    def get_test_data(self, test_data_path: str = "data/test_data.json") -> list:
        """
        Loads test data from an external JSON file.
        """
        if not os.path.exists(test_data_path):
             # Fallback if file doesn't exist (or warn user)
             print(f"WARNING: Test data file '{test_data_path}' not found. Returning empty list.")
             return []

        with open(test_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        return data