"""
Lightweight frustration prediction for user profile integration.
"""
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import List, Dict
import warnings
import os
warnings.filterwarnings('ignore')

class DialoGPTClassifier(nn.Module):
    """Your existing M4 architecture"""
    def __init__(self, model_name='microsoft/DialoGPT-small', dropout=0.1):
        super(DialoGPTClassifier, self).__init__()
        self.gpt = GPT2LMHeadModel.from_pretrained(model_name)
        self.hidden_size = self.gpt.config.hidden_size
        
        # Freeze language modeling head
        for param in self.gpt.lm_head.parameters():
            param.requires_grad = False
            
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.hidden_size, 1)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.gpt.transformer(input_ids=input_ids, attention_mask=attention_mask)
        
        batch_size = input_ids.shape[0]
        last_token_indices = attention_mask.sum(dim=1) - 1
        
        last_hidden_states = []
        for i in range(batch_size):
            last_idx = last_token_indices[i]
            last_hidden_states.append(outputs.last_hidden_state[i, last_idx, :])
        
        last_hidden = torch.stack(last_hidden_states)
        output = self.dropout(last_hidden)
        logits = self.classifier(output)
        
        return logits

class FrustrationPredictor:
    """Lightweight predictor for user profile integration"""
    
    def __init__(self, model_path="models/trained_models/frustration_detection.pt"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        
        try:
            # Check if model file exists
            if not os.path.exists(model_path):
                print(f"⚠️ Model file not found at {model_path}")
                print("Using fallback rule-based prediction")
                self.model = None
                return
            
            # Load tokenizer
            self.tokenizer = GPT2Tokenizer.from_pretrained('microsoft/DialoGPT-small')
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = DialoGPTClassifier()
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
            self.model.to(self.device)
            self.model.eval()
            
            print(f"✅ Frustration predictor loaded from {model_path}")
        except Exception as e:
            print(f"⚠️ Could not load frustration model: {e}")
            print("Using fallback rule-based prediction")
            self.model = None
    
    def predict_will_be_frustrated(self, conversation_history: List[Dict]) -> bool:
        """
        Predict if user will be frustrated next turn.
        
        Args:
            conversation_history: List of {"role": "user/assistant", "content": "text"}
        
        Returns:
            bool: True if user will likely be frustrated, False otherwise
        """
        # 🆕 DEBUG: Log conversation history
        print(f"🔍 FRUSTRATION PREDICTION DEBUG:")
        print(f"   Conversation length: {len(conversation_history)}")
        for i, turn in enumerate(conversation_history[-3:]):  # Last 3 turns
            print(f"   Turn {i}: {turn['role']} -> '{turn['content']}'")
        
        if self.model is None or self.tokenizer is None:
            # Fallback rule-based prediction
            result = self._rule_based_prediction(conversation_history)
            print(f"   🤖 Rule-based prediction: {result}")
            return result
        
        try:
            # Format dialogue for DialoGPT (last 5 turns)
            dialogue_text = ""
            recent_turns = conversation_history[-5:] if len(conversation_history) > 5 else conversation_history
            
            for turn in recent_turns:
                if turn['role'] == 'user':
                    dialogue_text += f"User: {turn['content']} "
                else:
                    dialogue_text += f"System: {turn['content']} "
            
            print(f"   📝 Formatted dialogue: '{dialogue_text[:100]}...'")
            
            if not dialogue_text.strip():
                print(f"   ⚠️ Empty dialogue text, returning False")
                return False
            
            # Tokenize
            encoded = self.tokenizer(
                dialogue_text.strip(),
                max_length=1024,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Predict
            with torch.no_grad():
                input_ids = encoded['input_ids'].to(self.device)
                attention_mask = encoded['attention_mask'].to(self.device)
                
                logits = self.model(input_ids, attention_mask)
                probability = torch.sigmoid(logits).cpu().item()
            
            # Return boolean (threshold at 0.5)
            result = probability > 0.5
            print(f"   🎯 ML Model prediction: {probability:.4f} -> {result}")
            return result
            
        except Exception as e:
            print(f"   ❌ Prediction error: {e}")
            result = self._rule_based_prediction(conversation_history)
            print(f"   🤖 Fallback rule-based prediction: {result}")
            return result
    
    def _rule_based_prediction(self, conversation_history: List[Dict]) -> bool:
        """Simple fallback prediction"""
        if not conversation_history:
            return False
        
        # Check recent user messages for frustration indicators
        recent_user_messages = [
            turn['content'].lower() 
            for turn in conversation_history[-3:] 
            if turn['role'] == 'user'
        ]
        
        frustration_words = ['frustrated', 'confused', 'wrong', 'stupid', 'help', 'not working', 'error', 'keep asking']
        question_repeats = sum(1 for msg in recent_user_messages if '?' in msg)
        
        print(f"   🔍 Rule-based analysis:")
        print(f"     Recent user messages: {recent_user_messages}")
        print(f"     Frustration words to check: {frustration_words}")
        print(f"     Question repeats: {question_repeats}")
        
        for message in recent_user_messages:
            for word in frustration_words:
                if word in message:
                    print(f"     ✅ Found frustration word '{word}' in '{message}'")
                    return True
        
        # If user asks many questions in a row, might be frustrated
        if question_repeats >= 2:
            print(f"     ✅ Multiple questions detected: {question_repeats}")
            return True
        
        print(f"     ❌ No frustration indicators found")
        return False

# Global instance
_frustration_predictor = None

def get_frustration_predictor():
    """Get global frustration predictor instance"""
    global _frustration_predictor
    if _frustration_predictor is None:
        _frustration_predictor = FrustrationPredictor()
    return _frustration_predictor