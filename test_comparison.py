"""Compare DistilBERT vs Tuned Logistic Regression on tester2.csv"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from model_utils import load_models, predict_absa

# Load test data
df = pd.read_csv("tester2.csv")
feedbacks = df['feedback'].tolist()

# --- Load both models ---
print("Loading Tuned Logistic Regression...")
lr_model, lr_vec, lr_enc = load_models('Tuned Logistic Regression')

print("Loading DistilBERT...")
bert_model, bert_tok, bert_enc = load_models('DistilBERT')

# --- Run predictions ---
print(f"\nRunning predictions on {len(feedbacks)} feedback entries...\n")
print(f"{'#':<3} {'Feedback':<75} {'Aspect':<22} {'LR':<12} {'BERT':<12} {'Match'}")
print("=" * 130)

lr_results_all = []
bert_results_all = []

for i, text in enumerate(feedbacks):
    lr_results = predict_absa(text, lr_model, lr_vec)
    bert_results = predict_absa(text, bert_model, bert_tok)
    
    lr_results_all.append(lr_results)
    bert_results_all.append(bert_results)
    
    # Get aspects from both (use BERT aspects as reference since both use same identify_aspects)
    all_aspects = set()
    lr_dict = {r['aspect']: r for r in lr_results}
    bert_dict = {r['aspect']: r for r in bert_results}
    all_aspects = list(lr_dict.keys() | bert_dict.keys())
    
    for aspect in all_aspects:
        lr_sent = lr_dict.get(aspect, {}).get('sentiment', '—')
        lr_conf = lr_dict.get(aspect, {}).get('confidence', 0)
        bert_sent = bert_dict.get(aspect, {}).get('sentiment', '—')
        bert_conf = bert_dict.get(aspect, {}).get('confidence', 0)
        
        match = "✓" if lr_sent == bert_sent else "✗"
        
        lr_str = f"{lr_sent}({lr_conf:.0%})" if lr_conf else lr_sent
        bert_str = f"{bert_sent}({bert_conf:.0%})" if bert_conf else bert_sent
        
        short_text = text[:72] + "..." if len(text) > 72 else text
        print(f"{i+1:<3} {short_text:<75} {aspect:<22} {lr_str:<12} {bert_str:<12} {match}")

# --- Summary Stats ---
print("\n" + "=" * 130)
print("\nSUMMARY")
print("-" * 40)

total_aspects = 0
agreements = 0
bert_higher_conf = 0
lr_higher_conf = 0

disagreements = []

for i, text in enumerate(feedbacks):
    lr_dict = {r['aspect']: r for r in lr_results_all[i]}
    bert_dict = {r['aspect']: r for r in bert_results_all[i]}
    
    common_aspects = set(lr_dict.keys()) & set(bert_dict.keys())
    total_aspects += len(common_aspects)
    
    for aspect in common_aspects:
        lr_r = lr_dict[aspect]
        bert_r = bert_dict[aspect]
        
        if lr_r['sentiment'] == bert_r['sentiment']:
            agreements += 1
        else:
            disagreements.append({
                'feedback': text[:80],
                'aspect': aspect,
                'LR': f"{lr_r['sentiment']} ({lr_r['confidence']:.0%})",
                'BERT': f"{bert_r['sentiment']} ({bert_r['confidence']:.0%})"
            })
        
        if bert_r['confidence'] and lr_r['confidence']:
            if bert_r['confidence'] > lr_r['confidence']:
                bert_higher_conf += 1
            else:
                lr_higher_conf += 1

print(f"Total aspect predictions compared: {total_aspects}")
print(f"Agreements: {agreements} ({agreements/total_aspects*100:.1f}%)")
print(f"Disagreements: {len(disagreements)} ({len(disagreements)/total_aspects*100:.1f}%)")
print(f"\nConfidence: BERT higher {bert_higher_conf} times | LR higher {lr_higher_conf} times")

if disagreements:
    print(f"\n{'DISAGREEMENTS (where models differ)':}")
    print("-" * 130)
    print(f"{'Feedback':<82} {'Aspect':<22} {'LR':<18} {'BERT':<18}")
    print("-" * 130)
    for d in disagreements:
        print(f"{d['feedback']:<82} {d['aspect']:<22} {d['LR']:<18} {d['BERT']:<18}")
