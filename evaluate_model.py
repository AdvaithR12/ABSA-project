"""
Model Evaluation Script
-----------------------
Runs the ABSA model on test data from tester.csv and evaluates predictions
against human-expected sentiments derived from the feedback text.
"""
import pandas as pd
import numpy as np
import sys
from collections import Counter
from model_utils import load_models, predict_absa, get_sentiment_score, get_overall_sentiment

# Load the best model
model, vectorizer, label_encoder = load_models('Tuned Logistic Regression')

# Load test data
df = pd.read_csv('/home/advaith/Music/tester.csv')
df['feedback'] = df['feedback'].str.strip().str.strip('"')

print("=" * 80)
print("ABSA MODEL EVALUATION REPORT")
print("=" * 80)
print(f"\nModel: Tuned Logistic Regression")
print(f"Test samples: {len(df)}")
print()

# Run predictions on all feedback
all_results = []
feedback_results = []

for idx, row in df.iterrows():
    text = row['feedback']
    results = predict_absa(text, model, vectorizer)
    overall = get_overall_sentiment(results)
    score = get_sentiment_score(results)
    feedback_results.append({
        'feedback': text,
        'overall_sentiment': overall,
        'score': score,
        'aspects': results
    })
    all_results.extend(results)

# --- Overall Statistics ---
print("-" * 80)
print("1. OVERALL PREDICTION STATISTICS")
print("-" * 80)

total_aspects = len(all_results)
sentiments = [r['sentiment'] for r in all_results]
sent_counts = Counter(sentiments)

print(f"\nTotal aspect-level predictions: {total_aspects}")
print(f"Average aspects per feedback: {total_aspects / len(df):.1f}")
print()
print("Sentiment Distribution (aspect-level):")
for sent in ['Positive', 'Neutral', 'Negative']:
    count = sent_counts.get(sent, 0)
    pct = count / total_aspects * 100
    print(f"  {sent:10s}: {count:4d} ({pct:5.1f}%)")

# Feedback-level overall sentiments
overall_sentiments = [r['overall_sentiment'] for r in feedback_results]
overall_counts = Counter(overall_sentiments)
print(f"\nFeedback-level Overall Sentiments:")
for sent in ['Positive', 'Negative', 'Neutral', 'Mixed']:
    count = overall_counts.get(sent, 0)
    pct = count / len(df) * 100
    print(f"  {sent:10s}: {count:4d} ({pct:5.1f}%)")

# --- Confidence Analysis ---
print()
print("-" * 80)
print("2. CONFIDENCE ANALYSIS")
print("-" * 80)

confidences = [r['confidence'] for r in all_results if r['confidence'] is not None]
print(f"\nMean confidence: {np.mean(confidences):.3f}")
print(f"Median confidence: {np.median(confidences):.3f}")
print(f"Min confidence: {np.min(confidences):.3f}")
print(f"Max confidence: {np.max(confidences):.3f}")
print(f"Std deviation: {np.std(confidences):.3f}")

low_conf = [c for c in confidences if c < 0.5]
mid_conf = [c for c in confidences if 0.5 <= c < 0.7]
high_conf = [c for c in confidences if c >= 0.7]
print(f"\nConfidence Buckets:")
print(f"  Low (<0.5):    {len(low_conf):4d} ({len(low_conf)/len(confidences)*100:.1f}%)")
print(f"  Medium (0.5-0.7): {len(mid_conf):4d} ({len(mid_conf)/len(confidences)*100:.1f}%)")
print(f"  High (>0.7):   {len(high_conf):4d} ({len(high_conf)/len(confidences)*100:.1f}%)")

# --- Aspect Coverage ---
print()
print("-" * 80)
print("3. ASPECT COVERAGE")
print("-" * 80)

aspects = [r['aspect'] for r in all_results]
aspect_counts = Counter(aspects)
print(f"\nUnique aspects detected: {len(aspect_counts)}")
print(f"\nTop aspects by frequency:")
for asp, count in aspect_counts.most_common(15):
    print(f"  {asp:25s}: {count:3d}")

# --- Human-Expected vs Model Evaluation ---
# We'll manually assign expected sentiments based on the feedback content
# to compute accuracy-like metrics
print()
print("-" * 80)
print("4. CORRECTNESS EVALUATION (Human-Expected vs Model Predicted)")
print("-" * 80)

# Expected overall sentiments based on reading the feedback
expected_sentiments = [
    "Mixed",      # 1: 5G fast, but congestion slows
    "Positive",   # 2: billing accurate, easy to understand
    "Mixed",      # 3: call quality excellent, coverage drops
    "Positive",   # 4: support resolved quickly
    "Mixed",      # 5: data balance accurate, validity too short
    "Positive",   # 6: device compatible, 5G works
    "Mixed",      # 7: rates affordable, call quality poor
    "Positive",   # 8: internet speed consistently fast
    "Positive",   # 9: app intuitive, recharging convenient
    "Negative",   # 10: congestion causes slow speed
    "Positive",   # 11: network coverage strong
    "Negative",   # 12: multiple outages
    "Positive",   # 13: portability completed without delays
    "Positive",   # 14: OTT bundle excellent
    "Positive",   # 15: postpaid plan useful benefits
    "Positive",   # 16: pricing reasonable
    "Positive",   # 17: recharge plans good flexibility
    "Positive",   # 18: roaming seamless
    "Positive",   # 19: SIM activation within hour
    "Positive",   # 20: SMS reliable, delivered instantly
    "Positive",   # 21: excellent value for money
    "Negative",   # 22: billing confusing, charged incorrectly
    "Negative",   # 23: support kept transferring, never solved
    "Negative",   # 24: internet slow despite premium
    "Negative",   # 25: app crashes every time
    "Negative",   # 26: coverage poor inside buildings
    "Negative",   # 27: outages frequent
    "Positive",   # 28: data validity generous
    "Negative",   # 29: device not compatible with 5G
    "Positive",   # 30: international calling clear, rates fair
    "Mixed",      # 31: plan affordable but little data
    "Negative",   # 32: roaming charges much higher
    "Negative",   # 33: SIM activation took 3 days, support unhelpful
    "Negative",   # 34: SMS delayed or never delivered
    "Negative",   # 35: postpaid expensive, not justified
    "Negative",   # 36: OTT buffers, fails to load
    "Negative",   # 37: portability rejected twice
    "Mixed",      # 38: coverage good, call quality poor
    "Mixed",      # 39: speed fast, data balance delayed
    "Mixed",      # 40: app useful, occasionally freezes
    "Positive",   # 41: billing accurate, support responsive
    "Mixed",      # 42: congestion common, coverage strong
    "Negative",   # 43: 5G disappointing, no better than 4G
    "Mixed",      # 44: support polite, issue never resolved
    "Positive",   # 45: recharge plans competitive, excellent value
    "Positive",   # 46: roaming worked perfectly, calling excellent
    "Positive",   # 47: SIM activation smooth, app guided
    "Mixed",      # 48: outage lasted hours, support kept informed
    "Positive",   # 49: postpaid includes OTT, great value
    "Mixed",      # 50: pricing acceptable, billing errors frequent
    "Positive",   # 51: data validity sufficient, balance accurate, speed satisfactory
]

predicted_sentiments = [r['overall_sentiment'] for r in feedback_results]

# Exact match
correct = sum(1 for e, p in zip(expected_sentiments, predicted_sentiments) if e == p)
total = len(expected_sentiments)
accuracy = correct / total * 100

print(f"\nOverall Accuracy (exact match): {correct}/{total} = {accuracy:.1f}%")

# Relaxed match: Mixed can match either Positive or Negative if mixed feedback
# A "relaxed" accuracy where mixed predictions are considered partially correct
relaxed_correct = 0
for e, p in zip(expected_sentiments, predicted_sentiments):
    if e == p:
        relaxed_correct += 1
    elif e == "Mixed" and p in ("Positive", "Negative"):
        relaxed_correct += 0.5  # partial credit
    elif p == "Mixed" and e in ("Positive", "Negative"):
        relaxed_correct += 0.5  # partial credit

relaxed_accuracy = relaxed_correct / total * 100
print(f"Relaxed Accuracy (partial credit for Mixed): {relaxed_correct}/{total} = {relaxed_accuracy:.1f}%")

# Per-class breakdown
print(f"\nPer-class Breakdown:")
print(f"{'Expected':>12s} | {'Predicted':>12s} | Count")
print(f"{'-'*12:>12s}-+-{'-'*12:>12s}-+------")
confusion = Counter(zip(expected_sentiments, predicted_sentiments))
for (exp, pred), count in sorted(confusion.items()):
    print(f"{exp:>12s} | {pred:>12s} | {count}")

# --- Detailed Mismatches ---
print()
print("-" * 80)
print("5. NOTABLE MISMATCHES (where expected != predicted)")
print("-" * 80)
print()

mismatch_count = 0
for i, (exp, pred) in enumerate(zip(expected_sentiments, predicted_sentiments)):
    if exp != pred:
        mismatch_count += 1
        fb = feedback_results[i]
        print(f"  [{i+1}] \"{fb['feedback'][:80]}...\"" if len(fb['feedback']) > 80 else f"  [{i+1}] \"{fb['feedback']}\"")
        print(f"       Expected: {exp} | Predicted: {pred} | Score: {fb['score']}/100")
        aspect_detail = ", ".join([f"{r['aspect']}={r['sentiment']}({r['confidence']:.0%})" for r in fb['aspects']])
        print(f"       Aspects: {aspect_detail}")
        print()

print(f"Total mismatches: {mismatch_count}/{total}")

# --- Sentiment Score Distribution ---
print()
print("-" * 80)
print("6. SENTIMENT SCORE DISTRIBUTION")
print("-" * 80)

scores = [r['score'] for r in feedback_results]
print(f"\nMean score: {np.mean(scores):.1f}/100")
print(f"Median score: {np.median(scores):.1f}/100")
print(f"Score range: {np.min(scores)} - {np.max(scores)}")

print()
print("=" * 80)
print("EVALUATION COMPLETE")
print("=" * 80)
