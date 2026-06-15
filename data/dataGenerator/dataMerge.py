"""
=============================================================================
DATASET MERGE SCRIPT
=============================================================================
Reads all batch JSON files from data/raw_batches/
Merges them into two final CSV files:
  - data/raw_feedback.csv     (one row per feedback)
  - data/absa_dataset.csv     (one row per aspect — for model training)
Validates every entry and logs any issues found.
=============================================================================
"""

import json
import os
import glob
import pandas as pd
import logging
from datetime import datetime

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/merge.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────────────────────
BATCHES_DIR  = "data/dataGenerator/raw_batches"
OUTPUT_DIR   = "data/raw"
RAW_CSV      = os.path.join(OUTPUT_DIR, "raw_feedback.csv")
ABSA_CSV     = os.path.join(OUTPUT_DIR, "absa_dataset.csv")

VALID_ASPECTS = {
    "Network Coverage", "Internet Speed", "Call Quality", "Customer Support",
    "Billing", "Recharge Plans", "Data Balance", "Roaming", "SIM Activation",
    "Mobile App Experience", "OTT Bundle Services", "Pricing", "Value for Money",
    "Data Validity", "5G Experience", "Network Outage", "Number Portability",
    "SMS Services", "Postpaid Plans", "Network Congestion",
    "International Calling", "Device Compatibility"
}

VALID_SENTIMENTS = {"Positive", "Negative", "Neutral"}

# ── Validation ────────────────────────────────────────────────────────────────
def validate_raw_entry(entry: dict, batch_file: str) -> bool:
    required_keys = {"feedback_id", "feedback_text", "aspects", "sentiments", "num_aspects"}
    if not required_keys.issubset(entry.keys()):
        log.warning(f"{batch_file}: Missing keys in raw entry {entry.get('feedback_id','?')}")
        return False
    aspects   = entry["aspects"].split("|")
    sentiments = entry["sentiments"].split("|")
    if len(aspects) != len(sentiments):
        log.warning(f"{batch_file}: Aspect/sentiment count mismatch in {entry['feedback_id']}")
        return False
    for a in aspects:
        if a.strip() not in VALID_ASPECTS:
            log.warning(f"{batch_file}: Invalid aspect '{a}' in {entry['feedback_id']}")
            return False
    for s in sentiments:
        if s.strip() not in VALID_SENTIMENTS:
            log.warning(f"{batch_file}: Invalid sentiment '{s}' in {entry['feedback_id']}")
            return False
    if not entry["feedback_text"] or len(str(entry["feedback_text"]).strip()) < 5:
        log.warning(f"{batch_file}: Empty feedback text in {entry['feedback_id']}")
        return False
    return True

def validate_absa_entry(entry: dict, batch_file: str) -> bool:
    required_keys = {"feedback_id", "feedback_text", "aspect", "sentiment"}
    if not required_keys.issubset(entry.keys()):
        log.warning(f"{batch_file}: Missing keys in absa entry {entry.get('feedback_id','?')}")
        return False
    if entry["aspect"].strip() not in VALID_ASPECTS:
        log.warning(f"{batch_file}: Invalid aspect '{entry['aspect']}' in {entry['feedback_id']}")
        return False
    if entry["sentiment"].strip() not in VALID_SENTIMENTS:
        log.warning(f"{batch_file}: Invalid sentiment '{entry['sentiment']}' in {entry['feedback_id']}")
        return False
    return True

# ── Main ──────────────────────────────────────────────────────────────────────
def merge():
    os.makedirs("logs", exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    batch_files = sorted(glob.glob(os.path.join(BATCHES_DIR, "batch_*.json")))

    if not batch_files:
        log.error(f"No batch files found in {BATCHES_DIR}. Check folder path.")
        return

    log.info(f"Found {len(batch_files)} batch files. Starting merge...")

    all_raw  = []
    all_absa = []
    seen_feedback_ids = set()

    total_raw_invalid  = 0
    total_absa_invalid = 0
    total_duplicates   = 0

    for batch_file in batch_files:
        fname = os.path.basename(batch_file)
        try:
            with open(batch_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            log.error(f"{fname}: JSON parse error — {e}. Skipping file.")
            continue
        except Exception as e:
            log.error(f"{fname}: Could not read file — {e}. Skipping file.")
            continue

        # Handle both formats:
        # Format A: {"raw_feedback": [...], "absa_dataset": [...]}  <- our format
        # Format B: [{"feedback_id":..., "feedback_text":..., "aspects":[...]}]  <- fallback
        if isinstance(data, dict) and "raw_feedback" in data and "absa_dataset" in data:
            raw_entries  = data["raw_feedback"]
            absa_entries = data["absa_dataset"]
        elif isinstance(data, list):
            # fallback: rebuild from list format
            log.warning(f"{fname}: Old list format detected. Rebuilding...")
            raw_entries, absa_entries = rebuild_from_list(data)
        else:
            log.error(f"{fname}: Unrecognized format. Skipping.")
            continue

        # Process raw entries
        for entry in raw_entries:
            fid = entry.get("feedback_id", "")
            if fid in seen_feedback_ids:
                total_duplicates += 1
                continue
            if validate_raw_entry(entry, fname):
                all_raw.append(entry)
                seen_feedback_ids.add(fid)
            else:
                total_raw_invalid += 1

        # Process absa entries
        for entry in absa_entries:
            if validate_absa_entry(entry, fname):
                all_absa.append(entry)
            else:
                total_absa_invalid += 1

        log.info(f"{fname}: raw={len(raw_entries)}, absa={len(absa_entries)}")

    # ── Build DataFrames ──────────────────────────────────────────────────────
    df_raw  = pd.DataFrame(all_raw)
    df_absa = pd.DataFrame(all_absa)

    # Clean whitespace
    df_raw["feedback_text"]  = df_raw["feedback_text"].str.strip()
    df_absa["feedback_text"] = df_absa["feedback_text"].str.strip()
    df_absa["aspect"]        = df_absa["aspect"].str.strip()
    df_absa["sentiment"]     = df_absa["sentiment"].str.strip()

    # ── Remove duplicate absa rows ────────────────────────────────────────────
    before = len(df_absa)
    df_absa = df_absa.drop_duplicates(subset=["feedback_id", "aspect"])
    after   = len(df_absa)
    if before != after:
        log.warning(f"Removed {before - after} duplicate absa rows.")

    # ── Save ──────────────────────────────────────────────────────────────────
    df_raw.to_csv(RAW_CSV,  index=False, encoding="utf-8")
    df_absa.to_csv(ABSA_CSV, index=False, encoding="utf-8")

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("MERGE COMPLETE — SUMMARY")
    log.info("=" * 60)
    log.info(f"Batch files processed     : {len(batch_files)}")
    log.info(f"Total feedbacks           : {len(df_raw)}")
    log.info(f"Total aspect rows         : {len(df_absa)}")
    log.info(f"Avg aspects per feedback  : {df_raw['num_aspects'].mean():.2f}")
    log.info(f"Duplicate feedbacks dropped: {total_duplicates}")
    log.info(f"Invalid raw entries dropped: {total_raw_invalid}")
    log.info(f"Invalid absa entries dropped: {total_absa_invalid}")
    log.info(f"\nSentiment distribution:")
    log.info(f"\n{df_absa['sentiment'].value_counts().to_string()}")
    log.info(f"\nAspect distribution:")
    log.info(f"\n{df_absa['aspect'].value_counts().to_string()}")
    log.info(f"\nSaved: {RAW_CSV}")
    log.info(f"Saved: {ABSA_CSV}")
    log.info("=" * 60)


def rebuild_from_list(data: list):
    """Fallback: rebuild raw and absa format from old list-style JSON."""
    raw_entries  = []
    absa_entries = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        fid  = entry.get("feedback_id", "")
        text = entry.get("feedback_text", "")
        aspects_list = entry.get("aspects", [])
        if not isinstance(aspects_list, list):
            continue
        aspect_names = [a.get("aspect","") for a in aspects_list]
        sentiments   = [a.get("sentiment","") for a in aspects_list]
        raw_entries.append({
            "feedback_id"  : fid,
            "feedback_text": text,
            "aspects"      : "|".join(aspect_names),
            "sentiments"   : "|".join(sentiments),
            "num_aspects"  : len(aspects_list)
        })
        for a in aspects_list:
            absa_entries.append({
                "feedback_id"  : fid,
                "feedback_text": text,
                "aspect"       : a.get("aspect",""),
                "sentiment"    : a.get("sentiment","")
            })
    return raw_entries, absa_entries


if __name__ == "__main__":
    merge()