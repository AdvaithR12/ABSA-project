"""
Generate augmentation training data:
1. Contrastive examples (positive X but negative Y)
2. Positive-surprise patterns (to fix BERT's Neutral over-prediction)
3. Implicit sentiment patterns (comparative, expectation-based)

Output: data/dataGenerator/raw_batches/batch_augment_contrastive.json
"""

import json
import os

# Starting feedback ID (after existing 6750 entries)
START_ID = 6751
current_id = START_ID

def make_id():
    global current_id
    fid = f"fb_{current_id:05d}"
    current_id += 1
    return fid


# ============================================================================
# SECTION 1: CONTRASTIVE EXAMPLES (positive X but negative Y)
# ============================================================================

contrastive_data = [
    # Network Coverage + Internet Speed
    ("The signal bars show full strength, but webpages take forever to load.",
     [("Network Coverage", "Positive"), ("Internet Speed", "Negative")]),
    ("Internet speed is blazing fast, yet the signal keeps dropping in my apartment.",
     [("Internet Speed", "Positive"), ("Network Coverage", "Negative")]),
    ("Coverage is spotty indoors, although when it works the download speeds are impressive.",
     [("Network Coverage", "Negative"), ("Internet Speed", "Positive")]),
    ("The network reaches everywhere now, but the browsing speed has gotten worse.",
     [("Network Coverage", "Positive"), ("Internet Speed", "Negative")]),
    ("Full bars on my phone, yet I cannot load a single webpage.",
     [("Network Coverage", "Positive"), ("Internet Speed", "Negative")]),
    ("Downloads finish in seconds, but the signal disappears every few minutes.",
     [("Internet Speed", "Positive"), ("Network Coverage", "Negative")]),

    # Call Quality + Network Coverage
    ("Call clarity is excellent, but I lose connection whenever I move between rooms.",
     [("Call Quality", "Positive"), ("Network Coverage", "Negative")]),
    ("The signal is strong everywhere, however voice calls echo badly.",
     [("Network Coverage", "Positive"), ("Call Quality", "Negative")]),
    ("Coverage is unreliable, though when connected the call quality is surprisingly good.",
     [("Network Coverage", "Negative"), ("Call Quality", "Positive")]),
    ("Crystal clear voice quality, yet the call drops if I step outside.",
     [("Call Quality", "Positive"), ("Network Coverage", "Negative")]),
    ("I have signal everywhere, but every conversation sounds like it's underwater.",
     [("Network Coverage", "Positive"), ("Call Quality", "Negative")]),

    # Customer Support + Billing
    ("The support agent was helpful, but my billing issue still hasn't been resolved.",
     [("Customer Support", "Positive"), ("Billing", "Negative")]),
    ("My bill was corrected quickly, although the support experience was frustrating.",
     [("Billing", "Positive"), ("Customer Support", "Negative")]),
    ("Support responded within minutes, yet they charged me for the service call.",
     [("Customer Support", "Positive"), ("Billing", "Negative")]),
    ("The billing is always accurate, but getting someone on the phone takes ages.",
     [("Billing", "Positive"), ("Customer Support", "Negative")]),
    ("The agent was knowledgeable, however the extra charge on my bill remains unexplained.",
     [("Customer Support", "Positive"), ("Billing", "Negative")]),
    ("My invoice is always correct, yet the support staff gives conflicting information.",
     [("Billing", "Positive"), ("Customer Support", "Negative")]),

    # Pricing + Value for Money
    ("The price is low, but you get what you pay for with very limited features.",
     [("Pricing", "Positive"), ("Value for Money", "Negative")]),
    ("It is expensive, yet the service quality makes it worthwhile.",
     [("Pricing", "Negative"), ("Value for Money", "Positive")]),
    ("The monthly cost is reasonable, however the data cap makes it poor value.",
     [("Pricing", "Positive"), ("Value for Money", "Negative")]),
    ("The package costs a lot, but includes everything I need and more.",
     [("Pricing", "Negative"), ("Value for Money", "Positive")]),
    ("Affordable plans available, but none of them offer enough data to be useful.",
     [("Pricing", "Positive"), ("Value for Money", "Negative")]),

    # Roaming + International Calling
    ("Roaming data worked perfectly abroad, but international calls were terrible.",
     [("Roaming", "Positive"), ("International Calling", "Negative")]),
    ("International calls are crystal clear, although the roaming charges are shocking.",
     [("International Calling", "Positive"), ("Roaming", "Negative")]),
    ("I could browse freely while traveling, yet calling home was nearly impossible.",
     [("Roaming", "Positive"), ("International Calling", "Negative")]),
    ("Calling overseas is cheap and clear, but data roaming costs a fortune.",
     [("International Calling", "Positive"), ("Roaming", "Negative")]),

    # Mobile App + Billing
    ("The app interface is beautiful, but it always fails when I try to pay my bill.",
     [("Mobile App Experience", "Positive"), ("Billing", "Negative")]),
    ("Paying through the app is seamless, although the app itself crashes frequently.",
     [("Billing", "Positive"), ("Mobile App Experience", "Negative")]),
    ("The app loads instantly, however the billing section shows incorrect amounts.",
     [("Mobile App Experience", "Positive"), ("Billing", "Negative")]),
    ("Bill payment works flawlessly in the app, yet the rest of the app is painfully slow.",
     [("Billing", "Positive"), ("Mobile App Experience", "Negative")]),

    # SIM Activation + Customer Support
    ("My SIM was activated in minutes, but the support staff was rude throughout.",
     [("SIM Activation", "Positive"), ("Customer Support", "Negative")]),
    ("The support team was incredibly patient, yet my activation took over a week.",
     [("Customer Support", "Positive"), ("SIM Activation", "Negative")]),
    ("Activation was instant, however nobody told me about the required documents beforehand.",
     [("SIM Activation", "Positive"), ("Customer Support", "Negative")]),

    # Internet Speed + Network Congestion
    ("Speed is great during the day, but becomes unusable during evening peak hours.",
     [("Internet Speed", "Positive"), ("Network Congestion", "Negative")]),
    ("The network handles congestion well, although base speeds are below average.",
     [("Network Congestion", "Positive"), ("Internet Speed", "Negative")]),
    ("Morning downloads are fast, yet everything crawls to a halt after 6 PM.",
     [("Internet Speed", "Positive"), ("Network Congestion", "Negative")]),
    ("Peak hour performance improved, but off-peak speeds actually got slower.",
     [("Network Congestion", "Positive"), ("Internet Speed", "Negative")]),

    # OTT Bundle + Pricing
    ("The streaming bundle is excellent, but adds too much to the monthly cost.",
     [("OTT Bundle Services", "Positive"), ("Pricing", "Negative")]),
    ("The price includes streaming, however the content library is very limited.",
     [("Pricing", "Positive"), ("OTT Bundle Services", "Negative")]),
    ("The OTT content is world-class, yet it makes the plan unaffordable.",
     [("OTT Bundle Services", "Positive"), ("Pricing", "Negative")]),
    ("Cheap streaming included, but the shows are outdated and the selection is tiny.",
     [("Pricing", "Positive"), ("OTT Bundle Services", "Negative")]),

    # Data Balance + Data Validity
    ("The data allowance is generous, but expires before I can use half of it.",
     [("Data Balance", "Positive"), ("Data Validity", "Negative")]),
    ("The validity period is long, although the data cap is too restrictive.",
     [("Data Validity", "Positive"), ("Data Balance", "Negative")]),
    ("I get plenty of data, yet the short validity means most goes to waste.",
     [("Data Balance", "Positive"), ("Data Validity", "Negative")]),
    ("The 90-day validity is great, but 2GB for three months is laughable.",
     [("Data Validity", "Positive"), ("Data Balance", "Negative")]),

    # Recharge Plans + Pricing
    ("The recharge options are flexible, but every good plan is overpriced.",
     [("Recharge Plans", "Positive"), ("Pricing", "Negative")]),
    ("Prices are competitive, however the recharge process is confusing.",
     [("Pricing", "Positive"), ("Recharge Plans", "Negative")]),
    ("So many recharge options to choose from, yet all the affordable ones lack data.",
     [("Recharge Plans", "Positive"), ("Pricing", "Negative")]),

    # 5G + Device Compatibility
    ("5G speeds are incredible, but my phone heats up excessively while using it.",
     [("5G Experience", "Positive"), ("Device Compatibility", "Negative")]),
    ("My device handles the network well, although 5G coverage barely exists here.",
     [("Device Compatibility", "Positive"), ("5G Experience", "Negative")]),
    ("The 5G experience is phenomenal, yet half the phones in the market are incompatible.",
     [("5G Experience", "Positive"), ("Device Compatibility", "Negative")]),

    # Network Outage + Customer Support
    ("The outage was brief, but nobody informed us it was happening.",
     [("Network Outage", "Neutral"), ("Customer Support", "Negative")]),
    ("Support kept us updated throughout, even though the outage lasted all day.",
     [("Customer Support", "Positive"), ("Network Outage", "Negative")]),
    ("The outage disrupted my work completely, though support resolved it quickly.",
     [("Network Outage", "Negative"), ("Customer Support", "Positive")]),
    ("They fixed the outage fast, but the lack of advance warning was unacceptable.",
     [("Network Outage", "Positive"), ("Customer Support", "Negative")]),

    # Number Portability + SIM Activation
    ("The porting process was smooth, but activating the new SIM took days.",
     [("Number Portability", "Positive"), ("SIM Activation", "Negative")]),
    ("My SIM activated instantly, however the number transfer failed twice.",
     [("SIM Activation", "Positive"), ("Number Portability", "Negative")]),
    ("Number porting completed in hours, yet the new SIM had no data for two days.",
     [("Number Portability", "Positive"), ("SIM Activation", "Negative")]),

    # SMS + Mobile App
    ("Text messages are always reliable, but the app never sends notifications properly.",
     [("SMS Services", "Positive"), ("Mobile App Experience", "Negative")]),
    ("The app notifications work perfectly, although SMS delivery has been unreliable lately.",
     [("Mobile App Experience", "Positive"), ("SMS Services", "Negative")]),

    # Postpaid + Billing
    ("The postpaid plan has great benefits, but the billing cycle is confusing.",
     [("Postpaid Plans", "Positive"), ("Billing", "Negative")]),
    ("Bills arrive on time and are clear, however the postpaid plan itself is limited.",
     [("Billing", "Positive"), ("Postpaid Plans", "Negative")]),
    ("The postpaid perks are amazing, yet my bill is different every month.",
     [("Postpaid Plans", "Positive"), ("Billing", "Negative")]),

    # Cross-category complex
    ("Network coverage improved massively, but the price increase was not justified.",
     [("Network Coverage", "Positive"), ("Pricing", "Negative")]),
    ("The plan is affordable, yet the network barely works in rural areas.",
     [("Pricing", "Positive"), ("Network Coverage", "Negative")]),
    ("5G is available in my city, but the data gets consumed three times faster.",
     [("5G Experience", "Positive"), ("Data Balance", "Negative")]),
    ("The mobile app tracks usage well, however it drains my battery significantly.",
     [("Mobile App Experience", "Positive"), ("Device Compatibility", "Negative")]),
    ("International calls are cheap, but the audio quality is terrible.",
     [("International Calling", "Positive"), ("Call Quality", "Negative")]),
    ("The recharge was easy, but my data balance didn't update for hours.",
     [("Recharge Plans", "Positive"), ("Data Balance", "Negative")]),
    ("Streaming works without buffering, yet the app crashes during live content.",
     [("OTT Bundle Services", "Positive"), ("Mobile App Experience", "Negative")]),
    ("The postpaid plan includes unlimited calls, but international minutes cost extra.",
     [("Postpaid Plans", "Positive"), ("International Calling", "Negative")]),
    ("Data validity is generous at 90 days, but the speeds throttle after 50 percent usage.",
     [("Data Validity", "Positive"), ("Internet Speed", "Negative")]),
    ("The network handles peak hours fine now, although regular speeds remain disappointing.",
     [("Network Congestion", "Positive"), ("Internet Speed", "Negative")]),
]


# ============================================================================
# SECTION 2: POSITIVE-SURPRISE PATTERNS (fix BERT's Neutral over-prediction)
# ============================================================================

positive_surprise_data = [
    # "faster/better/sooner than expected" patterns
    ("The activation completed much faster than I expected.", [("SIM Activation", "Positive")]),
    ("Number porting was surprisingly quick, done in under an hour.", [("Number Portability", "Positive")]),
    ("The support team resolved my issue faster than anticipated.", [("Customer Support", "Positive")]),
    ("Internet speeds exceeded my expectations completely.", [("Internet Speed", "Positive")]),
    ("The transfer process finished sooner than promised.", [("Number Portability", "Positive")]),
    ("My complaint was handled better than I could have hoped.", [("Customer Support", "Positive")]),
    ("The outage was restored much quicker than the estimated time.", [("Network Outage", "Positive")]),
    ("Call quality turned out to be much better than I anticipated.", [("Call Quality", "Positive")]),
    ("The roaming experience was smoother than I ever expected.", [("Roaming", "Positive")]),
    ("Data speeds are faster than what was advertised.", [("Internet Speed", "Positive")]),
    ("The app loaded quicker than I thought it would.", [("Mobile App Experience", "Positive")]),
    ("My bill was lower than expected this month.", [("Billing", "Positive")]),
    ("The 5G coverage expanded faster than they promised.", [("5G Experience", "Positive")]),
    ("Recharge benefits turned out better than described.", [("Recharge Plans", "Positive")]),
    ("The streaming quality is better than what competitors offer.", [("OTT Bundle Services", "Positive")]),

    # "surprisingly/effortless/seamless" patterns
    ("The entire porting process was surprisingly effortless.", [("Number Portability", "Positive")]),
    ("Activation was seamless and took no effort at all.", [("SIM Activation", "Positive")]),
    ("The support experience was surprisingly pleasant.", [("Customer Support", "Positive")]),
    ("Roaming connectivity was effortlessly maintained across three countries.", [("Roaming", "Positive")]),
    ("The bill payment process is surprisingly straightforward.", [("Billing", "Positive")]),
    ("The network transition to 5G was seamless for me.", [("5G Experience", "Positive")]),
    ("Surprisingly, the app works flawlessly even on my older phone.", [("Mobile App Experience", "Positive")]),
    ("The recharge process was effortless through the app.", [("Recharge Plans", "Positive")]),
    ("International call setup was surprisingly easy.", [("International Calling", "Positive")]),
    ("Switching plans was seamless with no downtime.", [("Postpaid Plans", "Positive")]),

    # "impressed/pleased/delighted" patterns
    ("I am genuinely impressed by how well the network performs indoors.", [("Network Coverage", "Positive")]),
    ("Pleasantly surprised by the call clarity on this network.", [("Call Quality", "Positive")]),
    ("Delighted with how the support team handled my complex issue.", [("Customer Support", "Positive")]),
    ("I was impressed that my data lasted the entire month.", [("Data Balance", "Positive")]),
    ("The value I get from this plan genuinely impresses me.", [("Value for Money", "Positive")]),
    ("Impressed by how the network handled the concert crowd.", [("Network Congestion", "Positive")]),
    ("Pleasantly surprised that the streaming bundle includes premium content.", [("OTT Bundle Services", "Positive")]),
    ("I am delighted with how quickly messages get delivered now.", [("SMS Services", "Positive")]),
    ("Impressed that my phone works perfectly with their latest features.", [("Device Compatibility", "Positive")]),

    # "professionally/efficiently handled" patterns
    ("The provider handled my number transfer professionally and quickly.", [("Number Portability", "Positive")]),
    ("My complaint was resolved efficiently in a single call.", [("Customer Support", "Positive")]),
    ("The billing dispute was handled professionally with a full refund.", [("Billing", "Positive")]),
    ("The outage communication was managed professionally.", [("Network Outage", "Positive")]),
    ("SIM replacement was handled efficiently at the store.", [("SIM Activation", "Positive")]),
    ("The international calling setup was handled professionally by support.", [("International Calling", "Positive")]),

    # "exceeded/surpassed" patterns
    ("Download speeds consistently exceed what was promised in the plan.", [("Internet Speed", "Positive")]),
    ("The network coverage has surpassed all my initial doubts.", [("Network Coverage", "Positive")]),
    ("Service quality exceeded what I was paying for.", [("Value for Money", "Positive")]),
    ("The plan benefits far exceed what competitors offer at this price.", [("Pricing", "Positive")]),
    ("OTT content quality exceeded my expectations from a telecom provider.", [("OTT Bundle Services", "Positive")]),
    ("5G performance surpassed everything I experienced on the old network.", [("5G Experience", "Positive")]),
    ("The data allowance exceeds what most people would ever need.", [("Data Balance", "Positive")]),

    # "no issues/worked perfectly/flawless" patterns
    ("Number porting worked without a single issue.", [("Number Portability", "Positive")]),
    ("The roaming service worked flawlessly in every country I visited.", [("Roaming", "Positive")]),
    ("Zero issues with the activation process from start to finish.", [("SIM Activation", "Positive")]),
    ("The billing has been accurate without fail for over a year.", [("Billing", "Positive")]),
    ("Not a single dropped call in three months of use.", [("Call Quality", "Positive")]),
    ("The app runs perfectly on every device I have tried.", [("Mobile App Experience", "Positive")]),
    ("SMS delivery has been flawless since I joined this provider.", [("SMS Services", "Positive")]),
    ("The postpaid plan works exactly as advertised with no hidden costs.", [("Postpaid Plans", "Positive")]),
]


# ============================================================================
# SECTION 3: IMPLICIT NEGATIVE PATTERNS
# ============================================================================

implicit_negative_data = [
    # Comparative negatives
    ("The plan costs more than what competitors offer for the same features.", [("Pricing", "Negative")]),
    ("Coverage is worse than my previous provider in this area.", [("Network Coverage", "Negative")]),
    ("Speeds are slower than what was advertised.", [("Internet Speed", "Negative")]),
    ("The support quality has declined compared to last year.", [("Customer Support", "Negative")]),
    ("My data runs out faster than it used to on the old plan.", [("Data Balance", "Negative")]),
    ("The app is less responsive than the previous version.", [("Mobile App Experience", "Negative")]),
    ("Call quality is worse in this area compared to the city.", [("Call Quality", "Negative")]),
    ("The validity period is shorter than competing plans.", [("Data Validity", "Negative")]),
    ("Roaming rates are higher than industry average.", [("Roaming", "Negative")]),
    ("The recharge options are more confusing than before.", [("Recharge Plans", "Negative")]),

    # "sooner/earlier/faster than expected" (negative context)
    ("My data quota depleted far sooner than the billing cycle.", [("Data Balance", "Negative")]),
    ("The validity expired earlier than I was told it would.", [("Data Validity", "Negative")]),
    ("The call dropped sooner than I could finish my sentence.", [("Call Quality", "Negative")]),
    ("The network went down again sooner than the last time.", [("Network Outage", "Negative")]),
    ("My balance hit zero faster than I could track it.", [("Data Balance", "Negative")]),

    # Implicit dissatisfaction (no explicit negative words)
    ("I have to restart my phone three times a day to get signal.", [("Network Coverage", "Negative")]),
    ("Every time I call support I get a different answer.", [("Customer Support", "Negative")]),
    ("I need to stand by the window to make a phone call.", [("Network Coverage", "Negative")]),
    ("The app requires me to log in fresh every single time.", [("Mobile App Experience", "Negative")]),
    ("My messages arrive hours after they were sent.", [("SMS Services", "Negative")]),
    ("I have to recharge twice a month because data runs out so fast.", [("Data Balance", "Negative")]),
    ("The bill shows items I never subscribed to.", [("Billing", "Negative")]),
    ("They keep changing the plan terms without notice.", [("Postpaid Plans", "Negative")]),
    ("I wait 20 minutes on hold every time I call support.", [("Customer Support", "Negative")]),
    ("The OTT content buffers even on a 100mbps wifi connection.", [("OTT Bundle Services", "Negative")]),
]


# ============================================================================
# BUILD OUTPUT
# ============================================================================

raw_feedback = []
absa_dataset = []

all_data = contrastive_data + positive_surprise_data + implicit_negative_data

for text, aspect_sentiments in all_data:
    fid = make_id()
    aspects_list = [a for a, s in aspect_sentiments]
    sentiments_list = [s for a, s in aspect_sentiments]
    
    raw_feedback.append({
        "feedback_id": fid,
        "feedback_text": text,
        "aspects": "|".join(aspects_list),
        "sentiments": "|".join(sentiments_list),
        "num_aspects": len(aspect_sentiments)
    })
    
    for aspect, sentiment in aspect_sentiments:
        absa_dataset.append({
            "feedback_id": fid,
            "feedback_text": text,
            "aspect": aspect,
            "sentiment": sentiment
        })

output = {
    "raw_feedback": raw_feedback,
    "absa_dataset": absa_dataset
}

# Save
output_path = os.path.join(os.path.dirname(__file__), "raw_batches", "batch_augment_contrastive.json")
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"Generated {len(raw_feedback)} feedback entries")
print(f"Generated {len(absa_dataset)} aspect-level rows")
print(f"Saved to: {output_path}")
print(f"\nBreakdown:")
print(f"  Contrastive examples: {len(contrastive_data)} feedbacks")
print(f"  Positive-surprise examples: {len(positive_surprise_data)} feedbacks")
print(f"  Implicit-negative examples: {len(implicit_negative_data)} feedbacks")
print(f"  Total: {len(all_data)} feedbacks, {len(absa_dataset)} aspect rows")
