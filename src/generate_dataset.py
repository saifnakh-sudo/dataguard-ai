"""
generate_dataset.py — Creates a synthetic spam/ham dataset for DataGuard AI.
Run this once on Day 1 before anything else.
"""
import pandas as pd
import numpy as np
import os

rng = np.random.default_rng(42)

HAM = [
    "Hey, are you coming to the meeting tomorrow?",
    "Can you pick up milk on the way home?",
    "I loved the presentation you gave today!",
    "Let me know when you are free to catch up.",
    "The report is ready, I will send it over now.",
    "Happy birthday! Hope you have an amazing day.",
    "Thank you so much for your help last week.",
    "I will be there in 10 minutes, just leaving now.",
    "Do you want to grab lunch later today?",
    "The weather looks great this weekend, perfect for a hike.",
    "Just checking in to see how things are going.",
    "Great job on the project, everyone is impressed.",
    "I sent you the files, let me know if they arrived.",
    "Can we reschedule to Thursday instead of Wednesday?",
    "I will finish the analysis tonight and share it tomorrow.",
    "Your feedback was really helpful, thanks for that.",
    "The kids are doing well, they say hello.",
    "Let me know if you need anything else from me.",
    "I saw your message, I will reply properly later tonight.",
    "The team meeting went well, big progress today.",
    "Is Saturday still good for you? We can meet at noon.",
    "I booked the restaurant for 7pm, see you there.",
    "No worries, take your time, there is no rush.",
    "The package arrived safely, thank you for sending it.",
    "I have finished reviewing the document and it looks good.",
    "How was your trip? I heard it went really well.",
    "I am on my way, stuck in traffic but should be there soon.",
    "Good luck with the exam tomorrow, you will do great.",
    "We really enjoyed having you over for dinner last week.",
    "The deadline has been extended to next Friday.",
]

SPAM = [
    "WINNER! You have been selected for a FREE prize. Call now!",
    "Congratulations! Claim your $1000 gift card today. Limited time!",
    "Urgent: Your account has been suspended. Click here to verify.",
    "You have won a lottery. Send your details to claim your reward.",
    "Free ringtones! Text WIN to 88888. No subscription needed!",
    "ALERT: Unusual activity detected. Reset your password immediately.",
    "Make money fast! Work from home and earn $500 a day guaranteed.",
    "Your mobile number has been chosen for a special offer. Reply YES.",
    "Buy now and save 90%! Limited stock available, act immediately.",
    "You are pre-approved for a loan of $50,000. Apply instantly online.",
    "Double your income with this one simple trick. Click here to learn.",
    "Hot singles in your area want to meet you. Join free today!",
    "Final notice: Your subscription will expire. Renew now to avoid loss.",
    "Free iPhone giveaway! Enter your details to claim your new phone.",
    "Your parcel could not be delivered. Pay $2.99 to reschedule.",
    "SPECIAL OFFER: Lose 20 pounds in 2 weeks with this amazing pill.",
    "Investment opportunity: 300% returns guaranteed. Invest now!",
    "You have been randomly selected for our VIP rewards program.",
    "URGENT: Act now or lose your benefits. Click the link below.",
    "Earn cash by completing simple surveys. Sign up for free today.",
]

# Expand to ~600 rows
ham_texts = [HAM[i % len(HAM)] + f" [{rng.integers(1000,9999)}]" for i in range(400)]
spam_texts = [SPAM[i % len(SPAM)] + f" [{rng.integers(1000,9999)}]" for i in range(200)]

texts = ham_texts + spam_texts
labels = [0] * 400 + [1] * 200

df = pd.DataFrame({'text': texts, 'label': labels})
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

os.makedirs('data', exist_ok=True)
df.to_csv('data/raw.csv', index=False)
print(f"Generated {len(df)} rows: {labels.count(0)} ham (0), {labels.count(1)} spam (1)")
print("Saved to data/raw.csv")
