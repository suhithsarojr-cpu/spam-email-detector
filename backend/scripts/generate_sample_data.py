"""Generate a small SYNTHETIC demo dataset (data/sample_emails.csv).

All text is invented from templates; it contains no real people or email data.
Because it is template-based, models score very well on it. Do not treat those
numbers as real-world performance. For meaningful results, train on a real dataset
(see README, "Using a real dataset").

Usage (from backend/):  python -m scripts.generate_sample_data
"""
import argparse
import csv
import random
from pathlib import Path

from app import config

NAMES = ["Alex", "Priya", "Jordan", "Sam", "Mei", "Chris", "Taylor", "Ravi", "Noor", "Jamie"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
TOPICS = ["budget review", "project kickoff", "design sync", "quarterly planning", "lab report", "team retro"]
PRODUCTS = ["watch", "gift card", "laptop", "phone", "vacation package", "tablet"]

HAM_SUBJECTS = [
    "Meeting moved to {day}", "Notes from the {topic}", "Question about the {topic}",
    "Lunch on {day}?", "Draft attached for review", "Re: {topic} agenda",
    "Assignment deadline reminder", "Your invoice from last month", "Can you review my pull request?",
    "Account setup for the new team member", "Urgent: server maintenance tonight",
]
HAM_BODY = [
    "Hi {n}, are you free to talk on {day}? I wanted to go over the {topic} before we finalize anything.",
    "Thanks for sending the draft. I added a few comments and the revised file is attached.",
    "Just a reminder that the {topic} is scheduled for {day} at 10 am in the usual room.",
    "Please find the meeting notes below. Let me know if I missed any action items.",
    "I will be out of office on {day}, so please send questions to {n} in the meantime.",
    "Could you share the latest numbers? Payment details for the vendor are in the shared folder.",
    "The offer letter has been signed by both parties and filed with the department.",
    "Happy to help. I have a free slot on {day} afternoon if you want to pair on this.",
    "The maintenance window is tonight from 11 pm. Please save your work and log out of the account.",
    "Great work on the {topic}. Let's catch up next week and plan the next steps.",
    "Here is the schedule for the week. Reply if you need to swap any sessions.",
]
SPAM_SUBJECTS = [
    "YOU HAVE WON a free {product}!!!", "Urgent: verify your account now", "Cheap meds, no prescription needed",
    "Congratulations! Claim your prize", "Earn $5000 per week from home", "Final notice: payment overdue",
    "Exclusive offer just for you", "Your loan is pre-approved", "Make money fast with crypto",
    "Meeting request: click to confirm", "Invoice #{num} attached - open immediately",
]
SPAM_BODY = [
    "Congratulations {n}! You have been selected to receive a free {product}. Click http://example.com/claim{num} now to claim your prize.",
    "Your account has been suspended. Verify your password immediately at http://example.net/login{num} to avoid permanent closure.",
    "Act now! Limited time offer: get cheap pills with no prescription. Order today and save 80%. Unsubscribe at http://example.org/u{num}.",
    "Earn extra income from home with no experience. Send your bank details to start receiving payments of $5000 weekly.",
    "You are pre-approved for a low interest loan. No credit check. Reply with your social security number to receive funds.",
    "FREE gift card waiting for you! Winner winner! Click here to redeem before the offer expires tonight.",
    "Invest in this new crypto coin and double your money in 24 hours. Guaranteed returns, risk free. Join now!",
    "Dear customer, we could not process your payment. Update your billing information at http://example.com/pay{num} to restore access.",
    "This is your final notice. Claim your {product} today. Buy now, discount ends soon. 100% satisfaction guaranteed.",
    "Hot singles in your area want to meet you. Click the link to view photos. Offer valid for a limited time only.",
]


def _fill(template: str, rng: random.Random) -> str:
    return template.format(n=rng.choice(NAMES), day=rng.choice(DAYS), topic=rng.choice(TOPICS),
                           product=rng.choice(PRODUCTS), num=rng.randint(100, 9999))


def build_rows(n_per_class: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for label, subjects, bodies in (("ham", HAM_SUBJECTS, HAM_BODY), ("spam", SPAM_SUBJECTS, SPAM_BODY)):
        for _ in range(n_per_class):
            body = " ".join(_fill(b, rng) for b in rng.sample(bodies, rng.randint(1, 3)))
            rows.append({"subject": _fill(rng.choice(subjects), rng), "body": body, "label": label})
    # a little label noise (3%) so evaluation is not trivially perfect
    for r in rng.sample(rows, int(len(rows) * 0.03)):
        r["label"] = "spam" if r["label"] == "ham" else "ham"
    rng.shuffle(rows)
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-per-class", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=config.DATA_DIR / "sample_emails.csv")
    a = p.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    rows = build_rows(a.n_per_class, a.seed)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["subject", "body", "label"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} synthetic rows to {a.out}")


if __name__ == "__main__":
    main()
