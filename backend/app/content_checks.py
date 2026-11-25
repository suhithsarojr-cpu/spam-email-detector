"""Transparent rule-based content checks that complement the ML model.
Heuristics only: they flag common spam-like patterns, not guaranteed filter behaviour."""
import re

PHRASES = ["free gift", "act now", "limited time", "click here", "you have won", "winner",
           "guaranteed", "no prescription", "risk free", "double your money", "100% free",
           "claim your prize", "verify your account", "buy now", "earn extra", "pre-approved"]
URGENCY = ["urgent", "immediately", "final notice", "expires", "last chance", "asap"]
SHORTENERS = ("bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd")
URL_RE = re.compile(r"https?://[^\s<>\"']+", re.I)


def _f(id, label, points, detail_bad, detail_ok):
    sev = "ok" if points == 0 else ("warn" if points < 2 else "bad")
    return {"id": id, "label": label, "severity": sev, "points": points,
            "detail": detail_ok if points == 0 else detail_bad}


def analyze(subject: str, body: str) -> dict:
    text, low = f"{subject}\n{body}", f"{subject}\n{body}".lower()
    letters = [c for c in subject if c.isalpha()]
    caps = len(letters) >= 8 and sum(c.isupper() for c in letters) / len(letters) > 0.6
    excl = text.count("!")
    phrases = [p for p in PHRASES if p in low]
    urgent = [w for w in URGENCY if w in low]
    urls = URL_RE.findall(text)
    short = [u for u in urls if any(d in u.lower() for d in SHORTENERS)]
    ip_url = [u for u in urls if re.match(r"https?://\d{1,3}(\.\d{1,3}){3}", u)]
    money = len(re.findall(r"[$€£]\s?\d", text))
    findings = [
        _f("caps", "Subject in capitals", 2 if caps else 0, "Most of the subject is uppercase.", "Subject capitalisation looks normal."),
        _f("exclaim", "Exclamation marks", 1.5 if excl >= 3 else 0, f"{excl} exclamation marks found.", "No excessive punctuation."),
        _f("phrases", "Spam-style phrases", min(3, 1.0 * len(phrases)), f"Found: {', '.join(phrases[:4])}.", "No common spam phrases found."),
        _f("urgency", "Urgency wording", 1 if urgent else 0, f"Urgency words: {', '.join(urgent[:3])}.", "No pressure/urgency wording."),
        _f("links", "Number of links", 1.5 if len(urls) > 3 else 0, f"{len(urls)} links in the message.", "Link count is reasonable."),
        _f("shorteners", "Shortened or IP-based links", 2 if (short or ip_url) else 0, "Contains URL shorteners or raw IP links.", "No shortened or IP-based links."),
        _f("money", "Money amounts", 1 if money >= 2 else 0, "Several currency amounts mentioned.", "No heavy money language."),
    ]
    score = round(min(10.0, sum(f["points"] for f in findings)), 1)
    return {"findings": findings, "rule_score": score}


def combine(p_spam: float, rule_score: float) -> tuple[int, str]:
    """Risk 0-100 = 70% ML spam probability + 30% rule score. Documented in README."""
    risk = round(100 * (0.7 * p_spam + 0.3 * rule_score / 10))
    verdict = "Likely spam" if risk >= 60 else "Suspicious" if risk >= 35 else "Looks fine"
    return risk, verdict
