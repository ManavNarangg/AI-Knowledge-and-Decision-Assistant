"""System 1 model #1 - Question-type Classification.

A multinomial Naive Bayes classifier over word unigrams/bigrams, trained at startup on a small
labelled seed set. It runs in microseconds, is deterministic, returns a full probability
distribution, and is used by the application to ROUTE the question: only questions that need
structured decisions pay for the LLM decomposition call.

The seed set is domain-neutral (education, HR, finance, IT) so it works across tenants.
"""
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

LABELS = ("factual", "policy", "procedural", "decision", "comparison")

TRAINING_DATA: dict[str, list[str]] = {
    "factual": [
        "What is the maximum amount of the award?",
        "When is the application deadline?",
        "How many days of annual leave do employees get?",
        "What is the minimum GPA listed for the scholarship?",
        "Who is the head of the finance office?",
        "What are the office opening hours?",
        "How much is the daily meal allowance?",
        "What is the value of the grant per semester?",
        "Where is the student services office located?",
        "How long is the probation period?",
        "What is the email address for HR support?",
        "How many credits are required per semester?",
        "What is the mileage reimbursement rate?",
        "What date does the fall semester start?",
    ],
    "policy": [
        "What is the policy on late submissions?",
        "What are the rules for working from home?",
        "Is it allowed to carry over unused leave to next year?",
        "What happens if a student fails a course?",
        "Are employees allowed to expense alcohol?",
        "What does the policy say about attendance?",
        "Can a scholarship be revoked?",
        "What is the code of conduct regarding gifts?",
        "Is overtime paid for salaried staff?",
        "What are the conditions for keeping the scholarship?",
        "What is the refund policy for cancelled trips?",
        "Are pets permitted in the office?",
        "What is the rule about combining two awards?",
        "What restrictions apply to international travel?",
    ],
    "procedural": [
        "How do I apply for the scholarship?",
        "What are the steps to submit an expense claim?",
        "How can I request parental leave?",
        "What documents do I need to submit with my application?",
        "Who do I contact to appeal a decision?",
        "How do I reset my password?",
        "What is the process for requesting remote work?",
        "How do I report a sick day?",
        "What should I do if my claim is rejected?",
        "Where do I send the signed form?",
        "How do I renew my scholarship for next year?",
        "What is the procedure to change my course?",
        "How do I book business travel?",
        "Walk me through filing a grievance.",
    ],
    "decision": [
        "Am I eligible for the merit scholarship with a GPA of 3.6 in my second year?",
        "I have worked here for 8 months and my rating is 4, can I work remotely?",
        "Can I get the need-based grant if my family income is 40000?",
        "I am a third year student with a sports scholarship, do I qualify for the merit award?",
        "Given that I joined 2 years ago, am I entitled to parental leave?",
        "My GPA is 3.2 and I am a first year, will I qualify?",
        "Do I qualify for reimbursement if I spent 300 on a hotel without a receipt?",
        "I have 5 years of service, how many leave days am I entitled to?",
        "Would I be approved for remote work if my role requires on-site equipment?",
        "Which scholarship category applies to me as a part-time student with 3.8 GPA?",
        "Is my 90 dollar dinner claim within the limit?",
        "Can I carry over 10 days if I have 3 years of tenure?",
        "I failed one course last term, can I still keep my scholarship?",
        "Am I eligible given my situation: full-time, 70 percent attendance, GPA 3.9?",
    ],
    "comparison": [
        "What is the difference between the merit and need-based scholarships?",
        "Compare sick leave and annual leave.",
        "Which pays more, the sports scholarship or the merit scholarship?",
        "How does hybrid work differ from fully remote work?",
        "What are the differences between economy and business class travel rules?",
        "Is the research grant better than the teaching assistantship?",
        "Merit scholarship vs need-based grant: which has stricter requirements?",
        "How do the rules for interns compare to full-time staff?",
        "Contrast the undergraduate and graduate application processes.",
        "Which option is more flexible, compressed hours or remote work?",
        "What distinguishes a warning from probation?",
        "Compare the reimbursement limits for domestic and international trips.",
    ],
}

TOKEN_RE = re.compile(r"[a-z]+|\d+(?:\.\d+)?")


def features(text: str) -> list[str]:
    words = ["<num>" if w[0].isdigit() else w for w in TOKEN_RE.findall(text.lower())]
    return words + [f"{a}_{b}" for a, b in zip(words, words[1:])]


@dataclass
class Classification:
    label: str
    confidence: float
    probabilities: dict[str, float]


class QuestionClassifier:
    def __init__(self, data: dict[str, list[str]] = TRAINING_DATA, alpha: float = 0.5):
        self.alpha = alpha
        self.counts: dict[str, Counter] = defaultdict(Counter)
        self.priors: dict[str, float] = {}
        total = sum(len(v) for v in data.values())
        for label, examples in data.items():
            self.priors[label] = math.log(len(examples) / total)
            for ex in examples:
                self.counts[label].update(features(ex))
        self.vocab = set().union(*self.counts.values())
        self.totals = {label: sum(c.values()) for label, c in self.counts.items()}

    def predict(self, text: str) -> Classification:
        feats = [f for f in features(text) if f in self.vocab]
        v = len(self.vocab)
        log_scores = {
            label: self.priors[label] + sum(
                math.log((self.counts[label][f] + self.alpha) / (self.totals[label] + self.alpha * v))
                for f in feats)
            for label in self.priors
        }
        peak = max(log_scores.values())
        exp = {k: math.exp(s - peak) for k, s in log_scores.items()}
        z = sum(exp.values())
        probs = {k: round(v / z, 4) for k, v in sorted(exp.items(), key=lambda x: -x[1])}
        label = next(iter(probs))
        return Classification(label, probs[label], probs)


classifier = QuestionClassifier()
