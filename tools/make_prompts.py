"""
tools/make_prompts.py — generate a prompts.jsonl of realistic long-in / short-out jobs.

Fifty prompts across the five job types the night would actually run (denial-letter
classification, AP invoice extraction, code-audit verdicts, acquisition scoring, call
transcript summaries). Documents are synthetic but shaped like the real thing: several
hundred tokens in, a short structured answer out. Deterministic (seeded).

    python tools/make_prompts.py [--n 50] [--out prompts.jsonl]
"""
from __future__ import annotations

import argparse
import json
import random

R = random.Random(20260901)

PAYERS = ["Aetna", "UnitedHealthcare", "Cigna", "Humana", "BCBS of Michigan", "Anthem", "Molina", "Priority Health"]
CPT = [("97110", "therapeutic exercise"), ("97140", "manual therapy"), ("99214", "office visit, moderate complexity"),
       ("20610", "large joint injection"), ("72148", "MRI lumbar spine w/o contrast"), ("93000", "ECG with interpretation"),
       ("97530", "therapeutic activities"), ("99285", "ED visit, high complexity")]
DENIALS = [
    ("CO-50", "not medically necessary per plan guidelines; documentation does not support frequency of {n} visits"),
    ("CO-197", "precertification/authorization absent for {desc}"),
    ("CO-97", "payment adjusted because the benefit for this service is included in the payment for another service already adjudicated"),
    ("CO-16", "claim lacks information needed for adjudication: missing modifier for {desc}"),
    ("CO-4", "procedure code {code} is inconsistent with the modifier used"),
    ("PR-204", "service not covered under the patient's current benefit plan"),
    ("CO-29", "time limit for filing has expired; claim received {days} days after date of service"),
    ("CO-151", "payment adjusted because the payer deems the information submitted does not support this many services"),
]

VENDORS = ["ACME Industrial Supply", "Northwind Electrical", "Great Lakes HVAC Parts", "Meridian Office Systems",
           "Blue Ridge Freight", "Cascade Chemical", "Summit Steel & Fastener", "Harbor Logistics LLC"]
ITEMS = ["3/4in copper fitting, 90deg", "MERV-13 filter 20x25x4", "Thermostat, commercial, 2-stage", "Refrigerant R-410A 25 lb",
         "Condenser fan motor 1/3 HP", "Pallet, standard 48x40", "Freight, LTL, zone 3", "Service call, 2 hr", "Contactor 40A 24V",
         "Copper line set 3/8-3/4 50ft", "Duct sealant, 1 gal", "Capacitor 45/5 uF"]

COMPANIES = ["Reliable Comfort HVAC", "TriCounty Plumbing & Air", "Lakeside Mechanical", "ProTemp Services", "Evergreen Climate",
             "Anchor Heating & Cooling", "Summit Air Solutions", "Northstar Mechanical"]

CODE_TEMPLATES = [
'''import threading

class Account:
    def __init__(self, balance):
        self.balance = balance
        self.lock = threading.Lock()

def transfer(src, dst, amount):
    if src.balance >= amount:
        src.balance -= amount
        dst.balance += amount
        return True
    return False

def batch_transfer(pairs):
    threads = [threading.Thread(target=transfer, args=p) for p in pairs]
    for t in threads: t.start()
    for t in threads: t.join()
''',
'''import sqlite3

def find_user(conn, username):
    cur = conn.cursor()
    cur.execute("SELECT id, email FROM users WHERE name = '%s'" % username)
    return cur.fetchone()

def update_email(conn, user_id, email):
    cur = conn.cursor()
    cur.execute("UPDATE users SET email = ? WHERE id = ?", (email, user_id))
    conn.commit()

def export(conn, path):
    rows = conn.execute("SELECT * FROM users").fetchall()
    with open(path, "w") as f:
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\\n")
''',
'''import os, subprocess, tempfile

def convert(src_path, fmt):
    out = tempfile.mktemp(suffix="." + fmt)
    cmd = f"convert {src_path} {out}"
    subprocess.run(cmd, shell=True, check=True)
    return out

def cleanup(paths):
    for p in paths:
        try:
            os.remove(p)
        except Exception:
            pass

def process_all(paths, fmt):
    outs = [convert(p, fmt) for p in paths]
    cleanup(paths)
    return outs
''',
'''from datetime import datetime

CACHE = {}

def get_rate(currency, fetch):
    key = currency
    if key in CACHE:
        return CACHE[key]
    rate = fetch(currency)
    CACHE[key] = rate
    return rate

def total_in_usd(lines, fetch):
    total = 0
    for amount, cur in lines:
        total += amount * get_rate(cur, fetch)
    return round(total, 2)

def parse_date(s):
    return datetime.strptime(s, "%m/%d/%Y")
''',
'''import hashlib, secrets

def hash_password(pw):
    return hashlib.md5(pw.encode()).hexdigest()

def verify(pw, stored):
    return hash_password(pw) == stored

def make_token(user_id):
    return f"{user_id}:{secrets.token_hex(8)}"

def check_token(token, user_id):
    uid, _ = token.split(":")
    return uid == str(user_id)
''',
]

AGENTS = ["Marcus", "Dana", "Priya", "Tom", "Alicia"]
ISSUES = ["no heat since last night", "AC blowing warm air", "thermostat screen blank", "water leaking from the indoor unit",
          "furnace short-cycling every few minutes", "loud banging when the system starts", "smell of gas near the furnace",
          "unit runs constantly but house won't cool"]


def denial_prompt(i):
    payer = R.choice(PAYERS); code, desc = R.choice(CPT); rc, tmpl = R.choice(DENIALS)
    n = R.randint(8, 24); days = R.randint(95, 200)
    reason = tmpl.format(n=n, desc=desc, code=code, days=days)
    dos = f"{R.randint(1,12):02d}/{R.randint(1,28):02d}/2026"
    letter = (f"{payer}\nClaims Department\n\nRE: Claim #{R.randint(10**9, 10**10-1)}   Member ID {R.randint(10**8,10**9-1)}\n"
              f"Provider: Great Lakes Rehabilitation Associates   Date of service: {dos}\n\n"
              f"Dear Provider,\n\nWe have reviewed the above claim for CPT {code} ({desc}). After review, the claim has been denied "
              f"for the following reason: {reason}. Remark code {rc}.\n\n"
              f"{R.choice(['Our clinical reviewer noted that the submitted progress notes do not document objective functional gains between visits.', 'The authorization on file covers dates through the prior month only.', 'Please note that services rendered by a non-participating provider are subject to the member out-of-network benefit.', 'Documentation submitted did not include the plan of care signed by the referring physician.'])} "
              f"You have the right to appeal this decision within {R.choice([60, 90, 180])} days of the date of this notice. Appeals must include "
              f"clinical documentation supporting medical necessity and any corrected claim information.\n\n"
              f"Sincerely,\n{payer} Provider Services")
    return ("Classify this denial letter and return ONLY a JSON object with keys: remark_code, category "
            "(one of: medical_necessity, authorization, bundling, coding_error, coverage, timely_filing, frequency), "
            "appealable (true/false), strongest_appeal_argument (one sentence), deadline_days (integer).\n\n" + letter)


def invoice_prompt(i):
    v = R.choice(VENDORS); n = R.randint(3, 7)
    lines = []; total = 0.0
    for _ in range(n):
        item = R.choice(ITEMS); qty = R.randint(1, 40); price = round(R.uniform(4, 900), 2)
        lines.append(f"  {qty:>3} x {item:<40} @ ${price:>8.2f}  = ${qty*price:>10.2f}"); total += qty * price
    tax = round(total * 0.06, 2); inv = f"{R.choice(['A','INV','N','GL'])}-{R.randint(10000, 99999)}"
    d = f"{R.randint(1,12):02d}/{R.randint(1,28):02d}/2026"; terms = R.choice(["Net 30", "Net 45", "2/10 Net 30", "Due on receipt"])
    po = f"PO-{R.randint(1000, 9999)}"
    doc = (f"{v}\n{R.randint(100,9999)} {R.choice(['Industrial Pkwy','Commerce Dr','Harbor Rd','Main St'])}, {R.choice(['Grand Rapids, MI','Toledo, OH','Fort Wayne, IN','Lansing, MI'])}\n\n"
           f"INVOICE {inv}    Date: {d}    Terms: {terms}    Customer PO: {po}\n\nBill to: Field Finesse Services LLC\n\n" + "\n".join(lines) +
           f"\n\n  Subtotal ${total:>10.2f}\n  Sales tax (6%) ${tax:>10.2f}\n  TOTAL DUE ${total+tax:>10.2f}\n\n"
           f"Remit to: {v}, Account ending {R.randint(1000,9999)}. {R.choice(['Late payments accrue 1.5% per month.', 'Thank you for your business.', 'Questions: ar@' + v.split()[0].lower() + '.com'])}")
    return ("Extract from this invoice and return ONLY JSON with keys: vendor, invoice_number, invoice_date (YYYY-MM-DD), "
            "po_number, payment_terms, subtotal, tax, total, line_item_count, due_date (YYYY-MM-DD, computed from terms).\n\n" + doc)


def audit_prompt(i):
    code = R.choice(CODE_TEMPLATES)
    return ("Audit this Python module. List each defect as 'SEVERITY: one-line description' (SEVERITY in HIGH/MED/LOW), "
            "at most 5 lines, then a final line 'VERDICT: PASS' or 'VERDICT: FAIL'. No prose.\n\n```python\n" + code + "```")


def scoring_prompt(i):
    c = R.choice(COMPANIES); rev = R.uniform(1.5, 12.0); margin = R.uniform(0.08, 0.22); techs = R.randint(2, 18)
    yrs = R.randint(6, 34); res = R.randint(35, 90); owner_age = R.randint(48, 71); churn = R.uniform(0.05, 0.3)
    doc = (f"Target: {c}\nRevenue (TTM): ${rev:.1f}M   EBITDA margin: {margin*100:.0f}%   Technicians: {techs}   Years operating: {yrs}\n"
           f"Mix: {res}% residential / {100-res}% commercial   Maintenance agreements: {R.randint(120, 2400)} active   Customer churn: {churn*100:.0f}%/yr\n"
           f"Owner age {owner_age}, {R.choice(['works in the business daily', 'semi-retired, GM runs operations', 'sole estimator and dispatcher'])}. "
           f"Fleet: {techs + R.randint(0,3)} vans, avg age {R.randint(3,9)} yrs. Software: {R.choice(['ServiceTitan', 'Housecall Pro', 'QuickBooks + paper tickets', 'FieldEdge'])}. "
           f"Google rating {R.uniform(3.6, 4.9):.1f} ({R.randint(40, 900)} reviews). {R.choice(['Two key techs are family members.', 'Largest customer is 22% of revenue.', 'No written employment agreements.', 'Lease on shop expires in 14 months.', 'Union shop; CBA renews next year.'])} "
           f"Asking price: {R.uniform(3.0, 6.5):.1f}x EBITDA.")
    return ("Score this HVAC acquisition target for a roll-up buyer. Return ONLY: 'SCORE: n/10' on the first line, then exactly "
            "three bullet lines: the biggest strength, the biggest risk, and the one diligence item to verify first.\n\n" + doc)


def transcript_prompt(i):
    a = R.choice(AGENTS); issue = R.choice(ISSUES); name = R.choice(["Mrs. Kowalski", "Mr. Bennett", "Ms. Ortiz", "Mr. Nguyen", "Mrs. Adebayo"])
    lines = [f"AGENT ({a}): Thanks for calling Field Finesse, this is {a}. How can I help?",
             f"CALLER: Hi, this is {name}. We've got {issue}.",
             f"AGENT: I'm sorry to hear that. Can I get the service address?",
             f"CALLER: {R.randint(100, 9999)} {R.choice(['Maple', 'Oak Ridge', 'Lakeshore', 'Birchwood', 'Hilltop'])} {R.choice(['Dr', 'Ln', 'Ave'])}.",
             f"AGENT: Got it. Is the system a {R.choice(['Carrier', 'Trane', 'Lennox', 'Goodman', 'Rheem'])}? Roughly how old?",
             f"CALLER: I think so, about {R.randint(4, 22)} years. {R.choice(['We had it serviced last spring.', 'Nobody has looked at it in years.', 'You guys installed it.', 'The previous owner put it in.'])}",
             f"AGENT: {R.choice(['Do you smell anything or hear anything unusual?', 'Is the breaker tripped?', 'Any water around the unit?'])}",
             f"CALLER: {R.choice(['No, nothing like that.', 'Yes, there is a burning smell.', 'The breaker was fine, I checked.', 'A little water on the floor, yes.'])}",
             f"AGENT: Okay. {'Because you mentioned a gas smell I need you to leave the house and call the gas utility first; we will come after they clear it.' if 'gas' in issue else 'I can get a technician out ' + R.choice(['today between 2 and 5', 'tomorrow morning 8 to 11', 'this afternoon, next available'])}. The diagnostic fee is ${R.choice([89, 99, 129])}, waived if you proceed with the repair.",
             f"CALLER: {R.choice(['That works.', 'Is there anything sooner?', 'Fine, please hurry, we have a newborn.', 'Okay. Do you take cards?'])}",
             f"AGENT: {R.choice(['Yes, we take all major cards.', 'I will flag it priority.', 'I will note that for dispatch.'])} You'll get a text when the tech is on the way. Anything else?",
             f"CALLER: No, that's it. Thanks {a}.",
             "AGENT: Thank you, have a good day."]
    return ("Summarize this call for the dispatcher. Return ONLY JSON with keys: caller, address, issue, system_brand, "
            "system_age_years, safety_flag (true/false), scheduled_window, diagnostic_fee, priority (low/normal/high), "
            "follow_up_needed (one sentence or null).\n\n" + "\n".join(lines))


GENERATORS = [denial_prompt, invoice_prompt, audit_prompt, scoring_prompt, transcript_prompt]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out", default="prompts.jsonl")
    a = ap.parse_args()
    with open(a.out, "w", encoding="utf-8") as f:
        for i in range(a.n):
            gen = GENERATORS[i % len(GENERATORS)]
            f.write(json.dumps({"type": gen.__name__.replace("_prompt", ""), "prompt": gen(i)}, ensure_ascii=False) + "\n")
    words = sum(len(json.loads(l)["prompt"].split()) for l in open(a.out, encoding="utf-8"))
    print(f"wrote {a.n} prompts to {a.out} (~{words} words, ~{int(words*1.4)} tokens; K3 cost ~${words*1.4*3/1e6 + a.n*200*15/1e6:.2f})")


if __name__ == "__main__":
    main()
