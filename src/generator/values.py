import random, datetime as dt

EQUIP = [("MX-101", "Ribbon Blender", "Room B-12"),
         ("GR-204", "Granulator", "Room B-14"),
         ("TP-330", "Tablet Press", "Room C-03"),
         ("CT-415", "Coating Pan", "Room C-07")]
STEPS = ["Remove product residue", "Rinse with purified water",
         "Wash with approved detergent", "Rinse until clear",
         "Wipe with 70% IPA", "Visual inspection"]
PARAMS = [("pH of final rinse", "pH", 5.0, 7.5),
          ("Conductivity", "uS/cm", 0.0, 1.3),
          ("TOC", "ppb", 0.0, 500.0)]
NAMES = ["A. Patel", "R. Mehta", "S. Desai", "K. Shah", "N. Joshi"]

# Phase 2: probability that a measurement has NO lower / NO upper limit (stored as null).
# Without this, null handling would never be tested by the field-level F1.
LIMIT_NULL_P = 0.25


def sample_record(rng: random.Random, idx: int) -> dict:
    eid, ename, loc = rng.choice(EQUIP)
    d = dt.date(2026, 1, 1) + dt.timedelta(days=rng.randint(0, 270))
    steps = [{"no": i + 1, "description": s,
              "result": rng.choices(["pass", "fail", "n/a"], [85, 10, 5])[0],
              "initials": "".join(w[0] for w in rng.choice(NAMES).replace(".", "").split())}
             for i, s in enumerate(rng.sample(STEPS, rng.randint(4, 6)))]
    meas = []
    for p, u, lo, hi in rng.sample(PARAMS, rng.randint(1, 3)):
        v = round(rng.uniform(lo, hi * 1.1), 2)   # sometimes out of limit on purpose
        lim_min = None if rng.random() < LIMIT_NULL_P else lo
        lim_max = None if rng.random() < LIMIT_NULL_P else hi
        meas.append({"parameter": p, "value": v, "unit": u,
                     "limit_min": lim_min, "limit_max": lim_max})
    devs = [] if rng.random() < 0.7 else ["Residue found on agitator shaft"]
    return {"document_id": f"CR-2026-{idx:05d}", "document_type": "cleaning_record",
            "equipment": {"id": eid, "name": ename, "location": loc},
            "date": d.isoformat(), "performed_by": rng.choice(NAMES),
            "reviewed_by": rng.choice(NAMES + [None]),
            "steps": steps, "measurements": meas, "deviations": devs}
