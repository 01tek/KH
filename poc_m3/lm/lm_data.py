"""Lane-based token stream for training with carried recurrent state (truncated BPTT across segments).

Each of B lanes reads its own contiguous stream from a uint16 token file. Every step returns the next
T tokens of every lane, so the model's state carries across steps: training context is unbounded.
Synthetic facts are spliced into the text ("The secret code of Zelith is 482913.") and queried many
segments later; answer tokens get an extra loss weight. Fact/query distances therefore range from a few
tokens to tens of thousands (set by p_fact, p_query, max_age).
"""
import json
import os

import numpy as np
import torch


def token_dtype(data_dir):
    """Token file dtype from meta.json (uint16 unless the tokenizer's vocab needed uint32)."""
    try:
        return np.dtype(json.load(open(os.path.join(data_dir, "meta.json"))).get("dtype", "uint16"))
    except (OSError, ValueError):
        return np.dtype(np.uint16)

SYL = ["ka", "zo", "ri", "vel", "thu", "mor", "sen", "dra", "li", "quo", "pax", "nev", "ul", "bri", "ash",
       "tor", "ime", "gal", "fen", "oro", "sab", "yel", "kri", "dun"]


CITIES = ["Lisbon", "Nairobi", "Oslo", "Lima", "Hanoi", "Quito", "Dakar", "Tunis", "Riga", "Perth", "Austin",
          "Denver", "Kyoto", "Seville", "Porto", "Krakow", "Bergen", "Cairo", "Havana", "Dublin"]
COLORS = ["red", "blue", "green", "yellow", "purple", "orange", "black", "white", "silver", "golden", "pink", "brown"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]
ANIMALS = ["fox", "otter", "falcon", "tiger", "dolphin", "rabbit", "wolf", "owl", "turtle", "panda", "eagle", "horse"]
FRUITS = ["mango", "cherry", "lemon", "peach", "plum", "apple", "banana", "grape", "melon", "kiwi"]
LANGS = ["French", "Spanish", "Arabic", "Japanese", "German", "Hindi", "Swahili", "Russian", "Korean", "Turkish"]
JOBS = ["teacher", "pilot", "baker", "nurse", "painter", "lawyer", "farmer", "chemist", "sailor", "architect"]
LISTS = {"city": None, "color": None, "day": None, "month": MONTHS, "animal": ANIMALS, "fruit": FRUITS, "lang": LANGS,
         "job": JOBS}

# (fact, query prefix, value kind). {n} = entity name, {v} = value. Template 0 is the original LM v1-v13 format.
TEMPLATES = [
    ("The secret code of {n} is {v}.", "What is the secret code of {n}? The secret code of {n} is", "d6"),
    ("The password for {n} is {v}.", "What is the password for {n}? The password for {n} is", "word"),
    ("{n}'s phone number is {v}.", "What is {n}'s phone number? {n}'s phone number is", "d7"),
    ("{n} lives in {v}.", "Where does {n} live? {n} lives in", "city"),
    ("{n} was born in {v}.", "When was {n} born? {n} was born in", "year"),
    ("Remember this: {n} = {v}.", "Recall: {n} =", "d4"),
    ("{n}'s favorite color is {v}.", "What is {n}'s favorite color? {n}'s favorite color is", "color"),
    ("The meeting with {n} is on {v}.", "When is the meeting with {n}? The meeting with {n} is on", "day"),
    ("Item {n} costs {v} dollars.", "How much does item {n} cost? Item {n} costs", "d3"),
    ("Agent {n} uses the codename {v}.", "What codename does agent {n} use? Agent {n} uses the codename", "word"),
]
TEMPLATES_XL = TEMPLATES + [  # --templates xl: 40 phrasings, for generalisation beyond a handful of patterns
    ("The access PIN for {n} is {v}.", "What is the access PIN for {n}? The access PIN for {n} is", "d4"),
    ("{n} works as a {v}.", "What does {n} work as? {n} works as a", "job"),
    ("{n} speaks {v} at home.", "What language does {n} speak at home? {n} speaks", "lang"),
    ("The pet of {n} is a {v}.", "What is the pet of {n}? The pet of {n} is a", "animal"),
    ("{n}'s favorite fruit is the {v}.", "What is {n}'s favorite fruit? {n}'s favorite fruit is the", "fruit"),
    ("{n} was hired in {v}.", "When was {n} hired? {n} was hired in", "month"),
    ("Flight {n} departs from gate {v}.", "From which gate does flight {n} depart? Flight {n} departs from gate", "d2"),
    ("Room {n} is on floor {v}.", "On which floor is room {n}? Room {n} is on floor", "d2"),
    ("The license plate of {n} reads {v}.", "What does the license plate of {n} read? The license plate of {n} reads", "d6"),
    ("{n} has {v} siblings.", "How many siblings does {n} have? {n} has", "d1"),
    ("Ticket {n} is valid until {v}.", "Until when is ticket {n} valid? Ticket {n} is valid until", "year"),
    ("The ship {n} sails to {v}.", "Where does the ship {n} sail to? The ship {n} sails to", "city"),
    ("Professor {n} teaches in room {v}.", "In which room does Professor {n} teach? Professor {n} teaches in room", "d3"),
    ("The recipe of {n} needs {v} grams of sugar.", "How many grams of sugar does the recipe of {n} need? The recipe of {n} needs", "d3"),
    ("{n} parked the car in spot {v}.", "Where did {n} park the car? {n} parked the car in spot", "d3"),
    ("The wifi password at {n} is {v}.", "What is the wifi password at {n}? The wifi password at {n} is", "word"),
    ("Order {n} ships on {v}.", "When does order {n} ship? Order {n} ships on", "day"),
    ("{n} paints everything {v}.", "What color does {n} paint everything? {n} paints everything", "color"),
    ("The museum in {n} opened in {v}.", "When did the museum in {n} open? The museum in {n} opened in", "year"),
    ("Doctor {n}'s pager number is {v}.", "What is Doctor {n}'s pager number? Doctor {n}'s pager number is", "d5"),
    ("{n} keeps a {v} in the garden.", "What does {n} keep in the garden? {n} keeps a", "animal"),
    ("The bank account of {n} ends in {v}.", "What does the bank account of {n} end in? The bank account of {n} ends in", "d4"),
    ("The team {n} plays on {v}s.", "When does the team {n} play? The team {n} plays on", "day"),
    ("{n} studied {v} in college.", "What did {n} study in college? {n} studied", "lang"),
    ("The bus to {n} leaves at {v} o'clock.", "When does the bus to {n} leave? The bus to {n} leaves at", "d1"),
    ("{n}'s badge number is {v}.", "What is {n}'s badge number? {n}'s badge number is", "d6"),
    ("The sculpture {n} weighs {v} kilograms.", "How much does the sculpture {n} weigh? The sculpture {n} weighs", "d3"),
    ("{n} grows {v} trees.", "What trees does {n} grow? {n} grows", "fruit"),
    ("The vault code of bank {n} is {v}.", "What is the vault code of bank {n}? The vault code of bank {n} is", "d7"),
    ("{n} was promoted in {v}.", "When was {n} promoted? {n} was promoted in", "month"),
]
HELDOUT = [  # never seen in training: tests whether archival writes generalise beyond the trained phrasings
    ("{n}'s locker combination is {v}.", "What is {n}'s locker combination? {n}'s locker combination is", "d5"),
    ("The vault of {n} opens with {v}.", "What opens the vault of {n}? The vault of {n} opens with", "word"),
    ("{n} moved to {v} last year.", "Where did {n} move last year? {n} moved to", "city"),
]


# --templates comp: compositional facts = sentence frame x attribute, so the model must learn "entity's ATTRIBUTE is
# VALUE" in general, not a list of sentences. Held-out evaluation (comp_heldout) uses unseen frames AND attributes.
FRAMES = [  # (fact, question prefix)
    ("{n}'s {a} is {v}.", "What is {n}'s {a}? {n}'s {a} is"),
    ("The {a} of {n} is {v}.", "What is the {a} of {n}? The {a} of {n} is"),
    ("For {n}, the {a} is {v}.", "For {n}, what is the {a}? For {n}, the {a} is"),
    ("Note: {n} has the {a} {v}.", "Which {a} does {n} have? {n} has the {a}"),
    ("According to the records, the {a} of {n} is {v}.", "According to the records, what is the {a} of {n}? It is"),
    ("{n} told me that their {a} is {v}.", "What did {n} say their {a} is? Their {a} is"),
]
FRAMES_HELDOUT = [
    ("Officially, {n}'s {a} reads {v}.", "Officially, what does {n}'s {a} read? It reads"),
    ("We confirmed that the {a} for {n} is {v}.", "What is the {a} for {n}? The {a} for {n} is"),
]
ATTRS = [  # (attribute, value kind)
    ("secret code", "d6"), ("locker number", "d3"), ("phone extension", "d4"), ("membership number", "d6"),
    ("PIN", "d4"), ("room number", "d3"), ("seat number", "d2"), ("employee ID", "d5"), ("zip code", "d5"),
    ("account number", "d7"), ("order number", "d6"), ("ticket number", "d5"), ("badge code", "d4"),
    ("parking spot", "d3"), ("apartment number", "d3"), ("license number", "d6"), ("shoe size", "d2"),
    ("password", "word"), ("nickname", "word"), ("codename", "word"), ("username", "word"), ("pet's name", "word"),
    ("boat's name", "word"), ("favorite word", "word"), ("secret word", "word"), ("call sign", "word"),
    ("home city", "city"), ("birthplace", "city"), ("favorite city", "city"), ("office city", "city"),
    ("birth year", "year"), ("graduation year", "year"), ("wedding year", "year"), ("arrival year", "year"),
    ("favorite color", "color"), ("car color", "color"), ("lucky color", "color"),
    ("favorite day", "day"), ("day off", "day"), ("gym day", "day"),
    ("birth month", "month"), ("vacation month", "month"), ("favorite animal", "animal"), ("pet", "animal"),
    ("favorite fruit", "fruit"), ("native language", "lang"), ("second language", "lang"), ("profession", "job"),
    ("dream job", "job"), ("hotel room", "d3"), ("gate number", "d2"), ("bank PIN", "d4"), ("safe combination", "d6"),
]
ATTRS_HELDOUT = [
    ("locker combination", "d5"), ("vault code", "d7"), ("radio frequency", "d3"), ("middle name", "word"),
    ("summer house city", "city"), ("retirement year", "year"), ("umbrella color", "color"), ("market day", "day"),
]


# real first names and surnames: facts about ordinary-looking people, so the write gate cannot key on made-up names
FIRST_NAMES = """James Mary Robert Patricia John Jennifer Michael Linda David Elizabeth William Barbara Richard Susan Joseph
Jessica Thomas Sarah Charles Karen Daniel Lisa Matthew Nancy Anthony Betty Mark Margaret Donald Sandra Steven Ashley Paul
Kimberly Andrew Emily Joshua Donna Kenneth Michelle Kevin Carol Brian Amanda George Melissa Timothy Deborah Ronald Stephanie
Edward Rebecca Jason Sharon Jeffrey Laura Ryan Cynthia Jacob Kathleen Gary Amy Nicholas Angela Eric Shirley Jonathan Anna
Stephen Brenda Larry Pamela Justin Emma Scott Nicole Brandon Helen Benjamin Samantha Samuel Katherine Gregory Christine
Alexander Debra Frank Rachel Patrick Carolyn Raymond Janet Jack Catherine Dennis Maria Jerry Heather Tyler Diane Aaron Ruth
Jose Julie Adam Olivia Nathan Joyce Henry Virginia Peter Victoria Zachary Kelly Douglas Lauren Harold Christina Kyle Joan
Ahmed Fatima Omar Aisha Youssef Leila Hassan Amina Karim Nadia Mohamed Salma Ali Yasmin Ibrahim Zainab Pierre Camille Louis
Chloe Antoine Manon Hugo Juliette Lucas Ines Mateo Sofia Diego Lucia Carlos Valentina Javier Elena Marco Giulia Luca Chiara
Hans Anke Lukas Lena Jonas Mia Sven Ingrid Lars Astrid Ivan Olga Dmitri Natasha Sergei Irina Kenji Yuki Hiroshi Aiko Wei Mei
Jun Lin Raj Priya Arjun Ananya Vikram Kavya Kwame Ama Kofi Abena Chidi Ngozi Tunde Ayesha""".split()
SURNAMES = """Smith Johnson Williams Brown Jones Garcia Miller Davis Rodriguez Martinez Hernandez Lopez Gonzalez Wilson Anderson
Thomas Taylor Moore Jackson Martin Lee Perez Thompson White Harris Sanchez Clark Ramirez Lewis Robinson Walker Young Allen
King Wright Scott Torres Nguyen Hill Flores Green Adams Nelson Baker Hall Rivera Campbell Mitchell Carter Roberts Gomez
Phillips Evans Turner Diaz Parker Cruz Edwards Collins Reyes Stewart Morris Morales Murphy Cook Rogers Gutierrez Ortiz
Morgan Cooper Peterson Bailey Reed Kelly Howard Ramos Kim Cox Ward Richardson Watson Brooks Chavez Wood James Bennett Gray
Mendoza Ruiz Hughes Price Alvarez Castillo Sanders Patel Myers Long Ross Foster Jimenez Dubois Laurent Bernard Moreau
Lefebvre Rossi Russo Ferrari Esposito Bianchi Muller Schmidt Schneider Fischer Weber Becker Ivanov Petrov Sokolov Tanaka
Suzuki Takahashi Watanabe Wang Zhang Chen Liu Singh Kumar Sharma Gupta Mensah Okafor Adeyemi Benali Alaoui Haddad Mansour
Nasser Khalil Saidi Amrani Bakr Farouk""".split()


class FactFactory:
    """templates: "secret" (template 0 only, the LM v1-v13 format), "multi" (all training templates), "heldout"."""

    def __init__(self, tok, rng, templates="secret", chat_frac=0.0, natural_names=0.0, no_think=False):
        self.tok, self.rng = tok, rng
        # no_think: render chat turns in non-thinking mode (Qwen3.5-4B's template otherwise opens a <think> block)
        self.tpl_kw = {"enable_thinking": False} if no_think else {}
        self.natural_names = natural_names  # share of facts about real-looking names (first, last or both)
        # chat_frac: share of queries rendered as a chat turn (tokenizer chat template), e.g. "<|im_start|>user\nWhat is
        # X's PIN?<|im_end|>\n<|im_start|>assistant\nX's PIN is 4821.<|im_end|>", so recall works inside a conversation
        self.chat_frac = chat_frac if getattr(tok, "chat_template", None) else 0.0
        if templates in ("comp", "comp_heldout", "mix"):  # expand frames x attributes into (fact, question, kind) triples
            fr, at = (FRAMES_HELDOUT, ATTRS_HELDOUT) if templates == "comp_heldout" else (FRAMES, ATTRS)
            self.tpl = [(f.replace("{a}", a), q.replace("{a}", a), k) for f, q in fr for a, k in at]
            if templates == "mix":  # + the 40 sentence phrasings (held-out sentences stay out)
                self.tpl = self.tpl + TEMPLATES_XL
        else:
            self.tpl = {"secret": TEMPLATES[:1], "multi": TEMPLATES, "xl": TEMPLATES_XL, "heldout": HELDOUT}[templates]

    def name(self):
        if self.natural_names and self.rng.random() < self.natural_names:  # no extra RNG draw when off: streams unchanged
            r = self.rng.random()
            first, last = str(self.rng.choice(FIRST_NAMES)), str(self.rng.choice(SURNAMES))
            return f"{first} {last}" if r < 0.5 else (first if r < 0.75 else last)
        n = "".join(self.rng.choice(SYL, size=self.rng.integers(2, 4)))
        return n.capitalize()

    def value(self, kind):
        r = self.rng
        if kind[0] == "d" and kind[1:].isdigit():
            k = int(kind[1:])
            return str(r.integers(10 ** (k - 1), 10 ** k))
        if kind == "word":
            return "".join(r.choice(SYL, size=r.integers(2, 4)))
        if kind == "city":
            return str(r.choice(CITIES))
        if LISTS.get(kind):
            return str(r.choice(LISTS[kind]))
        if kind == "year":
            return str(r.integers(1900, 2030))
        if kind == "color":
            return str(r.choice(COLORS))
        if kind == "day":
            return str(r.choice(DAYS))
        raise ValueError(kind)

    def make(self, avoid=None):
        # no RNG draw when there is one template, so "secret" reproduces the LM v1-v13 fact stream exactly
        fact_t, q_t, kind = self.tpl[self.rng.integers(len(self.tpl)) if len(self.tpl) > 1 else 0]
        name = self.name()
        for _ in range(20):  # avoid=names still pending in this lane: no two live facts share a key
            if not avoid or name not in avoid:
                break
            name = self.name()
        v = self.value(kind)
        enc = lambda s: self.tok(s, add_special_tokens=False)["input_ids"]
        fact = enc(" " + fact_t.format(n=name, v=v))
        if self.chat_frac and self.rng.random() < self.chat_frac and "? " in q_t:
            question, _, prefix = q_t.format(n=name).partition("? ")
            turn = self.tok.apply_chat_template([{"role": "user", "content": question + "?"}], tokenize=False,
                                                add_generation_prompt=True, **self.tpl_kw)
            turn = turn.replace(self.tok.bos_token or "\0", "")  # no BOS in the middle of a stream
            q_prefix = enc("\n" + turn + prefix)
            ans = enc(f" {v}")
            return {"name": name, "code": v, "fact": fact, "query": q_prefix + ans + enc(".<|im_end|>\n"),
                    "ans_start": len(q_prefix), "ans_len": len(ans)}
        q_prefix = enc(" " + q_t.format(n=name))
        ans = enc(f" {v}")
        return {"name": name, "code": v, "fact": fact, "query": q_prefix + ans + enc("."),
                "ans_start": len(q_prefix), "ans_len": len(ans)}


class LaneStream:
    def __init__(self, path, tok, B, T, seed=0, p_fact=0.1, p_query=0.1, max_pending=64, max_age=200,
                 fact_rate=None, query_rate=None, drill_frac=0.0, drill_k=8, age_warmup=0, age_target="random",
                 templates="secret", drill_gap="half", chat_frac=0.0, natural_names=0.0, no_think=False):
        self.data = np.memmap(path, dtype=token_dtype(os.path.dirname(path)), mode="r")
        self.B, self.T = B, T
        self.rng = np.random.default_rng(seed)
        self.ptr = self.rng.integers(0, len(self.data) - 10 * T, size=B)
        self.facts = FactFactory(tok, self.rng, templates, chat_frac, natural_names, no_think)
        self.drill_gap = drill_gap
        self.pending = [[] for _ in range(B)]  # (fact dict, step written)
        self.p_fact, self.p_query, self.max_pending, self.max_age = p_fact, p_query, max_pending, max_age
        # v2 curriculum: Poisson numbers of facts/queries per segment (instead of Bernoulli p_fact/p_query),
        # "drill" lanes with drill_k facts + their queries inside every segment, and a query-age curriculum
        # whose max age doubles from 1 segment up to max_age over age_warmup steps.
        self.fact_rate, self.query_rate = fact_rate, query_rate
        self.n_drill = int(round(drill_frac * B))
        self.drill_k, self.age_warmup = drill_k, age_warmup
        # age_target="loguniform": each query samples a target age log-uniformly in [1, max_age] (segments) and
        # asks the pending fact closest to it. With "random" (v2-v10) queries pop a random pending fact, which keeps
        # the pool young: lm9 had median query age 19 segments and 0% of queries >= 512 segments.
        self.age_target = age_target
        self.step = 0
        self.last_query_age = []

    def _max_age(self):
        if not self.age_warmup or self.step >= self.age_warmup:
            return self.max_age
        return max(1, int(self.max_age ** (self.step / self.age_warmup)))

    def _drill(self, b):
        """Segment with drill_k facts in the first half and queries for all of them in the second half."""
        T = self.T
        fs = [self.facts.make() for _ in range(self.drill_k)]
        ins = [("fact", f["fact"], None) for f in fs] + [("query", f["query"], f) for f in self.rng.permutation(fs)]
        n_ins = sum(len(x[1]) for x in ins)
        raw = self._take(b, max(2, T + 1 - n_ins))
        n = len(raw)
        if self.drill_gap == "far":  # facts in the first 15%, queries in the last 15%: gap >= 0.7 T, beyond a short GDN memory
            lo, hi = int(0.15 * n), int(0.85 * n)
            cuts = np.concatenate([np.sort(self.rng.integers(0, lo + 1, size=self.drill_k)),
                                   np.sort(self.rng.integers(hi, n + 1, size=self.drill_k))])
        else:
            half = n // 2
            cuts = np.concatenate([np.sort(self.rng.integers(0, half + 1, size=self.drill_k)),
                                   np.sort(self.rng.integers(half, n + 1, size=self.drill_k))])
        return raw, ins, cuts

    def _take(self, b, n):
        if self.ptr[b] + n >= len(self.data):
            self.ptr[b] = 0
        out = self.data[self.ptr[b]:self.ptr[b] + n].astype(np.int64)
        self.ptr[b] += n
        return out

    def next(self):
        T = self.T
        ids = np.empty((self.B, T + 1), dtype=np.int64)
        ans = np.zeros((self.B, T), dtype=bool)  # target positions that predict an answer token
        mem = np.zeros((self.B, T + 1), dtype=bool)  # input tokens that belong to an inserted fact or query
        self.last_query_age = []
        self.last_queries = []  # (lane, answer target positions, age in segments; -1 = drill) for per-age accuracy
        max_age = self._max_age()
        for b in range(self.B):
            if b < self.n_drill:
                raw, ins, cuts = self._drill(b)
            else:
                pend = [(f, s) for f, s in self.pending[b] if self.step - s <= max_age]
                ins = []
                n_f = self.rng.poisson(self.fact_rate) if self.fact_rate is not None else int(self.rng.random() < self.p_fact)
                once = self.age_target == "loguniform_once"
                live = {g["name"] for g, _ in pend} if once else None
                for _ in range(n_f):
                    if len(pend) < self.max_pending:
                        f = self.facts.make(avoid=live)
                        if once:
                            live.add(f["name"])
                        ins.append(("fact", f["fact"], None))
                        pend.append((f, self.step))
                n_q = self.rng.poisson(self.query_rate) if self.query_rate is not None else int(self.rng.random() < self.p_query)
                asked = set()
                for _ in range(n_q):
                    old = [i for i, (f, s) in enumerate(pend) if s < self.step and i not in asked]
                    if not old:
                        break
                    if self.age_target == "loguniform_once":
                        # log-uniform target age, nearest pending fact, asked exactly ONCE (popped). Nothing recurs,
                        # so a fact can only be answered from the context, never from weights.
                        target = np.exp(self.rng.uniform(0.0, np.log(max(1, max_age))))
                        i = min(old, key=lambda j: abs((self.step - pend[j][1]) - target))
                        f, s = pend.pop(i)
                    elif self.age_target == "loguniform":
                        # FLAWED (LM v11-v14, node sweep 1): facts persist and every query re-states the answer, so the
                        # same (name, code) recurs for thousands of steps and gets memorised in the WEIGHTS. A pure GDN
                        # then "recalls" 80% of facts 256+ segments old in training while in-context recall never forms.
                        target = np.exp(self.rng.uniform(0.0, np.log(max(1, max_age))))
                        i = min(old, key=lambda j: abs((self.step - pend[j][1]) - target))
                        f, s = pend[i]
                        pend[i] = (f, self.step)
                        asked.add(i)
                    else:
                        i = old[self.rng.integers(len(old))]
                        f, s = pend.pop(i)
                    ins.append(("query", f["query"], f))
                    self.last_query_age.append(self.step - s)
                self.pending[b] = pend
                n_ins = sum(len(x[1]) for x in ins)
                raw = self._take(b, max(2, T + 1 - n_ins))
                # insert at random cut points, in order (facts written this step come before any query)
                cuts = np.sort(self.rng.integers(0, len(raw) + 1, size=len(ins)))
            seq, prev = [], 0
            ages = iter(self.last_query_age[len(self.last_query_age) - sum(k == "query" for k, _, _ in ins):]
                        if b >= self.n_drill else [])
            for (kind, toks, f), c in zip(ins, cuts):
                seq.append(raw[prev:c])
                if kind == "query":
                    a0 = sum(len(x) for x in seq) + f["ans_start"]  # first answer token index in seq
                    pos = [a0 + j - 1 for j in range(f["ans_len"]) if 0 <= a0 + j - 1 < T]
                    for p_ in pos:
                        ans[b, p_] = True  # target at position p predicts token p+1
                    if pos and len(pos) == f["ans_len"]:
                        self.last_queries.append((b, pos, next(ages, -1)))
                seq.append(np.asarray(toks, dtype=np.int64))
                prev = c
            seq.append(raw[prev:])
            ids[b] = np.concatenate(seq)[:T + 1]
            pos = 0
            for i, piece in enumerate(seq):  # seq alternates raw text (even) and inserted fact/query tokens (odd)
                if i % 2:
                    mem[b, pos:pos + len(piece)] = True
                pos += len(piece)
        self.step += 1
        self.last_mem = torch.from_numpy(mem[:, :T])
        x = torch.from_numpy(ids[:, :-1])
        y = torch.from_numpy(ids[:, 1:])
        return x, y, torch.from_numpy(ans)


class Prefetcher:
    """Runs stream.next() in a background thread so batch construction overlaps the GPU step.
    next() returns (x, y, ans, mem, queries, query_ages, ptr, step): the metadata of *that* batch (the stream's own
    last_* attributes already describe later batches) and the stream position right after it (for checkpoints)."""

    def __init__(self, stream, depth=3):
        import queue
        import threading
        self.s, self.q = stream, queue.Queue(maxsize=depth)
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            x, y, ans = self.s.next()
            self.q.put((x.pin_memory(), y.pin_memory(), ans.pin_memory(), self.s.last_mem.pin_memory(),
                        list(self.s.last_queries), list(self.s.last_query_age), self.s.ptr.copy(), self.s.step))

    def next(self):
        return self.q.get()

