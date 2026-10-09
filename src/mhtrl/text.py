"""Text utilities: official HybridQA answer normalisation, EM/F1, tokenisation, BM25."""
import collections
import math
import re
import string

# --- answer normalisation and scoring (identical to HybridQA/evaluate_script.py) ---

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT = set(string.punctuation)


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def exact_match(gold: str, pred: str) -> float:
    return float(normalize_answer(gold) == normalize_answer(pred))


def f1_score(gold: str, pred: str) -> float:
    g = normalize_answer(gold).split()
    p = normalize_answer(pred).split()
    if not g or not p:
        return float(g == p)
    common = collections.Counter(g) & collections.Counter(p)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def contains_answer(answer: str, text: str) -> bool:
    """Token-boundary containment of the normalised answer in the normalised text.

    Mirrors the matching rule used by HybridQA's trace_answer.py, applied after
    normalisation so that punctuation and casing differences do not matter.
    """
    a = normalize_answer(answer)
    if not a:
        return False
    return (" " + a + " ") in (" " + normalize_answer(text) + " ")


# --- tokenisation for lexical features ---

_WORD = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    """a an the of in on at to for from by with and or is are was were be been being
    what which who whom whose when where why how that this these those it its as
    did does do has have had he she they his her their them there than then not
    no yes into about over under after before during while also only many much
    one two most more less least name named called s""".split()
)


def words(text: str):
    return _WORD.findall(text.lower())


def content_words(text: str):
    return [w for w in words(text) if w not in STOPWORDS]


def numbers(text: str):
    return set(re.findall(r"\d+(?:[.,]\d+)*", text))


class BM25:
    """Okapi BM25 with document frequencies estimated on a fixed corpus."""

    def __init__(self, corpus_tokens, k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n_docs = len(corpus_tokens)
        df = collections.Counter()
        total = 0
        for toks in corpus_tokens:
            df.update(set(toks))
            total += len(toks)
        self.avgdl = total / max(1, self.n_docs)
        self.idf = {t: math.log(1.0 + (self.n_docs - n + 0.5) / (n + 0.5)) for t, n in df.items()}
        self.default_idf = math.log(1.0 + (self.n_docs + 0.5) / 0.5)

    def score(self, query_tokens, doc_tokens) -> float:
        if not doc_tokens:
            return 0.0
        tf = collections.Counter(doc_tokens)
        dl = len(doc_tokens)
        s = 0.0
        for t in set(query_tokens):
            f = tf.get(t, 0)
            if f == 0:
                continue
            idf = self.idf.get(t, self.default_idf)
            s += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return s
