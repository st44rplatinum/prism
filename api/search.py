"""Query classification and search.

The original draft guessed a query's kind from punctuation and casing alone,
which mis-sorted several real inputs: `NR_003051.3` came back as HGVS because a
bare "r" sat in the marker tuple, and a lowercase `tp53` never registered as a
symbol because the test required the string to equal its own uppercase.

The fix is to stop guessing where we can look. We hold the full gene catalog
and every UniProt accession in memory, so exact identity is a lookup, and only
genuinely ambiguous free text falls through to fuzzy matching.
"""

from __future__ import annotations

import re

from api.store import Store, parse_protein_change

# NM_000546.6:p.Arg175His - a transcript/protein accession, a colon, then a
# coordinate prefixed by one of the HGVS sequence-type letters.
HGVS_RE = re.compile(
    r"^(?P<ref>[A-Za-z0-9_.]+)\s*:\s*(?P<kind>[cgpnmr])\.\s*(?P<change>.+)$",
    re.IGNORECASE,
)

REFSEQ_PREFIXES = ("NM_", "NR_", "NC_", "NG_", "NP_", "XM_", "XP_", "ENST", "ENSP", "ENSG")

# UniProt accession, per their published pattern.
UNIPROT_RE = re.compile(
    r"^([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})$"
)

# "TP53 R175H" / "TP53:R175H" / "TP53 p.Arg175His"
GENE_AND_CHANGE_RE = re.compile(r"^(?P<gene>[A-Za-z0-9\-]+)[\s:]+(?P<change>\S+)$")

THREE_TO_ONE_LOWER = {
    "ala": "A", "arg": "R", "asn": "N", "asp": "D", "cys": "C",
    "gln": "Q", "glu": "E", "gly": "G", "his": "H", "ile": "I",
    "leu": "L", "lys": "K", "met": "M", "phe": "F", "pro": "P",
    "ser": "S", "thr": "T", "trp": "W", "tyr": "Y", "val": "V",
}
THREE_LETTER_CHANGE_RE = re.compile(
    r"^(?:p\.)?(?P<wt>[A-Za-z]{3})(?P<pos>\d+)(?P<mut>[A-Za-z]{3})$"
)


def normalise_change(text: str) -> str | None:
    """Accept `R175H`, `p.R175H` or `p.Arg175His`; return the 1-letter form."""
    t = text.strip()
    if t.lower().startswith("p."):
        t = t[2:]

    three = THREE_LETTER_CHANGE_RE.match(text.strip())
    if three:
        wt = THREE_TO_ONE_LOWER.get(three.group("wt").lower())
        mut = THREE_TO_ONE_LOWER.get(three.group("mut").lower())
        if wt and mut:
            return f"{wt}{three.group('pos')}{mut}"

    parsed = parse_protein_change(t)
    return str(parsed) if parsed else None


def classify(query: str, store: Store) -> str:
    """Classify a search query.

    Order matters: the most specific and least ambiguous forms are tested
    first, and identity lookups against the catalog beat any pattern guess.
    """
    q = query.strip()
    if not q:
        return "text"

    # 1. Full HGVS - requires the colon AND a valid sequence-type prefix, so
    #    a trailing colon alone ("TP53:") no longer qualifies.
    if HGVS_RE.match(q):
        return "hgvs"

    # 2. "GENE CHANGE" pairs, checked before bare-symbol matching so that
    #    "TP53 R175H" is not written off as free text.
    pair = GENE_AND_CHANGE_RE.match(q)
    if pair and store.has_gene(pair.group("gene")) and normalise_change(pair.group("change")):
        return "protein_change"

    # 3. Known identifiers, by lookup rather than by shape.
    if store.has_gene(q):
        return "symbol"
    if store.gene_by_accession(q) or UNIPROT_RE.match(q.upper()):
        return "accession"
    if q.upper().startswith(REFSEQ_PREFIXES):
        return "accession"

    # 4. A bare protein change with no gene, e.g. "R175H".
    if normalise_change(q):
        return "protein_change"

    return "text"


def search(query: str, store: Store, limit: int = 10) -> tuple[str, list[dict]]:
    """Return (query_kind, hits)."""
    q = query.strip()
    kind = classify(q, store)
    hits: list[dict] = []

    if kind == "symbol":
        rec = store.gene_record(q)
        if rec:
            hits.append(_gene_hit(rec))

    elif kind == "accession":
        symbol = store.gene_by_accession(q)
        if symbol:
            hits.append(_gene_hit(store.gene_record(symbol)))

    elif kind == "protein_change":
        pair = GENE_AND_CHANGE_RE.match(q)
        if pair:
            symbol = pair.group("gene").upper()
            change = normalise_change(pair.group("change"))
            hits.append(_change_hit(store, symbol, change))
        else:
            # Bare change with no gene: offer it against every gene whose
            # sequence actually carries that wild-type residue at that
            # position, which is usually a very short list.
            change = normalise_change(q)
            parsed = parse_protein_change(change) if change else None
            if parsed:
                for symbol in store.genes:
                    seq = store.sequence(symbol) or ""
                    if parsed.pos0 < len(seq) and seq[parsed.pos0] == parsed.wt_aa:
                        hits.append(_change_hit(store, symbol, change))
                    if len(hits) >= limit:
                        break

    elif kind == "hgvs":
        m = HGVS_RE.match(q)
        ref = m.group("ref")
        change = normalise_change(m.group("change"))
        symbol = store.gene_by_accession(ref)
        if symbol and change:
            hits.append(_change_hit(store, symbol, change))
        elif change:
            hits.append({"kind": "hgvs", "variant": change, "accession": ref})
        else:
            hits.append({"kind": "hgvs", "accession": ref})

    else:
        # Free text over symbols and protein names. Prefix matches on the
        # symbol rank above substring matches anywhere.
        ql = q.lower()
        prefix, substring = [], []
        for symbol in store.genes:
            rec = store.gene_record(symbol)
            if symbol.lower().startswith(ql):
                prefix.append(_gene_hit(rec))
            elif ql in symbol.lower() or ql in rec["protein_name"].lower():
                substring.append(_gene_hit(rec))
        hits = prefix + substring

    return kind, hits[:limit]


def _gene_hit(rec: dict) -> dict:
    return {
        "kind": "symbol",
        "symbol": rec["symbol"],
        "accession": rec["accession"],
        "protein_name": rec["protein_name"],
    }


def _change_hit(store: Store, symbol: str, change: str | None) -> dict:
    rec = store.gene_record(symbol)
    hit = {
        "kind": "protein_change",
        "symbol": symbol,
        "variant": change,
        "accession": rec["accession"] if rec else None,
        "protein_name": rec["protein_name"] if rec else None,
    }
    parsed = parse_protein_change(change) if change else None
    if parsed and rec:
        known = store.lookup_label(symbol, parsed)
        if known:
            hit["label"] = known[0]
    return hit
