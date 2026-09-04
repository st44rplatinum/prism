"""Canonical human protein sequences from UniProt, with a local JSON cache.

We deliberately anchor every variant to the reviewed (Swiss-Prot) canonical
isoform rather than to whichever RefSeq transcript a ClinVar submitter used.
Variants whose reported wild-type residue disagrees with the canonical sequence
are dropped downstream - that mismatch is the cheapest reliable signal that a
submission refers to a different isoform and would otherwise inject label noise
at the wrong position.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import requests

UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"
UNIPROT_ENTRY = "https://rest.uniprot.org/uniprotkb/{acc}.json"
HGNC_SYMBOL = "https://rest.genenames.org/fetch/symbol/{symbol}"
HUMAN = "9606"


@dataclass
class ProteinRecord:
    gene: str
    accession: str
    entry_name: str
    protein_name: str
    sequence: str
    length: int

    @property
    def too_long_for_esm(self) -> bool:
        return self.length > 1022


class UniProtClient:
    def __init__(self, cache_path: Path, timeout: float = 30.0, pause: float = 0.15):
        self.cache_path = Path(cache_path)
        self.timeout = timeout
        self.pause = pause
        self._cache: dict[str, dict] = {}
        if self.cache_path.exists():
            self._cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
        self._session = requests.Session()
        self._session.headers["User-Agent"] = "esm2-vep/0.1 (research)"

    def _save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._cache, indent=1), encoding="utf-8")
        tmp.replace(self.cache_path)

    def fetch(self, gene: str) -> ProteinRecord | None:
        """Canonical reviewed human protein for a gene symbol, or None."""
        if gene in self._cache:
            blob = self._cache[gene]
            return ProteinRecord(**blob) if blob else None

        query = f"gene_exact:{gene} AND organism_id:{HUMAN} AND reviewed:true"
        params = {
            "query": query,
            "fields": "accession,id,protein_name,sequence,gene_primary",
            "format": "json",
            "size": "25",
        }
        try:
            resp = self._session.get(UNIPROT_SEARCH, params=params, timeout=self.timeout)
            resp.raise_for_status()
            results = resp.json().get("results", [])
        except (requests.RequestException, ValueError) as exc:
            raise RuntimeError(f"UniProt lookup failed for {gene}: {exc}") from exc
        time.sleep(self.pause)

        record = self._pick(gene, results)
        if record is None:
            # UniProt's text index applies English stopwording, so a symbol like
            # WAS turns `gene_exact:WAS` into a match-all and the real entry
            # (P42768) never appears. HGNC is the naming authority for human
            # gene symbols, so resolve the symbol there and fetch by accession.
            record = self._fetch_via_hgnc(gene)
        self._cache[gene] = asdict(record) if record else None
        self._save()
        return record

    def _fetch_via_hgnc(self, gene: str) -> ProteinRecord | None:
        try:
            resp = self._session.get(
                HGNC_SYMBOL.format(symbol=gene),
                headers={"Accept": "application/json"},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            docs = resp.json().get("response", {}).get("docs", [])
        except (requests.RequestException, ValueError):
            return None
        time.sleep(self.pause)

        accessions = [a for d in docs for a in (d.get("uniprot_ids") or [])]
        if not accessions:
            return None

        try:
            entry = self._session.get(
                UNIPROT_ENTRY.format(acc=accessions[0]), timeout=self.timeout
            )
            entry.raise_for_status()
            item = entry.json()
        except (requests.RequestException, ValueError):
            return None
        time.sleep(self.pause)

        seq = item.get("sequence", {}).get("value", "")
        if not seq:
            return None
        desc = (
            item.get("proteinDescription", {})
            .get("recommendedName", {})
            .get("fullName", {})
            .get("value", "")
        )
        return ProteinRecord(
            gene=gene,
            accession=item.get("primaryAccession", accessions[0]),
            entry_name=item.get("uniProtkbId", ""),
            protein_name=desc,
            sequence=seq,
            length=len(seq),
        )

    @staticmethod
    def _pick(gene: str, results: list[dict]) -> ProteinRecord | None:
        """Choose the entry whose *primary* gene name matches exactly.

        A gene_exact search also returns entries carrying the symbol only as a
        synonym, so we re-check the primary name. If nothing matches exactly we
        return None rather than guessing: an earlier version fell back to the
        longest hit, which silently resolved WAS to a 1464-residue lipid
        transfer protein instead of the 502-residue Wiskott-Aldrich protein and
        mis-positioned every variant in the gene. A missing gene is a visible
        failure; a wrong gene is not.
        """
        exact = []
        for item in results:
            primaries = {
                g.get("geneName", {}).get("value", "").upper()
                for g in item.get("genes", [])
            }
            # Compared case-insensitively: UniProt writes C9orf72 where ClinVar
            # and HGNC write C9ORF72.
            if gene.upper() in primaries:
                exact.append(item)
        if not exact:
            return None
        # Several reviewed entries can legitimately share a primary symbol
        # (immunoglobulin-like families); UniProt returns them in relevance
        # order, so take the first rather than imposing our own tie-break.
        best = exact[0]
        seq = best.get("sequence", {}).get("value", "")
        if not seq:
            return None
        desc = (
            best.get("proteinDescription", {})
            .get("recommendedName", {})
            .get("fullName", {})
            .get("value", "")
        )
        return ProteinRecord(
            gene=gene,
            accession=best.get("primaryAccession", ""),
            entry_name=best.get("uniProtkbId", ""),
            protein_name=desc,
            sequence=seq,
            length=len(seq),
        )

    def fetch_many(self, genes, progress=True) -> dict[str, ProteinRecord]:
        out: dict[str, ProteinRecord] = {}
        genes = list(genes)
        for i, gene in enumerate(genes, 1):
            rec = self.fetch(gene)
            if rec is not None:
                out[gene] = rec
            if progress and (i % 25 == 0 or i == len(genes)):
                print(f"  UniProt {i}/{len(genes)} resolved={len(out)}", flush=True)
        return out
