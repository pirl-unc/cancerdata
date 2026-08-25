# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tissue definitions and HPA confidence thresholds for CTA restriction analysis.

Ported (HPA-only) from ``tsarina.tissues`` so oncoref can regenerate the
bundled ``cancer-testis-antigens.csv`` restriction/filter columns from HPA data
alone. The MS-related constants and the CTA-universe membership rules (the
gene-family exclusion and the never-expressed rescue) are intentionally NOT
here — they live in :mod:`oncoref.cta`.

Three tiers of reproductive tissue sets determine how strictly a gene's
expression must be confined to qualify as a cancer-testis antigen:

- **Core**: testis, ovary, placenta -- the classic CT restriction definition.
- **Extended**: core + other reproductive-tract tissues (cervix, endometrium,
  epididymis, fallopian tube, prostate, seminal vesicle, vagina).
- **Permissive**: extended + breast (including lactating breast).

Thymus is excluded from all restriction checks because AIRE-mediated expression
in medullary thymic epithelial cells (mTECs) is expected for CTAs and does not
indicate somatic tissue leakage.
"""

from __future__ import annotations

# ── Tissue sets ─────────────────────────────────────────────────────────────

CORE_REPRODUCTIVE_TISSUES: frozenset[str] = frozenset({"ovary", "placenta", "testis"})

EXTENDED_REPRODUCTIVE_TISSUES: frozenset[str] = CORE_REPRODUCTIVE_TISSUES | frozenset(
    {
        "cervix",
        "endometrium",
        "epididymis",
        "fallopian tube",
        "prostate",
        "seminal vesicle",
        "vagina",
    }
)

PERMISSIVE_REPRODUCTIVE_TISSUES: frozenset[str] = EXTENDED_REPRODUCTIVE_TISSUES | frozenset(
    {"breast", "lactating breast"}
)

#: Tissues excluded from "somatic" calculations: all reproductive tissues plus
#: thymus (AIRE-driven CTA expression is expected, not somatic leakage).
NON_SOMATIC_TISSUES: frozenset[str] = PERMISSIVE_REPRODUCTIVE_TISSUES | frozenset({"thymus"})

#: All tissues considered reproductive for the protein (IHC) restriction call.
#: HPA uses "endometrium 1"/"endometrium 2" in the normal_tissue table, so those
#: split labels are included alongside the consensus "endometrium".
ALL_REPRODUCTIVE_TISSUES: frozenset[str] = NON_SOMATIC_TISSUES | frozenset(
    {"endometrium 1", "endometrium 2"}
)


# ── Adaptive HPA protein → RNA thresholds ──────────────────────────────────

#: Mapping from HPA antibody reliability tier to the minimum deflated RNA
#: reproductive fraction required for a gene to pass the adaptive confidence
#: filter. Higher-confidence protein data allows a more relaxed RNA threshold.
HPA_ADAPTIVE_PROTEIN_RNA_THRESHOLDS: dict[str, float] = {
    "Enhanced": 0.80,
    "Supported": 0.90,
    "Approved": 0.95,
    "Uncertain": 0.97,
    "Missing": 0.97,
}

#: RNA expression floor (nTPM) for the ``never_expressed`` flag. A gene with no
#: HPA protein (IHC) data and a maximum RNA nTPM below this value across all
#: tissues is flagged ``never_expressed`` -- it passes the reproductive-
#: restriction filter only because the ``+1`` deflation pseudocount yields a 1.0
#: fraction when every tissue is below 1 nTPM, but HPA lacks the signal to
#: confirm restriction.
HPA_EXPRESSION_FLOOR_NTPM: float = 2.0


# ── Protein reliability ordering ────────────────────────────────────────────

#: HPA antibody reliability tiers, ordered from strongest to weakest.
PROTEIN_RELIABILITY_ORDER: list[str] = [
    "Enhanced",
    "Supported",
    "Approved",
    "Uncertain",
]

#: HPA IHC levels that count as "protein detected" in a tissue.
PROTEIN_DETECTED_LEVELS: frozenset[str] = frozenset({"Low", "Medium", "High"})


# ── Safety-critical tissue groups ──────────────────────────────────────────

#: Safety-critical tissue groups with nTPM threshold. Genes with max nTPM >=
#: threshold in any tissue in the group get flagged.
#:
#: The ``brain`` group spans two HPA vocabularies on purpose. Ten names are
#: present in the bulk ``rna_tissue_consensus`` and so drive the RNA safety flag
#: (``rna_brain_max_ntpm``): amygdala, basal ganglia, cerebellum, cerebral
#: cortex, choroid plexus, hippocampal formation, hypothalamus, midbrain,
#: retina, spinal cord. Four names (medulla oblongata, pons, thalamus, white
#: matter) come from HPA's finer brain-specific dataset; they are absent from
#: the consensus -- inert for the RNA flag -- but are kept for vocabulary parity
#: with tsarina. These are conceptual/RNA safety labels, not an IHC selection
#: vocabulary; use ``hpa.resolve_safety_tissue_group`` before matching an HPA
#: normal-tissue source.
SAFETY_TISSUE_GROUPS: dict[str, set[str]] = {
    "brain": {
        "amygdala",
        "basal ganglia",
        "cerebellum",
        "cerebral cortex",
        "choroid plexus",
        "hippocampal formation",
        "hypothalamus",
        "medulla oblongata",
        "midbrain",
        "pons",
        "retina",
        "spinal cord",
        "thalamus",
        "white matter",
    },
    "heart": {"heart muscle"},
    "lung": {"lung"},
    "liver": {"liver"},
    "pancreas": {"pancreas"},
}

#: Default nTPM threshold for safety flags.
SAFETY_NTPM_THRESHOLD: float = 5.0


def adaptive_rna_threshold(protein_reliability: str) -> float:
    """Return the minimum deflated RNA reproductive fraction for a protein tier.

    Unknown labels (incl. "no data" / "Missing") use the strict "Missing" floor.
    """
    normalized = str(protein_reliability).strip().casefold()
    by_casefold = {k.casefold(): k for k in HPA_ADAPTIVE_PROTEIN_RNA_THRESHOLDS}
    key = by_casefold.get(normalized, "Missing")
    return HPA_ADAPTIVE_PROTEIN_RNA_THRESHOLDS[key]
