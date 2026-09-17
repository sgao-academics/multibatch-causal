"""Gene-symbol handling shared by the figure scripts.

Two of the genes the inferred networks nominate are carried under legacy symbols, and both are
spelled differently again in the TCGA/Xena matrices:

    legacy                      HGNC approved   HGNC         Xena HiSeqV2 writes
    --------------------------  --------------  -----------  ----------------------
    C9orf84  (HGNC previous)    SHOC1           HGNC:26535   C9orf84   (same, lower orf)
    MGC29506 (HGNC alias)       MZB1            HGNC:30125   MGC29506  (alias form)

A plain `symbol in gene_set` test fails twice over: once on the rename and once on case, because
Xena spells the unnamed-reading-frame loci "C11orf86" where the network tables use "C11ORF86".
Either way the row is dropped without a warning, which is how two hub genes came to be missing
from the tissue-specificity statistics.

`canonical()` maps any spelling to the name the paper prints. `index()` builds the case-insensitive
lookup used when a Xena row name has to be tied back to a gene of a network.
"""

# First entry of each group is the HGNC-approved symbol, i.e. the one the manuscript prints.
#
# Only these two loci are renamed. The unnamed-reading-frame genes (C11ORF86 and 51 others) differ
# from Xena by case alone, not by symbol: they keep the network table's spelling and are matched by
# the case-insensitive lookups below, because rewriting them would touch every gene list and every
# edge in the paper for a change no database cares about.
GROUPS = (
    ('SHOC1', 'C9ORF84', 'C9orf84'),
    ('MZB1', 'MGC29506'),
    # Three rows of the Xena HiSeqV2 matrix carry no symbol at all: the annotation falls back to
    # "?" followed by the Entrez identifier, so the name reaches the edge tables as "?|729884" and
    # friends. The identifiers resolve as follows (Entrez, checked 2026-09-17):
    #   729884 -> TMPRSS11E   (discontinued id; current 28983, alias TMPRSS11E2)
    #   340602 -> EZHIP       (alias CXorf67)
    #   652919 -> RGPD7       (record withdrawn by HGNC; no current approved symbol)
    # Renaming them changes no count: none of the three collides with a name already in the
    # 1,775-gene panel, so the 12 edges that touch them keep their gene pairs.
    ('TMPRSS11E', '?|729884'),
    ('EZHIP', '?|340602'),
    ('RGPD7', '?|652919'),
)

CANONICAL = {}
for _group in GROUPS:
    for _spelling in _group:
        CANONICAL[_spelling] = _group[0]
        CANONICAL[_spelling.upper()] = _group[0]
del _group, _spelling


def canonical(symbol):
    """The HGNC-approved symbol for `symbol`, or `symbol` itself when it is not a legacy alias."""
    return CANONICAL.get(symbol) or CANONICAL.get(symbol.upper()) or symbol


def index(symbols):
    """Case-insensitive lookup: upper-cased spelling -> canonical symbol.

    Use this when the result is a dictionary key of your own making, e.g. when a Xena row has to be
    filed under a network gene. Xena's "C9orf84" resolves to "SHOC1" and its "C11orf86" to the
    network table's "C11ORF86".
    """
    idx = {}
    for s in symbols:
        c = canonical(s)
        idx.setdefault(c.upper(), c)
    return idx


def labels(container):
    """Case-insensitive lookup into a container's own labels: upper-cased spelling -> the label.

    Use this when the value has to be passed back to the container it came from, as `df.loc[key]`
    needs: the label must be the spelling that container actually carries ("C9orf84"), not the
    canonical one.
    """
    idx = {}
    for s in container:
        idx.setdefault(canonical(str(s)).upper(), s)
    return idx
