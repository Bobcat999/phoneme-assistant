#idk why I needed to do this this is gonna be a very simple program
from g2p_en import G2p

g2p = G2p()

def grapheme_to_phoneme(grapheme):
    unfiltered_phonemes = g2p(grapheme)
    normalized = []
    for ph in unfiltered_phonemes:
        # Remove any digits that might represent stress
        ph_norm = ''.join([ch for ch in ph if not ch.isdigit() and ch != " "])
        if(ph_norm == ""):
            continue
        normalized.append(ph_norm)
    return normalized