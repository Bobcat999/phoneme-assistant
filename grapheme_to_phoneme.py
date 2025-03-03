#idk why I needed to do this this is gonna be a very simple program
from g2p_en import G2p

g2p = G2p()

def grapheme_to_phoneme(grapheme):
    return g2p(grapheme)