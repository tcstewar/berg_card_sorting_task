import nengo
import nengo_spa as spa

D = 64

model = spa.Network()
with model:
    piles = spa.State(D, label='piles')
    vocab = piles.vocab
    
    def stim_func(t):
        return 'PILE1*(RED+CIRCLE)+PILE2*(GREEN+STAR)'
    stim_piles = spa.Transcode(stim_func, output_vocab=vocab, label='stim_piles')
    stim_piles >> piles
    
    card = spa.State(D, label='card')
    def stim_card(t):
        return 'RED*COLOR+STAR*SHAPE'
    stim_card = spa.Transcode(stim_card, output_vocab=vocab, label='stim_card')
    stim_card >> card
    
    attention = spa.State(D, label='attention')
    vocab = attention.vocab
    attention_wta = spa.WTAAssocMem(threshold=-0.1, input_vocab=vocab, mapping=['COLOR', 'SHAPE'], 
                                label='attention')
    attention >> attention_wta
    1.5*attention_wta >> attention
    
    with spa.ActionSelection():
        spa.ifmax(0.5, spa.sym('0')>>attention)
        spa.ifmax(0.3, -5*attention_wta >> attention)

    feature = spa.State(D, label='feature')
    card * ~attention >> feature
    
    choice = spa.WTAAssocMem(threshold=0.1, input_vocab=vocab, mapping=['PILE1', 'PILE2'], 
                             label='choice')
    2*piles * ~feature >> choice
    
    
    
    
    
    