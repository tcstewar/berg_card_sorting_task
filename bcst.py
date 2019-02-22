import nengo
import nengo_spa as spa

D = 32

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
    def stim_attention(t):
        options = ['COLOR', 'SHAPE']
        index = int(t / 0.5)
        return options[index % len(options)]
    stim_attention = spa.Transcode(stim_attention, output_vocab=vocab, label='stim_attention')
    stim_attention >> attention
    
    feature = spa.State(D, label='feature')
    card * ~attention >> feature
    
    choice = spa.State(D, label='choice')
    piles * ~feature >> choice
    
    
    
    
    
    