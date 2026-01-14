import pyspiel
import numpy as np

game = pyspiel.load_game("dark_hex(board_size=3)")
state = game.new_initial_state()

def show(player):
    t = np.array(state.information_state_tensor(player))
    print(f"player {player} info_state size = {t.size}")
    nz = np.flatnonzero(t)
    print(f"nonzeros={len(nz)}", "first few idx:", nz[:20], "first few vals:", t[nz[:20]])

show(0); show(1)
# play a couple of moves and show again:
state.apply_action(state.legal_actions()[0])
show(0); show(1)