import rlcard
from rlcard.games.doudizhu.utils import cards2str

env = rlcard.make('doudizhu')
state, player_id = env.reset()

print(type(env.game.round.seen_cards))
