# FEUP-PLOG
Projects developed for Logic Programming class
Implementation of the board game [Morelli](http://www.boardspace.net/portuguese/about_morelli.html) in Prolog.
The game features the modes Player vs. Player, Player vs. Computer and Computer vs. Computer. For the AI, three difficulty modes are available:
  1 - Random: the computer randomly selects a move from the set of possible moves
  2 - Greedy: the computer selects the move that results in what it considers to be its best possible state, according to its heuristic.
  3 - MiniMax: the MiniMax algorithm is applied to discover the best move, alpha beta pruning is applied to improve the runtime cost of the operation considerably
  Language: Prolog
