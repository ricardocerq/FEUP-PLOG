:-use_module(library(random)).
:-use_module(library(lists)).
:- dynamic playing/2.
%----------------------------------UTILS
list_of([],0, _X):-!.
list_of([X|Xs], N, X) :- N1 is N - 1,  list_of(Xs, N1, X).

list_get([E|_Xs], I, I, E).
list_get([_X|Xs], I, T, E):- T1 is T+1, list_get(Xs, I, T1, E).
list_get(L, I, E):- list_get(L, I, 0, E).

list_set([_X|Xs], 0, E, [E|Xs]):-!.
list_set([X|Xs], I, E, [X|Ys]) :-  I1 is I - 1, list_set(Xs, I1, E, Ys).

matrix_get(Matrix, X, Y, E) :- list_get(Matrix, Y, L), list_get(L, X, E).

matrix_set(Matrix, X, Y, E, NewMatrix) :- list_get(Matrix, Y, L), list_set(L, X, E, L2), list_set(Matrix, Y, L2, NewMatrix). %228.92

in_bounds(X, Y, S):- X>=0, Y>=0, X<S, Y<S.

max_of(inf, _X, inf):-!.
max_of(_X, inf, inf):-!.
max_of(minf, X, X):-!.
max_of(X, minf, X):-!.

max_of(X, Y, X):- X >= Y, !.
max_of(X, Y, Y):- Y > X .


min_of(inf, X, X):-!.
min_of(X, inf, X):-!.
min_of(minf, _X, minf):-!.
min_of(_X, minf, minf):-!.


min_of(X, Y, X):- X =< Y, !.
min_of(X, Y, Y):- Y < X .

greater_than(minf, minf):-!.
greater_than(inf, _X):-!.
greater_than(_X, inf):-!, fail.
greater_than(minf, _X):-!, fail.
greater_than(_X, minf):-!.
greater_than(X, Y):- X > Y.

less_than(minf, _X):-!.
less_than(_X, minf):-!, fail.
less_than(inf, _X):-!, fail.
less_than(_X, inf):-!.
less_than(X, Y):- X < Y.

switch_op(l_e, g_e).
switch_op(g_e, l_e).

for(I, J, I):- J >= I. 
for(I, J, K) :- I < J, I1 is I + 1, for(I1, J, K).    


list_length([], T, T).
list_length([_L|Ls], T , X):- T1 is T + 1,  list_length(Ls, T1, X). 
list_length(Ls, X) :- list_length(Ls, 0, X). 

matrix_height(Matrix, H) :-list_length(Matrix, H).
matrix_width(Matrix, W) :- list_get(Matrix, 0, X), list_length(X, W).

%----------------------------------Board
no_piece('.').
swap_piece('x', 'o').
swap_piece('o', 'x').
swap_piece('X', 'O').
swap_piece('O', 'X').
normal_piece('x').
normal_piece('o').
king_of('O', 'o').
king_of('X', 'x').

set_upGame(N, G):- build_board(N, M), gen_board(M, G).

build_board(N, B) :- no_piece(Empty), list_of(L, N, Empty), list_of(B, N, L).

gen_board(M, G):-  matrix_width(M, W), matrix_height(M, H), initial_piece_number(W, N), gen_X(M, 0, G1, W, H, N, N, Ocount, Xcount), gen_Y(G1, 1, G, W, H, Ocount, Xcount,  _ , _).

initial_piece_number(S, N):- N is (S * 2 + (S-2)*2)//2.

gen_X(G, X, G, W, _H, Ocount, Xcount, Ocount, Xcount):-  X =:= W. 
gen_X(M, X, G, W, H, Ocount, Xcount, FinalOcount, FinalXcount) :-  X < W, gen_piece(M, X, 0, W, H, G1, Ocount, Xcount, Ocount2, Xcount2), X1 is X+1, gen_X(G1, X1, G, W, H, Ocount2, Xcount2, FinalOcount, FinalXcount).

gen_Y(G, Y, G, _W, H, Ocount, Xcount,Ocount,Xcount):- Y =:= H-1. 
gen_Y(M, Y, G, W, H, Ocount, Xcount,  FinalOcount, FinalXcount) :- Y < H-1, gen_piece(M, 0, Y, W, H, G1, Ocount, Xcount, Ocount2, Xcount2), Y1 is Y+1, gen_Y(G1, Y1, G, W, H,Ocount2, Xcount2, FinalOcount, FinalXcount).

gen_piece(M, X, Y, W, H, NewM, 0, Xcount, 0, Xcount2) :- set_x_piece(M, X, Y, W, H, NewM, 0, Xcount, 0, Xcount2).
gen_piece(M, X, Y, W, H, NewM, Ocount, 0, Ocount2, 0) :- set_o_piece(M, X, Y, W, H, NewM, Ocount, 0, Ocount2, 0).
gen_piece(M, X, Y, W, H, NewM, Ocount, Xcount, Ocount2, Xcount2) :- Ocount > 0, Xcount > 0, random(0, 2, N), if(N =:= 0, set_x_piece(M, X, Y, W, H, NewM, Ocount, Xcount, Ocount2, Xcount2), set_o_piece(M, X, Y, W, H, NewM, Ocount, Xcount, Ocount2, Xcount2)).

set_x_piece(M, X, Y, W, H, NewM, Ocount, Xcount, Ocount2, Xcount2):- X1 is W-X-1, Y1 is H-Y-1, matrix_set(M, X, Y, 'x', Temp), matrix_set(Temp, X1, Y1, 'o', NewM), Xcount2 is Xcount - 1, Ocount2 = Ocount.
set_o_piece(M, X, Y, W, H, NewM, Ocount, Xcount, Ocount2, Xcount2):- X1 is W-X-1, Y1 is H-Y-1, matrix_set(M, X, Y, 'o', Temp), matrix_set(Temp, X1, Y1, 'x', NewM), Ocount2 is Ocount - 1, Xcount2 = Xcount.


%----------------------------------Input
                                                                                    %lowercase                              %uppercase
convert_letter(M1, X):- name(M1, L), list_length(L, 1), list_get(L, 0, N), if((N =< 122, N >= 97 ), X is N - 97, if((N =< 90, N >= 65 ), X is N - 65, false)).

convert_number(M1, Y):- Y is M1 - 1.

process_pos(Pos, X, Y):- name(Pos, [L|Ns]), name(Latom, [L]), convert_letter(Latom, X), number_codes(N, Ns), convert_number(N, Y). 


get_move(Validator, [X1, Y1], [X2, Y2]):- repeat, read(Pos1), read(Pos2), process_pos(Pos1, X1, Y1), process_pos(Pos2, X2, Y2), if(Validator, true , (print('Invalid Move!!'), nl, fail)).

%----------------------------------Logic
valid_move(Matrix, [X1, Y1], [X2, Y2], Team):-  matrix_get(Matrix, X1, Y1, Team), 
                                                matrix_height(Matrix, Size),
                                                move_pattern([X1, Y1],[X2, Y2], Size, P),
                                                unblocked(Matrix, [X1, Y1], [X2, Y2], P),
                                                distance_to_center(Size, [X1, Y1], L1), 
                                                distance_to_center(Size, [X2, Y2], L2), 
                                                L1 > L2, L2 =\= 0. 

move_pattern_aux([X2, Y2], [X2, Y2], Size, [_DirX, _DirY]):- X2 < Size, X2 >= 0, Y2 < Size, Y2 >= 0.
move_pattern_aux([X1, Y1], [X2, Y2], Size, [DirX, DirY]):- X1 < Size, X1 >= 0, Y1 < Size, Y1 >= 0, X3 is X1 + DirX, Y3 is Y1 + DirY, move_pattern_aux([X3, Y3], [X2, Y2], Size, [DirX, DirY]) .
move_pattern([X1,Y1],[X2, Y2], Size, [DirX, DirY]):- for(-1, 1, DirX), for(-1, 1, DirY),(DirX + 2*DirY =\= 0), Xi is X1 + DirX, Yi is Y1 + DirY, move_pattern_aux([Xi, Yi], [X2, Y2], Size, [DirX, DirY]).

current_winner(Matrix, Team):- matrix_height(Matrix, S), S2 is S // 2, matrix_get(Matrix, S2, S2, Piece), if(no_piece(Piece), Team='none', king_of(Piece, Team)).

%check to see if there is a piece in the way
unblocked_aux(Matrix, [X2, Y2],[X2, Y2], _P):- matrix_get(Matrix, X2, Y2, Piece), no_piece(Piece), !.
unblocked_aux(Matrix, [X1, Y1], [X2, Y2], [X, Y]):- /*X1 \= X2, Y1 \= Y2,*/ matrix_get(Matrix, X1, Y1, Piece), no_piece(Piece), X3 is X1 + X, Y3 is Y1 + Y, unblocked_aux(Matrix, [X3,Y3],[X2, Y2] , [X, Y]).

unblocked(Matrix, [X1, Y1], [X2, Y2], [X, Y]) :- X3 is X1 + X, Y3 is Y1 + Y, unblocked_aux(Matrix, [X3, Y3], [X2, Y2], [X, Y]).

possible_moves(Board, Team, List):- findall([I,F], ( matrix_get(Board, Xi, Yi, Team), I = [Xi, Yi], valid_move(Board, I, F, Team), I \= F), List).

game_not_over(Board, Team):- matrix_get(Board, Xi, Yi, Team), valid_move(Board, [Xi, Yi], [_Xf, _Yf], Team).


throne(Matrix, _, [], _, Matrix).
throne(Matrix, Size, [M1|Modified], Team, NewMatrix):- M1 = [Xi, Yi], S2 is Size//2, 
                                                  X is Xi-S2, Y is Yi-S2,
                                                  X1 is S2-X, Y1 is S2-Y, 
                                                  X2 is S2+Y, Y2 is S2-X, 
                                                  X3 is S2-Y, Y3 is S2+X,
                                                  if((matrix_get(Matrix, X1, Y1, Team), matrix_get(Matrix, X2, Y2, Team), matrix_get(Matrix, X3, Y3, Team)),
                                                     (place_king(Matrix, Size, Team , NewMatrix), !),
                                                     throne(Matrix, Size, Modified, Team, NewMatrix)).

place_king(Matrix, Size, Team, NewMatrix):- S2 is Size // 2, king_of(King, Team), matrix_set(Matrix, S2, S2, King, NewMatrix).

commit_move(Matrix, Move, Team, NewM):-               no_piece(None),
                                                      Move = [[X1, Y1], [X2, Y2]],
                                                      matrix_set(Matrix, X1, Y1, None, Temp1), 
                                                      matrix_set(Temp1, X2, Y2, Team, Temp2), 
                                                      matrix_height(Matrix, Size),
                                                      findall([X, Y], (for(-1,1,X), for(-1,1, Y), \+((X=:=0, Y=:=0))), CaptureDirections),
                                                      calculate_captures(Temp2, Size, [X2, Y2], Team, CaptureDirections, [], Modified),
                                                      swap_pieces(Temp2, Modified, Team, Temp3),
                                                      throne(Temp3, Size, [[X2, Y2]|Modified], Team, NewM).

swap_pieces(Matrix, [], _, Matrix).
swap_pieces(Matrix, [[X, Y]|Modified],Team, NewM):- matrix_set(Matrix, X, Y, Team, Temp1), swap_pieces(Temp1, Modified, Team, NewM).
calculate_captures(_Matrix, _Size, _XY, _Team, [], Modified, Modified).
calculate_captures(Matrix, Size, [X, Y], Team, [[DirX, DirY]|Dirs], ModifiedAcc, Modified):- X1 is X + DirX, 
                                                                           Y1 is Y + DirY, 
                                                                           X2 is X + 2*DirX, 
                                                                           Y2 is Y + 2*DirY,
                                                                           swap_piece(Team, OtherTeam),
                                                                           if((in_bounds(X2, Y2, Size), matrix_get(Matrix, X1, Y1, OtherTeam), matrix_get(Matrix, X2, Y2, Team)),
                                                                             (append(ModifiedAcc, [[X1, Y1]], ModifiedAcc2), calculate_captures(Matrix, Size, [X,Y], Team, Dirs, ModifiedAcc2, Modified)),
                                                                             calculate_captures(Matrix, Size, [X,Y], Team, Dirs, ModifiedAcc, Modified)
                                                                           ).
calculate_captures(Matrix, Size, [X, Y], Team, ModifiedAcc, Modified) :-  findall([X, Y], (for(-1,1,X), for(-1,1, Y), \+((X=:=0, Y=:=0))), CaptureDirections), calculate_captures(Matrix, Size, [X, Y], Team, CaptureDirections, ModifiedAcc, Modified).


distance_to_center(Size, [X, Y], L):- S2 is Size // 2, L is max(abs(S2- X), abs(S2- Y)).
                          


%----------------------------------AI
%-------Board evaluation
update_counts(1, C1, NC1, C2, NC2, C3, NC3):- NC1 is C1+1, NC2 is C2, NC3 is C3, !.
update_counts(2, C1, NC1, C2, NC2, C3, NC3):- NC1 is C1, NC2 is C2+1, NC3 is C3, !.
update_counts(3, C1, NC1, C2, NC2, C3, NC3):- NC1 is C1, NC2 is C2, NC3 is C3+1, !.
update_counts(4, C1, NC1, C2, NC2, C3, NC3):- NC1 is C1, NC2 is C2, NC3 is C3, !.

%count number of incomplete frames
count_2_3(_Board, _Team, [], Count1, Count1, Count2, Count2, Count3, Count3).
count_2_3(Board, Team, [[Xi, Yi]|Others], CurrentCount1, Count1, CurrentCount2, Count2, CurrentCount3, Count3):- matrix_height(Board, Size),
                                                                                          S2 is Size//2, 
                                                                                          X is Xi-S2, Y is Yi-S2,
                                                                                          X1 is S2-X, Y1 is S2-Y, 
                                                                                          X2 is S2+Y, Y2 is S2-X, 
                                                                                          X3 is S2-Y, Y3 is S2+X,
                                                                                           ((matrix_get(Board, X1, Y1, Team), memberchk([X1, Y1], Others))-> (C1 is 1, select([X1, Y1], Others, O1)); (C1 is 0, O1 = Others)),
                                                                                                ((matrix_get(Board, X2, Y2, Team), memberchk([X2, Y2], O1))-> (C2 is 1, select([X2, Y2], O1, O2)); (C2 is 0, O2 = O1)),
                                                                                                ((matrix_get(Board, X3, Y3, Team), memberchk([X3, Y3], O2))-> (C3 is 1, select([X3, Y3], O2, O3)); (C3 is 0, O3 = O2)),
                                                                                                 Count is C1 + C2 + C3 + 1,
                                                                                                 update_counts(Count, CurrentCount1, NextCount1, CurrentCount2, NextCount2, CurrentCount3, NextCount3),
                                                                                                 !,
                                                                                                 count_2_3(Board, Team, O3, NextCount1, Count1, NextCount2, Count2, NextCount3,Count3).

count_2_3(Board, Team, Positions, Count1, Count2, Count3):- count_2_3(Board, Team, Positions, 0, Count1, 0, Count2, 0, Count3).

%assign the board a value
evaluate_board(Board, Team, Score):- findall([Xi,Yi], ( matrix_get(Board, Xi, Yi, Team)), List), 
                                     length(List, N), current_winner(Board, Winner), 
                                     swap_piece(Team, Opponent),
                                     if(Winner=Team, WinScore = 1, if(Winner = Opponent, WinScore = -1, WinScore = 0)), 
                                     count_2_3(Board, Team, List, _, Count2, Count3), 
                                     count_2_3(Board, Opponent, List, _, OCount2, OCount3),
                                     C2 is Count2 - OCount2,
                                     C3 is Count3 - OCount3, 
                                     Score is (N + 50*WinScore + C2 * 5 + C3 * 10).

%-------min max
move_best(_Board, _Team, _Depth, [], Move, V, Move, V).

move_best(Board, Team, Depth, [FirstMove|OtherMoves], CurrentBest, CurrentBestV, Move, V):-     commit_move(Board, FirstMove, Team, NewBoard), !,
                                                                                                D1 is Depth - 1,
                                                                                                swap_piece(Team, NextTeam),
                                                                                                move_aux(NewBoard, NextTeam, D1, _NewMove, NewV),
                                                                                                if(greater_than(NewV, CurrentBestV),
                                                                                                   (NewBest = FirstMove, NewBestV = NewV), 
                                                                                                   (NewBest = CurrentBest, NewBestV = CurrentBestV)),
                                                                                              
                                                                                                !,
                                                                                                move_best(Board, Team, Depth, OtherMoves, NewBest, NewBestV, Move, V).
                                                                                                

move_aux(Board, Team, 0, _Move, V):- evaluate_board(Board, Team, Value), V is -Value.
move_aux(Board, Team, Depth, Move, V) :- possible_moves(Board, Team, List), Extreme = minf, !, move_best(Board, Team, Depth, List, _, Extreme, Move, V).

min_max_move(Board, Team, Depth, Move):- move_aux(Board, Team, Depth, Move, _V).








%--------minmax with alpha beta pruning
alpha_beta_max(Board, OriginalTeam, _Team, _BestMove,_Alpha, _Beta, 0, Return):- evaluate_board(Board, OriginalTeam, Return), !.
alpha_beta_max(Board, OriginalTeam, Team, BestMove, Alpha, Beta, Depth, Return):- possible_moves(Board, Team, List), 
                                                                                  if(List = [], 
                                                                                     (if(current_winner(Board, OriginalTeam), 
                                                                                         Return is 10000000000, 
                                                                                         Return is -10000000000)), 
                                                                                     (!, alpha_beta_max_aux(Board, OriginalTeam, Team, [], Alpha, Beta, List, Depth, Return, BestMove))).

alpha_beta_max_aux(_Board, _OriginalTeam, _Team, FinalMove, Alpha, _Beta, [], _Depth, Alpha, FinalMove).
alpha_beta_max_aux(Board, OriginalTeam, Team, BestMove, Alpha, Beta, [FirstMove|OtherMoves], Depth, Return, FinalMove):-  
                                                                                                               %print('1'),
                                                                                                               commit_move(Board, FirstMove, Team, NextBoard), 
                                                                                                               D1 is Depth - 1, 
                                                                                                               swap_piece(Team, OpponentTeam), 
                                                                                                               %print('2'),
                                                                                                               alpha_beta_min(NextBoard, OriginalTeam, OpponentTeam, _, Alpha, Beta, D1, Score),
                                                                                                               %print('3'),
                                                                                                               if(
                                                                                                                    Score>= Beta, 
                                                                                                                    (Return is Beta, FinalMove = FirstMove), 
                                                                                                                    (if(
                                                                                                                          Score > Alpha, 
                                                                                                                          (Alpha1 is Score, BestMove1 = FirstMove), 
                                                                                                                          (Alpha1 is Alpha, BestMove1 = BestMove )),
                                                                                                                     !,
                                                                                                                     (%print('3'), 
                                                                                                                      alpha_beta_max_aux(Board, OriginalTeam, Team, BestMove1, Alpha1, Beta, OtherMoves, Depth, Return, FinalMove)))).


alpha_beta_min(Board, OriginalTeam, _Team, _BestMove, _Alpha, _Beta, 0, Return):- evaluate_board(Board, OriginalTeam, Return), !.
alpha_beta_min(Board, OriginalTeam, Team, BestMove, Alpha, Beta, Depth, Return):- possible_moves(Board, Team, List),
                                                                                  if(List = [], 
                                                                                     (if(current_winner(Board, OriginalTeam), 
                                                                                         Return is  10000000000, 
                                                                                         Return is -10000000000)), 
                                                                                     (!,alpha_beta_min_aux(Board, OriginalTeam, Team, [], Alpha, Beta, List, Depth, Return, BestMove))).

alpha_beta_min_aux(_Board, _OriginalTeam, _Team, FinalMove, _Alpha, Beta, [], _Depth, Beta, FinalMove).
alpha_beta_min_aux(Board, OriginalTeam, Team, BestMove, Alpha, Beta, [FirstMove|OtherMoves], Depth, Return, FinalMove):-  %print('5'),
                                                                                                               commit_move(Board, FirstMove, Team, NextBoard), 
                                                                                                               D1 is Depth - 1, 
                                                                                                               swap_piece(Team, OpponentTeam),
                                                                                                               %print('6'),
                                                                                                               alpha_beta_max(NextBoard, OriginalTeam, OpponentTeam, _, Alpha, Beta, D1, Score),
                                                                                                               %print('7'),
                                                                                                               if(
                                                                                                                    Score =< Alpha, 
                                                                                                                    (Return is Alpha, FinalMove = FirstMove), 
                                                                                                                    (if(
                                                                                                                          Score < Beta, 
                                                                                                                          (Beta1 is Score, BestMove1 = FirstMove), 
                                                                                                                          (Beta1 is Beta, BestMove1 = BestMove )),
                                                                                                                     !,
                                                                                                                     ( %print('8'),
                                                                                                                       alpha_beta_min_aux(Board, OriginalTeam, Team, BestMove1, Alpha, Beta1, OtherMoves, Depth, Return, FinalMove)))).

min_max_move_alpha_beta(Board, Team, Depth, Move):- alpha_beta_max(Board, Team, Team, Move, -100000000000, 100000000000, Depth, _Value).%, nl, nl, print(Value), nl.


%----------------------------------Draw board
print_cell(Cell):- print(Cell).
board_format("     A   B   C   D   E   F   G   H   I   J   K   L   M\n    --------------------------------------------------- \n 1 | .   .   .   .   .   .   .   .   .   .   .   .   . |  1\n   |    -------------------------------------------    |\n 2 | . | .   .   .   .   .   .   .   .   .   .   . | . |  2\n   |   |    -----------------------------------    |   |\n 3 | . | . | .   .   .   .   .   .   .   .   . | . | . |  3\n   |   |   |    ---------------------------    |   |   |\n 4 | . | . | . | .   .   .   .   .   .   . | . | . | . |  4\n   |   |   |   |    -------------------    |   |   |   |\n 5 | . | . | . | . | .   .   .   .   . | . | . | . | . |  5\n   |   |   |   |   |    -----------    |   |   |   |   |\n 6 | . | . | . | . | . | .   .   . | . | . | . | . | . |  6\n   |   |   |   |   |   |    ---    |   |   |   |   |   |\n 7 | . | . | . | . | . | . | . | . | . | . | . | . | . |  7\n   |   |   |   |   |   |    ---    |   |   |   |   |   |\n 8 | . | . | . | . | . | .   .   . | . | . | . | . | . |  8\n   |   |   |   |   |    -----------    |   |   |   |   |\n 9 | . | . | . | . | .   .   .   .   . | . | . | . | . |  9\n   |   |   |   |    -------------------    |   |   |   |\n10 | . | . | . | .   .   .   .   .   .   . | . | . | . | 10\n   |   |   |    ---------------------------    |   |   |\n11 | . | . | .   .   .   .   .   .   .   .   . | . | . | 11\n   |   |    -----------------------------------    |   |\n12 | . | .   .   .   .   .   .   .   .   .   .   . | . | 12\n   |    -------------------------------------------    |\n13 | .   .   .   .   .   .   .   .   .   .   .   .   . | 13\n    ---------------------------------------------------\n     A   B   C   D   E   F   G   H   I   J   K   L   M     ").

print_board_3([],[]):- print('\n'),print('\n').
print_board_3(Format, [[]|Rest]):-  !, print_board_3(Format, Rest).
print_board_3([46|FormatRest], [[E | RestOfLine]|Rest]):-  print_cell(E), print_board_3(FormatRest, [RestOfLine|Rest]).
print_board_3([FormatFirst| FormatRest], Board):- FormatFirst \= 46, format("~1c", [FormatFirst]), print_board_3(FormatRest, Board).

print_board(Board):- board_format(Format), print_board_3(Format, Board).


%----------------------------------Game Loop

decide_move('Human', Board, Team, Move):- 
                                         Validator =.. [valid_move, Board, [X1, Y1],[X2, Y2], Team],
                                         get_move(Validator, [X1, Y1], [X2, Y2]), Move = [[X1, Y1], [X2, Y2]].

decide_move('Random AI', Board, Team, Move):- possible_moves(Board, Team, List), list_length(List, N), random(0, N, R), list_get(List, R, Move).

decide_move('Greedy AI', Board, Team, Move):- min_max_move(Board, Team, 1, Move).
decide_move('Min Max With Alpha Beta Pruning', Board, Team, Move):- min_max_move_alpha_beta(Board, Team, 2, Move).


play_game(Board, CurrentTeam):- 
                                   game_not_over(Board, CurrentTeam),!,
                                   print_board(Board),!,
                                   print('Playing: '), atom_codes(CurrentTeam, Codes), format("~1c", Codes), print('\n\n'), 
                                   playing(Player, CurrentTeam),
                                   decide_move(Player, Board, CurrentTeam, Move), 
                                   commit_move(Board, Move, CurrentTeam, NextBoard), 
                                   swap_piece(CurrentTeam, NextTeam),
                                   !,
                                   play_game(NextBoard, NextTeam).

play_game(Board, _CurrentTeam):-  print_board(Board), print('Game Over'), nl, current_winner(Board, Team), print('The winner is '), print(Team), nl, nl,!,morelli.

play_game :- set_upGame(13, Board), !, play_game(Board, 'x').


morelli:- user_select_play_mode('x'), user_select_play_mode('o'), !, play_game, retractall(playing(_Player, _Team)).

%----------------------------------Mode Selection
set_mode_option(N, Team):- player_num_assoc(N, Player), asserta(playing(Player, Team)).

player_num_assoc(1, 'Human').
player_num_assoc(2, 'Random AI').
player_num_assoc(3, 'Greedy AI').
player_num_assoc(4, 'Min Max With Alpha Beta Pruning').

print_player_assoc:- \+((player_num_assoc(N, Player), print(N), print(' :- '), print(Player), nl, fail)).

user_select_play_mode(Team):- print('Select Mode for Team '), print(Team), nl, print_player_assoc, repeat, read(X), set_mode_option(X, Team).

%--------Game
:-morelli.

        




















