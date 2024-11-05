"""Awari game mechanics"""

#
# Based on this earlier alpha-zero-genral Awari implementation:
#   https://github.com/keesverstoep/alpha-zero-general/tree/master/awari
#

from typing import Tuple

import chex
import jax.numpy as jnp
import numpy as np
import pax
import jax
from jax import lax

from games.env import Enviroment
from utils import select_tree

# Array used for implementing the sowing logic, which has
# to skip the pit from which the stones are sown from.
next = jnp.int32([
    # Added extra column 12 so a pass move can "pass through"
    # the regular sowing code without modifying the board:
    # 0  1  2  3  4  5  6  7   8   9  10  11 12
    [ 1, 2, 3, 4, 5, 7, 7, 8,  9, 10, 11, 0, 12, ],
    [ 1, 2, 3, 4, 5, 6, 8, 8,  9, 10, 11, 0, 12, ],
    [ 1, 2, 3, 4, 5, 6, 7, 9,  9, 10, 11, 0, 12, ],
    [ 1, 2, 3, 4, 5, 6, 7, 8, 10, 10, 11, 0, 12, ],
    [ 1, 2, 3, 4, 5, 6, 7, 8,  9, 11, 11, 0, 12, ],
    [ 1, 2, 3, 4, 5, 6, 7, 8,  9, 10,  0, 0, 12, ],
])

# Currently uses the following simple board representation:
# - pit[0:6]  : regular pits by player0 (canonical: the one to move)
# - pit[6:12] : regular pits by player1 (canonical: the one not to move)
# - pit[12] :   an empty pit for the sowing logic in case of passing move
# - pit[17]:    home pit of player0
# - pit[23]:    home pit of player1
# Other pit positions are unused.

PIT_MAX = 24
PIT_HOME_0 = 17
PIT_HOME_1 = 23

NUM_STONES = 48

class AwariGame(Enviroment):
    """Awari game environment"""

    board: chex.Array
    who_play: chex.Array
    terminated: chex.Array
    terminated_count: chex.Array
    terminated_invalid: chex.Array
    terminated_pass: chex.Array
    winner: chex.Array
    num_cols: int = PIT_MAX
    num_rows: int = 1

    def __init__(self, num_cols: int = PIT_MAX, num_rows: int = 1):
        super().__init__()
        self.reset()

    def num_actions(self):
        # for awari 6 own pits to choose from, plus one pass if no stones left
        return 7

    def invalid_actions(self) -> chex.Array:
        # for awari can only select pit that has stones
        # - may only select pit to sow that eradicates opponent
        #   when that is the only move left
        # For this we will have have to try all moves:
        # - if some them leaves the opponent with a move and others
        #   do not, only the ones that do are valid
        _board = self.board
        x0 = lax.slice_in_dim(_board, 0, 6)
        x1 = lax.slice_in_dim(_board, 6, 12)
        x2 = lax.slice_in_dim(_board, 12, 18)
        x3 = lax.slice_in_dim(_board, 18, 24)
        _mirror = jax.lax.concatenate([x1, x0, x3, x2], 0)

        pieces = jnp.where(self.thisplayer, _board, _mirror)
        child = jnp.where(self.thisplayer, _mirror, _board)

        remaining_other = jnp.int32([0, 0, 0, 0, 0, 0])

        def cond1(state):
            # i, _, _ = state
            i, _ = state
            return (i < 6)

        def body1(state):
            i, rem_other = state
            newboard = self.sow_canonical(pieces, child, i)
            # returns *mirrored* canonical board, so look at 0..5 for remaining opp stones
            rem_other = rem_other.at[i].set(lax.slice_in_dim(newboard, 0, 6).sum())
            return (i + 1, rem_other)

        state = tuple([0, remaining_other])
        lasti, remaining_other = lax.while_loop(cond_fun=cond1, body_fun=body1, init_val=state)

        remainingall_other = lax.slice_in_dim(remaining_other, 0, 6).sum()

        # regular invalidations, reserve space for pass:
        invalidthis = (pieces[0:7] == 0)

        # if thisplayer and remainingall other is zero, then any nonempty pit is fine
        # if thisplayer and some other remaining pits are zero, these moves are invalid

        def cond2(state):
            i, _ = state
            return (i < 6)
        
        def body2(state):
            i, invalidthis_loc = state
            this_move_invalid_loc = jnp.logical_and(remainingall_other > 0, remaining_other[i] == 0)
            invalidthis_loc = invalidthis_loc.at[i].set(jnp.logical_or(invalidthis_loc[i], this_move_invalid_loc))
            return (i + 1, invalidthis_loc)

        state2 = tuple([0, invalidthis])
        lasti, invalidthis = lax.while_loop(cond_fun=cond2, body_fun=body2, init_val=state2)

        # allow pass when no moves
        invalid = invalidthis
        invalid1 = invalid.at[6].set(True)
        invalid2 = invalid1.at[6].set(jnp.logical_not(jnp.all(invalid1)))

        return invalid2

    def reset(self):
        # Initially all regular pits have 4 stones, and the home pits are empty
        self.board = jnp.int32([4, 4, 4, 4, 4, 4,
                                4, 4, 4, 4, 4, 4,
                                0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0])
        self.who_play = jnp.array(1, dtype=jnp.int32)
        self.thisplayer = jnp.array(1, dtype=jnp.bool_)
        self.otherplayer = jnp.array(0, dtype=jnp.bool_)
        self.terminated = jnp.array(0, dtype=jnp.bool_)
        self.terminated_count = jnp.array(0, dtype=jnp.bool_)
        self.terminated_invalid = jnp.array(0, dtype=jnp.bool_)
        self.terminated_pass = jnp.array(0, dtype=jnp.bool_)
        self.winner = jnp.array(0, dtype=jnp.int32)
        self.count = jnp.array(0, dtype=jnp.int32)

    def sow_canonical(self, pieces: chex.Array, child: chex.Array, move: int):
        pit1 = move
        seeds = pieces[pit1]
        pit2 = pit1 + 6
        capture = 0
        oldseeds = seeds
        next_ptr = next[pit1]

        # take away seeds
        pieces = child.at[pit2].set(0)

        # sow them
        def cond1(state):
            seeds_cur, pit2_cur, pieces_cur = state
            return (seeds_cur > 0)

        def body1(state):
            seeds_cur, pit2_cur, pieces_cur = state
            pit2_cur = next_ptr[pit2_cur]
            temp = jnp.add(pieces_cur[pit2_cur], 1)
            pieces2 = pieces_cur.at[pit2_cur].set(temp)
            return (seeds_cur - 1, pit2_cur, pieces2)

        state = tuple([seeds, pit2, pieces])
        seeds, pit2, pieces2 = lax.while_loop(cond_fun=cond1, body_fun=body1, init_val=state)
        pieces = pieces2

        # deal with captures
        def cond2(state):
            pit2_cur, captures_cur, pieces_cur = state
            ok1 = (pit2_cur < 6)
            ok2 = (pit2_cur >= 0)
            ok3 = (pieces_cur[pit2_cur] == 2)
            ok4 = (pieces_cur[pit2_cur] == 3)
            return jnp.logical_and(ok1, jnp.logical_and(ok2, jnp.logical_or(ok3, ok4)))

        def body2(state):
            pit2_cur, captures_cur, pieces_cur = state
            captures_cur = jnp.add(captures_cur, pieces_cur[pit2_cur])
            pieces_new = pieces_cur.at[pit2_cur].set(0)
            return (pit2_cur - 1, captures_cur, pieces_new)

        captures = 0
        state = tuple([pit2, captures, pieces])
        pit2_new, captures_new, pieces_new = lax.while_loop(cond_fun=cond2, body_fun=body2, init_val=state)
        captures = captures_new

        # canonical board, so captures are for current player orientation,
        # but the resulting board is mirrored, so we need to use the other home pit here
        ret_pieces = pieces_new.at[PIT_HOME_1].set(jnp.add(pieces_new[PIT_HOME_1], captures))
        return ret_pieces

    def sow(self, pieces: chex.Array, child: chex.Array, move: int):
        pieces_new = self.sow_canonical(pieces, child, move)

        # now return the board in the standard player=1  orientation
        # Note that sowing left them in a mirrored bord
        x0 = lax.slice_in_dim(pieces_new, 0, 6)
        x1 = lax.slice_in_dim(pieces_new, 6, 12)
        x2 = lax.slice_in_dim(pieces_new, 12, 18)
        x3 = lax.slice_in_dim(pieces_new, 18, 24)
        pieces_mirror = jax.lax.concatenate([x1, x0, x3, x2], 0)
        ret_pieces = jnp.where(self.thisplayer, pieces_mirror, pieces_new)
        return ret_pieces

    @pax.pure
    def step(self, action: chex.Array) -> Tuple["AwariGame", chex.Array]:
        """One step of the game.

        An invalid move will terminate the game with reward -1.
        """

        # need checks for passing, it's only allowed when no other options exist
        pass_move = (action == 6)

        _board = self.board
        x0 = lax.slice_in_dim(_board, 0, 6)
        x1 = lax.slice_in_dim(_board, 6, 12)
        x2 = lax.slice_in_dim(_board, 12, 18)
        x3 = lax.slice_in_dim(_board, 18, 24)
        _mirror = jax.lax.concatenate([x1, x0, x3, x2], 0)
        pieces = jnp.where(self.thisplayer, _board, _mirror)
        child = jnp.where(self.thisplayer, _mirror, _board)

        # Check for invalid moves: only allowed to select pit that has seeds
        invalid_move = jnp.logical_and(action < 6, pieces[action] == 0)

        # Check if a passing is valid:
        invalid_pass0 = jnp.logical_and(self.thisplayer, jnp.logical_and(pass_move, jnp.any(x0)))
        invalid_pass1 = jnp.logical_and(self.otherplayer, jnp.logical_and(pass_move, jnp.any(x1)))
        invalid_pass = jnp.logical_or(invalid_pass0, invalid_pass1)
        invalid_move = jnp.logical_or(invalid_move, invalid_pass)

        # - check if move that leaves the opponent no stones is the only option,
        #   otherwise it is invalid

        # trick to skip the sowing code when doing a pass move: pit 12 is always empty
        move = jnp.where(pass_move, 12, action)
        
        newboard = self.sow(pieces, child, move)

        # after (forced) pass move, all remaining seeds go to the opponent
        pass_move = jnp.where(invalid_move, False, pass_move)
        remaining_pieces = lax.slice_in_dim(newboard, 0, 12)
        remaining_pieces_count = remaining_pieces.sum()
        remaining_pits = jnp.int32([0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0])
        remaining_pits0 = remaining_pits.at[PIT_HOME_0].set(newboard[PIT_HOME_0])
        remaining_pits1 = remaining_pits0.at[PIT_HOME_1].set(newboard[PIT_HOME_1])

        newboard_this_player = select_tree(jnp.logical_and(pass_move, self.thisplayer),
                remaining_pits1.at[PIT_HOME_1].add(remaining_pieces_count), newboard)
        newboard_other_player = select_tree(jnp.logical_and(pass_move, self.otherplayer),
                remaining_pits1.at[PIT_HOME_0].add(remaining_pieces_count), newboard)
        newboard_after_pass = select_tree(self.thisplayer, newboard_this_player, newboard_other_player)

        self.board = select_tree(self.terminated, self.board, newboard)
        self.board = select_tree(pass_move, newboard_after_pass, self.board)

        # set winner if one player has a majority of stones
        self.winner = 0
        winner0 = (self.board[PIT_HOME_0] > NUM_STONES / 2)
        winner1 = (self.board[PIT_HOME_1] > NUM_STONES / 2)
        self.winner = jnp.where(winner0, 1, self.winner)
        self.winner = jnp.where(winner1, -1, self.winner)
        reward_ = self.winner * self.who_play

        # end turn updating players, move counters and termination status
        self.who_play = -self.who_play
        self.thisplayer = jnp.logical_not(self.thisplayer)
        self.otherplayer = jnp.logical_not(self.otherplayer)
        self.count = self.count + 1
        # termination due to exceeding move limit, causing draw:
        self.terminated_count = (self.count >= self.max_num_steps())
        self.terminated = jnp.logical_or(self.terminated, self.count >= self.max_num_steps())
        # termination due to win/loss
        self.terminated = jnp.logical_or(self.terminated, reward_ != 0)
        # termination after pass due to no move left
        self.terminated_pass = (pass_move)
        self.terminated = jnp.logical_or(self.terminated, pass_move)
        # termination due to invalid move, which also causes negative reward
        self.terminated_invalid = (invalid_move)
        self.terminated = jnp.logical_or(self.terminated, invalid_move)
        reward_ = jnp.where(invalid_move, -1.0, reward_)
        return self, reward_

    def render(self) -> None:
        """Render the game on screen."""
        board = self.observation()
        for row in reversed(range(self.num_rows)):
            for col in range(self.num_cols):
                print(board[row, col].item(), end=" ")
                if col == 5 or col == 11:
                    print("", end="  ")
            print()
        print()
        print("who_play", self.who_play)
        print("thisplayer", self.thisplayer)
        print("otherplayer", self.otherplayer)
        print("terminated", self.terminated)
        print("terminated_count", self.terminated_count)
        print("terminated_invalid", self.terminated_invalid)
        print("terminated_pass", self.terminated_pass)
        print("winner", self.winner)
        print("count", self.count)

    def observation(self) -> chex.Array:
        ret_board = self.board
        return jnp.reshape(ret_board, ret_board.shape[:-1] + (self.num_rows, self.num_cols))

    def canonical_observation(self) -> chex.Array:
        _board = self.board
        x0 = lax.slice_in_dim(_board, 0, 6)
        x1 = lax.slice_in_dim(_board, 6, 12)
        x2 = lax.slice_in_dim(_board, 12, 18)
        x3 = lax.slice_in_dim(_board, 18, 24)
        _mirror = jax.lax.concatenate([x1, x0, x3, x2], 0)
        ret_board = jnp.where(self.otherplayer, _mirror, _board)
        return jnp.reshape(ret_board, ret_board.shape[:-1] + (self.num_rows, self.num_cols))

    def is_terminated(self):
        return self.terminated

    def is_terminated_special(self):
        return jnp.logical_or(self.terminated_pass, self.terminated_invalid)

    def max_num_steps(self) -> int:
        # protect against endless game
        # TODO: correct would be to check for exact repetition of game positions
        return 150

    def symmetries(self, state, action_weights):
        # For awari no additional symmetries
        out = [(state, action_weights)]
        return out
