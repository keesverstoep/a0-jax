"""
Human/AwariOracle vs AI play
"""

# Based on play.py, but extended with support for an Awari agent that can
# do Awari database lookups.  This agent can be instructed to make a tunable
# average level of mistakes so it can act as an external reference for
# AlphaZero-trained agents.

import pickle
import random
import warnings
from functools import partial
from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
from fire import Fire

from games.env import Enviroment
from tree_search import improve_policy_with_mcts, recurrent_fn
from utils import env_step, import_class, replicate, reset_env

# Awari database support
#
# See the dataset publication "Awari game score database"
#    https://research.vu.nl/en/datasets/awari-game-score-database
#    https://doi.org/10.48338/VU01-11WJKE
# The supplementary code for this dataset is included in subdirectory "awari_score_db"

from awari_score_db.db_board import db_board
from awari_score_db.db_lookup import gen_board_and_child_scores

next = [
    [ 1, 2, 3, 4, 5, 7, 7, 8,  9, 10, 11, 0, ],
    [ 1, 2, 3, 4, 5, 6, 8, 8,  9, 10, 11, 0, ],
    [ 1, 2, 3, 4, 5, 6, 7, 9,  9, 10, 11, 0, ],
    [ 1, 2, 3, 4, 5, 6, 7, 8, 10, 10, 11, 0, ],
    [ 1, 2, 3, 4, 5, 6, 7, 8,  9, 11, 11, 0, ],
    [ 1, 2, 3, 4, 5, 6, 7, 8,  9, 10,  0, 0, ],
    # 0  1  2  3  4  5  6  7   8   9  10  11 
]

class AwariOraclePlayer:
    def __init__(self, env, mistake_fraction = 0.0, mistake_max = 4):
        self.env = env
        self.mistake_fraction = mistake_fraction
        self.mistake_max = mistake_max

    def can_move(self, pieces, player):
        if player == 1:
            for i in range(6):
                if pieces[i] > 0:
                    return True
        else:
            for i in range(6, 12):
                if pieces[i] > 0:
                    return True
        return False

    def sow(self, pieces, move, player, verbose):
        # OLD comment:
        # assumes a canonical board as input and returns a canonical board

        pit1 = move
        seeds = pieces[pit1]
        pit2 = pit1 + 6
        capture = 0
        oldseeds = seeds
        next_ptr = next[pit1]

        # TODO
        # child_pieces = pieces.mirror()
        child_pieces = pieces[6:12] + pieces[0:6] + pieces[18:24] + pieces[12:18]

        # take away seeds
        child_pieces[pit2] = 0

        # sow them
        while seeds > 0:
            pit2 = next_ptr[pit2]
            child_pieces[pit2] += 1
            seeds -= 1

        captures = 0
        while (pit2 < 6 and pit2 >= 0) and ((child_pieces[pit2] == 2) or (child_pieces[pit2] == 3)):
            # capture
            captures += child_pieces[pit2]
            child_pieces[pit2] = 0
            pit2 -= 1

        # now return the board in the canonical player=1  orientation
        # ret_child_pieces = child_pieces.mirror()
        ret_child_pieces = child_pieces[6:12] + child_pieces[0:6] + child_pieces[18:24] + child_pieces[12:18]

        # update the captures in the returned child
        if captures > 0:
            if verbose: print('captured ' + str(captures) + ' by ' + str(player))

            ret_child_pieces[17] += captures

        return ret_child_pieces
    
    def get_legal_moves(self, pieces, verbose):
        """Returns all the legal moves for the current player, assuming canonical board
        """
        moves = set()  # stores the legal moves.

        # canonical board, so player is 1, not -1
        player = 1 

        if verbose:
            print('get legal moves of player ', player, ' for ', pieces)

        # NOTE: board is in canonical form

        # first find moves that allow the opponent to respond
        for i in range(6):
            if pieces[i] != 0:
                # check that when doing this move, the opponent can respond
                # child = self.sow(i, player, verbose)
                # board = child.mirror()
                board = self.sow(pieces, i, player, verbose)
                if self.can_move(board, - player):
                    if verbose: print('add move ', i, 'since opponent can move in', board)
                    moves.add(i)
                else:
                    if verbose: print('skip move ', i, 'since opponent cannot move in', board)

        if len(moves) == 0:
            # retry, now adding moves that will allow no response
            for i in range(6):
                if pieces[i] != 0:
                    if verbose: print('add move ' + str(i))
                    moves.add(i)

        return list(moves)
    
    def valid_moves(self, pieces, player, verbose):
        # return a fixed size binary vector
        valids = [0] * 7  ## self.action_size()

        all_pieces = pieces + [0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0]
        legalMoves = self.get_legal_moves(all_pieces, verbose)
        if len(legalMoves) == 0:
            valids[-1] = 1
        else:
            for x in legalMoves:
                valids[x] = 1

        return valids

    def play(self, env, verbose):
        obs = env.canonical_observation()[0]
        pieces = [int(obs[0]), int(obs[1]), int(obs[2]), int(obs[3]), int(obs[4]), int(obs[5]),
                  int(obs[6]), int(obs[7]), int(obs[8]), int(obs[9]), int(obs[10]), int(obs[11])]
        if verbose:
            print("pieces", pieces)

        # canoncial board is as seen from player to move:
        player = 1 

        # neede to do a validity check since the reported scores may
        # not be consistent with this, as they give a max score
        # that may dependent on wheher a wipeout is acceptable or not
        valid = self.valid_moves(pieces, player, verbose)
        if verbose:
            print('op: possible moves: ', valid)

        b = db_board(pieces[0], pieces[1], pieces[2], pieces[3], pieces[4], pieces[5],
                     pieces[6], pieces[7], pieces[8], pieces[9], pieces[10], pieces[11])

        overall_score, scores = gen_board_and_child_scores(b)
        if verbose:
            print("score", overall_score)
            print("child_scores", scores)

        best = -127
        best_i = 6
        # first find the best score, picking an arbitrary best when the same
        for i in range(6):
            # NOTE: the validity may not be observed by the scores
            if valid[i] and scores[i] != 127:
                # pick best move, or choose between best ones;
                # allow for mistakes if so desired
                if scores[i] > best or (scores[i] == best and random.random() < 0.5):
                    best = scores[i]
                    best_i = i

        # find the next best score, picking an arbitrary next best when the same
        next_best = -127
        next_best_i = 6
        for i in range(6):
            # NOTE: the validity may not be observed by the scores
            if valid[i] and scores[i] != 127:
                # pick best move, or choose between best ones;
                # allow for mistakes if so desired
                if (scores[i] < best and scores[i] > next_best) or (scores[i] == next_best and random.random() < 0.5):
                    next_best = scores[i]
                    next_best_i = i

        select = best_i
        if random.random() < self.mistake_fraction:
            # take other choice, if not absurd:
            if next_best != -127 and (best - next_best) <= self.mistake_max:
                select = next_best_i
                if verbose:
                    print('op: select suboptimal move')
        if verbose:
            print('op: select ' + str(select))

        return select

    def print_oracle_status(self, env, player):
        # use Python interface for the published Awari database dataset

        # get a canonical view, seen from the current player
        pieces = env.canonical_observation()[0]
        print("pieces", pieces)
        b = db_board(int(pieces[0]), int(pieces[1]), int(pieces[2]),
                              int(pieces[3]), int(pieces[4]), int(pieces[5]),
                              int(pieces[6]), int(pieces[7]), int(pieces[8]),
                              int(pieces[9]), int(pieces[10]), int(pieces[11]))
        score, child_scores = gen_board_and_child_scores(b)
        print("score", score)
        print("child_scores", child_scores)
        return child_scores

class PlayResults(NamedTuple):
    win_count: chex.Array
    draw_count: chex.Array
    loss_count: chex.Array


@partial(
    jax.jit,
    static_argnames=("num_simulations", "disable_mcts", "random_action"),
)
def play_one_move(
    agent,
    env: Enviroment,
    rng_key: chex.Array,
    disable_mcts: bool = False,
    num_simulations: int = 1024,
    random_action: bool = True,
):
    """Play a move using agent's policy"""
    if disable_mcts:
        action_logits, value = agent(env.canonical_observation())
        action_weights = jax.nn.softmax(action_logits, axis=-1)
    else:
        batched_env: Enviroment = replicate(env, 1)  # type: ignore
        rng_key, rng_key_1 = jax.random.split(rng_key)  # type: ignore
        policy_output = improve_policy_with_mcts(
            agent,
            batched_env,
            rng_key_1,  # type: ignore
            rec_fn=recurrent_fn,
            num_simulations=num_simulations,
        )
        action_weights = policy_output.action_weights[0]
        root_idx = policy_output.search_tree.ROOT_INDEX
        value = policy_output.search_tree.node_values[0, root_idx]

    if random_action:
        action = jax.random.categorical(rng_key, jnp.log(action_weights), axis=-1)
    else:
        action = jnp.argmax(action_weights)
    return action, action_weights, value


def agent_vs_agent(
    agent1,
    agent2,
    env: Enviroment,
    rng_key: chex.Array,
    disable_mcts: bool = False,
    num_simulations_per_move: int = 1024,
):
    """A game of agent1 vs agent2."""

    def cond_fn(state):
        env, step = state[0], state[-1]
        # pylint: disable=singleton-comparison
        not_ended = env.is_terminated() == False
        not_too_long = step <= env.max_num_steps()
        return jnp.logical_and(not_ended, not_too_long)

    def loop_fn(state):
        env, a1, a2, _, rng_key, turn, step = state
        rng_key_1, rng_key = jax.random.split(rng_key)
        action, _, _ = play_one_move(
            a1,
            env,
            rng_key_1,
            disable_mcts=disable_mcts,
            num_simulations=num_simulations_per_move,
        )
        env, reward = env_step(env, action)
        state = (env, a2, a1, turn * reward, rng_key, -turn, step + 1)
        return state

    state = (
        reset_env(env),
        agent1,
        agent2,
        jnp.array(0),
        rng_key,
        jnp.array(1),
        jnp.array(1),
    )
    state = jax.lax.while_loop(cond_fn, loop_fn, state)
    return state[3]


@partial(jax.jit, static_argnums=(4, 5, 6))
def agent_vs_agent_multiple_games(
    agent1,
    agent2,
    env,
    rng_key,
    disable_mcts: bool = False,
    num_simulations_per_move: int = 1024,
    num_games: int = 128,
) -> PlayResults:
    """Fast agent vs agent evaluation."""
    rng_key_list = jax.random.split(rng_key, num_games)
    rng_keys = jnp.stack(rng_key_list, axis=0)  # type: ignore
    avsa = partial(
        agent_vs_agent,
        disable_mcts=disable_mcts,
        num_simulations_per_move=num_simulations_per_move,
    )
    batched_avsa = jax.vmap(avsa, in_axes=(None, None, 0, 0))
    envs = replicate(env, num_games)
    results = batched_avsa(agent1, agent2, envs, rng_keys)
    win_count = jnp.sum(results == 1)
    draw_count = jnp.sum(results == 0)
    loss_count = jnp.sum(results == -1)
    return PlayResults(
        win_count=win_count, draw_count=draw_count, loss_count=loss_count
    )


def agent_vs_oracle(
    agent,
    env: Enviroment,
    disable_mcts: bool = False,
    num_simulations_per_move: int = 1024,
    num_games: int = 2,
    verbose: bool = False,
    # verbose: bool = True,
):
    """A number of games of agent vs oracle."""
    rng_key = jax.random.PRNGKey(random.randint(0, 999999))
    
    result_agent_win = 0
    result_oracle_win = 0
    result_draw = 0

    for game_id in range(num_games):
        env = reset_env(env)
        # An agent that has a 10% chance of making a max 5 stone error:
        awari_oracle_player = AwariOraclePlayer(env, 0.10, 5)
        # To use a stronger player:
        # awari_oracle_player = AwariOraclePlayer(env, 0.05, 5)
        # To flip starting player:
        # agent_turn = (game_id % 2)
        agent_turn = ((game_id + 1) % 2)
        agent_started = ((agent_turn % 2) == 0)

        # To be safe set a max number of moves per game, though in practice
        # the game logic will implement lower limit:
        for i in range(200):
            if verbose:
                print()
                print(f"Move {i}")
                print("======")
                print()
                env.render()
            player = 1 if env.thisplayer else -1
            if i % 2 == agent_turn:
                s = env.canonical_observation()
                if verbose:
                    print()
                    print("#  s =", s)
                rng_key_1, rng_key = jax.random.split(rng_key)
                action, action_weights, value = play_one_move(
                    agent,
                    env,
                    rng_key_1,
                    disable_mcts=disable_mcts,
                    num_simulations=num_simulations_per_move,
                    random_action=False,
                )
                if verbose:
                    print("#  A(s) =", action_weights)
                    print("#  V(s) =", value)
                env, reward = env_step(env, action)
                if verbose:
                    print(f"#  Agent selected action {action}, got reward {reward}")
            else:
                if verbose:
                    awari_oracle_player.print_oracle_status(env, player)
                action = awari_oracle_player.play(env, verbose)
                env, reward = env_step(env, jnp.array(action, dtype=jnp.int32))
                if verbose:
                    print(f"#  Fallible Awari Oracle selected action {action}, got reward {reward}")
            if env.is_terminated().item():
                break
        else:
            print("Timeout!")
        if verbose:
            print()
            print("Final board")
            print("===========")
            print()
            env.render()
            print()

        if env.winner == 1:
            if agent_started:
                result_agent_win += 1
            else:
                result_oracle_win += 1
        elif env.winner == -1:
            if agent_started:
                result_oracle_win += 1
            else:
                result_agent_win += 1
        else:
            result_draw += 1

    print("agent_win", result_agent_win, "oracle_win", result_oracle_win, "draw", result_draw)

def human_vs_agent(
    agent,
    env: Enviroment,
    human_first: bool = True,
    disable_mcts: bool = False,
    num_simulations_per_move: int = 1024,
):
    """A game of human vs agent."""
    env = reset_env(env)
    agent_turn = 1 if human_first else 0
    rng_key = jax.random.PRNGKey(random.randint(0, 999999))

    for i in range(1000):
        print()
        print(f"Move {i}")
        print("======")
        print()
        env.render()
        if i % 2 == agent_turn:
            print()
            s = env.canonical_observation()
            print("#  s =", s)
            rng_key_1, rng_key = jax.random.split(rng_key)
            action, action_weights, value = play_one_move(
                agent,
                env,
                rng_key_1,
                disable_mcts=disable_mcts,
                num_simulations=num_simulations_per_move,
                random_action=False,
            )
            print("#  A(s) =", action_weights)
            print("#  V(s) =", value)
            env, reward = env_step(env, action)
            print(f"#  Agent selected action {action}, got reward {reward}")
        else:
            action = input("> ")
            action = env.parse_action(action)
            env, reward = env_step(env, jnp.array(action, dtype=jnp.int32))
            print(f"#  Human selected action {action}, got reward {reward}")
        if env.is_terminated().item():
            break
    else:
        print("Timeout!")
    print()
    print("Final board")
    print("===========")
    print()
    env.render()
    print()


def main(
    game_class: str = "games.awari_game.AwariGame",
    agent_class="policies.resnet_policy.ResnetPolicyValueNet256",
    ckpt_filename: str = "./awari_agent.ckpt",
    against_human: bool = False,
    human_first: bool = False,
    disable_mcts: bool = False,
    num_simulations_per_move: int = 128,
    verbose: bool = False,
):
    """Load agent's weight from disk and start the game."""
    if num_simulations_per_move == 0:
        disable_mcts=True
    warnings.filterwarnings("ignore")
    env = import_class(game_class)()
    agent = import_class(agent_class)(
        input_dims=env.observation().shape,
        num_actions=env.num_actions(),
    )
    with open(ckpt_filename, "rb") as f:
        agent = agent.load_state_dict(pickle.load(f)["agent"])
    agent = agent.eval()
    if against_human:
        human_vs_agent(
            agent,
            env,
            human_first=human_first,
            disable_mcts=disable_mcts,
            num_simulations_per_move=num_simulations_per_move,
        )
    else:
        agent_vs_oracle(
            agent,
            env,
            disable_mcts=disable_mcts,
            num_simulations_per_move=num_simulations_per_move,
            num_games=20,
            verbose=verbose,
        )

if __name__ == "__main__":
    Fire(main)
