import math
import random
import time


class Nim():

    def __init__(self, initial=[1, 3, 5, 7]):
        """
        Initialize game board.
        Each game board has
            - `piles`: a list of how many elements remain in each pile
            - `player`: 0 or 1 to indicate which player's turn
            - `winner`: None, 0, or 1 to indicate who the winner is
        """
        self.piles = initial.copy()
        self.player = 0
        self.winner = None

    @classmethod
    def available_actions(cls, piles):
        """
        Nim.available_actions(piles) takes a `piles` list as input
        and returns all of the available actions `(i, j)` in that state.

        Action `(i, j)` represents the action of removing `j` items
        from pile `i` (where piles are 0-indexed).
        """
        actions = set()
        for i, pile in enumerate(piles):
            for j in range(1, pile + 1):
                actions.add((i, j))
        return actions

    @classmethod
    def other_player(cls, player):
        """
        Nim.other_player(player) returns the player that is not
        `player`. Assumes `player` is either 0 or 1.
        """
        return 0 if player == 1 else 1

    def switch_player(self):
        """
        Switch the current player to the other player.
        """
        self.player = Nim.other_player(self.player)

    def move(self, action):
        """
        Make the move `action` for the current player.
        `action` must be a tuple `(i, j)`.
        """
        pile, count = action

        # Check for errors
        if self.winner is not None:
            raise Exception("Game already won")
        elif pile < 0 or pile >= len(self.piles):
            raise Exception("Invalid pile")
        elif count < 1 or count > self.piles[pile]:
            raise Exception("Invalid number of objects")

        # Update pile
        self.piles[pile] -= count
        self.switch_player()

        # Check for a winner
        if all(pile == 0 for pile in self.piles):
            self.winner = self.player


class NimAI():

    def __init__(self, alpha=0.5, epsilon=0.1):
        """
        Initialize AI with an empty Q-learning dictionary,
        an alpha (learning) rate, and an epsilon rate.

        The Q-learning dictionary maps `(state, action)` pairs to a Q-value.
        """
        self.q = dict()
        self.alpha = alpha
        self.epsilon = epsilon

    def get_q_value(self, state, action):
        """
        Return the Q-value for the state `state` and the action `action`.
        If no Q-value exists yet in `self.q`, return 0.
        """
        # Convert state to a tuple to be hashable for dictionary lookup
        state_tuple = tuple(state)
        # Check if the state-action pair exists in Q table, return its value, else return 0
        return self.q.get((state_tuple, action), 0)

    def update_q_value(self, state, action, old_q, reward, future_rewards):
        """
        Update the Q-value for the state `state` and the action `action`
        given the previous Q-value `old_q`, a current reward `reward`,
        and an estimate of future rewards `future_rewards`.

        The Q-learning update formula is:
            Q(s, a) <- old_q + alpha * (reward + future_rewards - old_q)
        """
        # Compute the new Q-value using the Q-learning formula
        new_q = old_q + self.alpha * (reward + future_rewards - old_q)
        # Convert state to a tuple to store in Q-table
        state_tuple = tuple(state)
        # Update the Q-value in the dictionary
        self.q[(state_tuple, action)] = new_q

    def best_future_reward(self, state):
        """
        Given a state `state`, find the best possible reward from available actions
        by looking up the Q-values for each action and returning the highest one.
        If there are no available actions, return 0.
        """
        # Get all possible actions for the given state
        available_actions = Nim.available_actions(state)
        # If no actions available, return 0
        if not available_actions:
            return 0
        # Find the maximum Q-value of available actions
        best_reward = float('-inf')
        for action in available_actions:
            q_value = self.get_q_value(state, action)
            best_reward = max(best_reward, q_value)
        return best_reward

    def choose_action(self, state, epsilon=True):
        """
        Given a state `state`, return an action `(i, j)` to take.

        If `epsilon` is False, the AI will select the best action based on Q-values.
        If `epsilon` is True, the AI will use epsilon-greedy: sometimes select a random action.
        """
        # Get all possible actions for the current state
        available_actions = list(Nim.available_actions(state))

        # If epsilon is set to False, choose the best action (greedy)
        if not epsilon:
            best_action = None
            best_q_value = float('-inf')
            for action in available_actions:
                q_value = self.get_q_value(state, action)
                if q_value > best_q_value:
                    best_q_value = q_value
                    best_action = action
            return best_action

        # If epsilon is set to True, use epsilon-greedy strategy
        if random.random() < self.epsilon:
            # Choose a random action with epsilon probability
            return random.choice(available_actions)
        else:
            # Choose the best action with the remaining probability
            best_action = None
            best_q_value = float('-inf')
            for action in available_actions:
                q_value = self.get_q_value(state, action)
                if q_value > best_q_value:
                    best_q_value = q_value
                    best_action = action
            return best_action


def train(n):
    """
    Train an AI by playing `n` games against itself.
    """

    player = NimAI()

    # Play n games
    for i in range(n):
        print(f"Playing training game {i + 1}")
        game = Nim()

        # Keep track of last move made by either player
        last = {
            0: {"state": None, "action": None},
            1: {"state": None, "action": None}
        }

        # Game loop
        while True:

            # Keep track of current state and action
            state = game.piles.copy()
            action = player.choose_action(game.piles)

            # Keep track of last state and action
            last[game.player]["state"] = state
            last[game.player]["action"] = action

            # Make move
            game.move(action)
            new_state = game.piles.copy()

            # When game is over, update Q values with rewards
            if game.winner is not None:
                player.update(state, action, new_state, -1)
                player.update(
                    last[game.player]["state"],
                    last[game.player]["action"],
                    new_state,
                    1
                )
                break

            # If game is continuing, no rewards yet
            elif last[game.player]["state"] is not None:
                player.update(
                    last[game.player]["state"],
                    last[game.player]["action"],
                    new_state,
                    0
                )

    print("Done training")

    # Return the trained AI
    return player


def play(ai, human_player=None):
    """
    Play human game against the AI.
    `human_player` can be set to 0 or 1 to specify whether
    human player moves first or second.
    """

    # If no player order set, choose human's order randomly
    if human_player is None:
        human_player = random.randint(0, 1)

    # Create new game
    game = Nim()

    # Game loop
    while True:

        # Print contents of piles
        print()
        print("Piles:")
        for i, pile in enumerate(game.piles):
            print(f"Pile {i}: {pile}")
        print()

        # Compute available actions
        available_actions = Nim.available_actions(game.piles)
        time.sleep(1)

        # Let human make a move
        if game.player == human_player:
            print("Your Turn")
            while True:
                pile = int(input("Choose Pile: "))
                count = int(input("Choose Count: "))
                if (pile, count) in available_actions:
                    break
                print("Invalid move, try again.")

        # Have AI make a move
        else:
            print("AI's Turn")
            pile, count = ai.choose_action(game.piles, epsilon=False)
            print(f"AI chose to take {count} from pile {pile}.")

        # Make move
        game.move((pile, count))

        # Check for winner
        if game.winner is not None:
            print()
            print("GAME OVER")
            winner = "Human" if game.winner == human_player else "AI"
            print(f"Winner is {winner}")
            return
