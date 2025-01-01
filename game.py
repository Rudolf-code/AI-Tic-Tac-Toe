import numpy as np
import random


class AITac:
    symbols = {1: 'X', -1: 'O', 0: ' ', 2: 'W'}

    def __init__(self,  length=3, min_wildcards=0, max_wildcards=2, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1):
        self.q_table = {}  # {(state, action): q_value}
        self.learning_rate = learning_rate  # Alpha
        self.discount_factor = discount_factor  # Gamma
        self.exploration_rate = exploration_rate  # Epsilon
        self.original_exploration_rate = exploration_rate

        self.grid_length = length
        self.grid = list()
        for i in range(length):
            self.grid.append([0] * length)

        self.max_wildcards = max_wildcards
        self.min_wildcards = min_wildcards
        if self.max_wildcards < 0:
            self.max_wildcards = 0

        if self.max_wildcards > self.grid_length - 2:
            self.max_wildcards = self.grid_length - 2

        if self.min_wildcards < 0:
            self.min_wildcards = 0

        if self.min_wildcards > self.max_wildcards:
            self.min_wildcards = self.max_wildcards

    def add_random_wildcards(self, board):
        empty_positions = [(i, j) for i in range(self.grid_length) for j in range(self.grid_length) if board[i][j] == 0]

        # Determine how many wildcards to add
        num_wildcards = random.randint(self.min_wildcards, self.max_wildcards)

        for a in range(num_wildcards):
            i, j = random.choice(empty_positions)
            board[i][j] = 2  # Place a wildcard
            empty_positions.remove((i, j))

        return board

    def get_state(self, board):
        # Convert the 2D board into a tuple for hashing.
        return tuple(map(tuple, board))

    def get_possible_actions(self, board):
        # Return list of available moves on the board.
        return [(i, j) for i in range(self.grid_length) for j in range(self.grid_length) if board[i][j] == 0]

    def choose_action(self, board):
        # Choose an action based on exploration/exploitation strategy.
        state = self.get_state(board)
        if np.random.uniform(0, 1) < self.exploration_rate:
            # Exploration: Random move
            actions = self.get_possible_actions(board)
            return random.choice(actions)
        else:
            # Exploitation: Choose best known move from Q-table
            q_values = {action: self.q_table.get((state, action), 0) for action in self.get_possible_actions(board)}
            return max(q_values, key=q_values.get)

    def update_q_value(self, board, action, reward, next_board):
        # Update the Q-value for a (state, action) pair.
        state = self.get_state(board)
        next_state = self.get_state(next_board)

        current_q = self.q_table.get((state, action), 0)

        # Max Q-value for the next state
        future_q = max([self.q_table.get((next_state, a), 0) for a in self.get_possible_actions(next_board)], default=0)

        # Update Q-value using Q-learning formula
        self.q_table[(state, action)] = current_q + self.learning_rate * (
            reward + self.discount_factor * future_q - current_q)

    def reset_board(self):
        # Initialize an empty board.
        self.grid = list()
        for i in range(self.grid_length):
            self.grid.append([0] * self.grid_length)

        self.add_random_wildcards(self.grid)

    def is_winning_line(self, line, player):
        """Helper function to check if a line (row, column, or diagonal) is a winning line for the given player.
        A line is winning if all cells are the player or wildcards (2)."""
        return all(cell == player or cell == 2 for cell in line)

    def check_winner(self, board):
        """Check if there's a winner on a variable-sized board with wildcards.
        Wildcards are represented by 2, empty spaces by 0.
        Return 1 if AITac wins, -1 if opponent wins, 0 otherwise."""
        n = len(board)  # Get the size of the board (n x n)

        # Check rows for a winner
        for row in board:
            if self.is_winning_line(row, 1):
                return 1  # Player 1 (AITac) wins
            if self.is_winning_line(row, -1):
                return -1  # Player -1 (Opponent) wins

        # Check columns for a winner
        for col in range(n):
            if self.is_winning_line([board[row][col] for row in range(n)], 1):
                return 1
            if self.is_winning_line([board[row][col] for row in range(n)], -1):
                return -1

        # Check main diagonal (top-left to bottom-right) for a winner
        if self.is_winning_line([board[i][i] for i in range(n)], 1):
            return 1
        if self.is_winning_line([board[i][i] for i in range(n)], -1):
            return -1

        # Check anti-diagonal (top-right to bottom-left) for a winner
        if self.is_winning_line([board[i][n - 1 - i] for i in range(n)], 1):
            return 1
        if self.is_winning_line([board[i][n - 1 - i] for i in range(n)], -1):
            return -1

        return 0  # No winner

    def is_draw(self, board):
        # Check if the board is full.
        for i in range(self.grid_length):
            for j in range(self.grid_length):
                if board[i][j] == 0:
                    return False

        return True

    def apply_action(self, board, action, player):
        # Apply the move of the current player on the board.
        board[action[0]][action[1]] = player

    def play_game(self):
        # Run a game between two AITac agents (self-play) and return the reward
        self.reset_board()
        board = self.grid
        current_player = 1  # 1 for AITac1, -1 for AITac2

        # Separate move histories for each player
        move_history_1 = []
        move_history_2 = []

        while True:
            # Choose action
            if current_player == 1:
                action = self.choose_action(board)
                move_history_1.append((self.get_state(board), action))  # Record for AITac1
            else:
                action = self.choose_action(board)
                move_history_2.append((self.get_state(board), action))  # Record for AITac2

            # Apply action
            self.apply_action(board, action, current_player)

            # Check for winner or draw
            winner = self.check_winner(board)
            if winner != 0:
                # Assign rewards for win/loss
                if winner == 1:
                    reward_1 = 1  # AITac1 wins
                    reward_2 = -1  # AITac2 loses
                else:
                    reward_1 = -1  # AITac1 loses
                    reward_2 = 1  # AITac2 wins
                break
            elif self.is_draw(board):  # Draw
                reward_1 = 0.5
                reward_2 = 0.5
                break

            # Switch players
            current_player *= -1

        # Update Q-values for AITac1
        for i, (state, action) in enumerate(move_history_1):
            if i < len(move_history_1) - 1:
                next_state = move_history_1[i + 1][0]
                self.update_q_value(state, action, reward_1, next_state)
            else:
                self.update_q_value(state, action, reward_1, self.get_state(board))

        # Update Q-values for AITac2
        for i, (state, action) in enumerate(move_history_2):
            if i < len(move_history_2) - 1:
                next_state = move_history_2[i + 1][0]
                self.update_q_value(state, action, reward_2, next_state)
            else:
                self.update_q_value(state, action, reward_2, self.get_state(board))

        return reward_1

    def train(self, iterations=10000):
        # Train the AITac by self-play for a number of iterations.
        self.exploration_rate = self.original_exploration_rate

        for iteration in range(iterations):
            reward = self.play_game()
            if iteration % 1000 == 0:
                print(f"Iteration {iteration}")

    def print_board(self, board):
        # Print the tic-tac-toe board with cells numbered 1 to 9.
        cell_number = 1
        for row in board:
            row_symbols = []
            for cell in row:
                if cell == 0:
                    row_symbols.append(str(cell_number))  # Display the cell number if it's empty
                else:
                    row_symbols.append(self.symbols[cell])  # Display X or O for filled cells
                cell_number += 1
            print(' '.join(row_symbols))  # No lines between rows

    def get(self, line_index: int, board):
        return board[int(line_index/self.grid_length)][line_index % self.grid_length]

    def convert_1d_to_2d(self, line_index: int):
        return int(line_index/self.grid_length), line_index % self.grid_length

    def human_move(self, board):
        # Get the human player's move.
        while True:
            try:
                inp = int(input("Enter number (1 to "+str(self.grid_length*self.grid_length)+"):"))
                inp = inp - 1
                if self.get(inp, board) == 0:
                    return self.convert_1d_to_2d(inp)
                else:
                    print("Spot is already taken, choose another.")
            except (IndexError, ValueError):
                print("Invalid input. Please enter a number between 1 and "+str(self.grid_length*self.grid_length))

    def play_against_human(self):
        # Function to play against the AI.
        while True:
            self.exploration_rate = 0
            self.reset_board()
            board = self.grid
            ai_player = random.choice([-1, 1])  # 1 = AI, -1 = Human
            human_player = ai_player*-1
            current_player = 1
            print("You are "+self.symbols[human_player]+". AI is "+self.symbols[ai_player]+".")

            while True:
                self.print_board(board)
                if current_player == human_player:
                    # Human turn
                    print("Your turn:")
                    action = self.human_move(board)
                else:
                    # AI's turn
                    print("AI's turn:")
                    action = self.choose_action(board)

                self.apply_action(board, action, current_player)

                # Check for winner
                winner = self.check_winner(board)
                if winner != 0:
                    self.print_board(board)
                    if winner == ai_player:
                        print("AI wins!")
                    else:
                        print("You win!")
                    break
                elif self.is_draw(board):
                    self.print_board(board)
                    print("It's a draw!")
                    break

                # Switch player
                current_player *= -1

    def print_q_table(self):
        # Print the Q-table for all states and actions in a human-readable format.
        if not self.q_table:
            print("Q-table is empty.")
            return

        # Group actions by state
        state_action_dict = {}
        for (state, action), q_value in self.q_table.items():
            if state not in state_action_dict:
                state_action_dict[state] = []
            state_action_dict[state].append((action, q_value))

        # Print each state with its actions and corresponding Q-values
        for state, actions in state_action_dict.items():
            print("State:")
            # Convert the state tuple back into a board format and print it
            for row in state:
                print(' '.join([str(cell) for cell in row]))

            print("Actions and Q-values:")
            for action, q_value in actions:
                print(f"Action: {action}, Q-value: {q_value:.2f}")

            print('-' * 40)


ai_tac = AITac(3, 0, 0, 0.1, 0.9, 0.4)
ai_tac.train(60000)  # more iterations lead to a smarter AI

#ai_tac.print_q_table()
ai_tac.play_against_human()
