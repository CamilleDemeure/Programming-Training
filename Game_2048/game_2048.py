import random
import pickle
import os


class Game2048:
    directions = {
        "LEFT": (0, -1),
        "RIGHT": (0, 1),
        "UP": (-1, 0),
        "DOWN": (1, 0)
    }

    @staticmethod
    def initBoard(n=4):
        """
        Creation of a board of size nxn with one value

        Notes:
            0 refers to an empty space
        Args:
            n (int): size of the board nxn
        Returns:
            board (list)
        """
        board = []
        for i in range(4):
            board.append([0]*4)
        Game2048.addValue(board)
        return board

    @staticmethod
    def isFull(board):
        """
        Test if the board is full or not

        Args:
            board (list): the board of the game written as a list

        Returns:
            (bool): True if the board is full, else return False
        """
        return 0 not in Game2048.getValues(board)

    @staticmethod
    def getLocationEmptySpaces(board):
        """
        Return the list of the positions of empty spaces

        Args:
            board (list): the board of the game

        Returns:
            (list): the list of positions of empty squares on the board
        """
        location_of_empty_spaces = []
        for x in range(len(board)):
            for y in range(len(board[0])):
                if Game2048.getValue(board, (x, y)) == 0:
                    location_of_empty_spaces.append((x, y))
        return location_of_empty_spaces

    @staticmethod
    def getValue(board, coord):
        """
        Returns the value of the square at the indicated postion

        Args:
            board (list): the board of the game
            coord (tuple): the coordinates at which we want to retrieve
            the value of the board

        Returns:
            (int): Value of the board at the given position
        """
        x, y = coord
        try:
            return board[x][y]
        except ValueError:
            print("The coordinates entered are out of the range of the board")

    @staticmethod
    def getValueInGivenDirection(board, x, y, direction):
        """
        Return the value of the square adjacent to the square
        of position (x, y) when we look in one of the four direction

        Args:
            board (list): the game board
            x (int): the coordinates x of the square
            y (int): the coordinates y of the square

        Returns:
            direction (tuple): value of the square adjacent to (x, y)
            following the given direction
        """
        dir_x, dir_y = direction
        x_final, y_final = x + dir_x, y + dir_y
        if x_final < 0 or y_final < 0:
            return None
        return Game2048.getValue(board, (x_final, y_final))

    @staticmethod
    def getValues(board):
        """
        Return the values of the whole board

        Args:
            board (list): the game board

        Returns:
            (list): list of value of the squares of the board
        """
        squares = []
        for row in board:
            for square in row:
                squares.append(square)
        return squares

    @staticmethod
    def lengthMaxValue(board):
        """
        Return the number of characters of the max value
        in the game board

        Args:
            grille (list): the game board

        Returns:
            (int): number of characters of the max value on the board
        """
        return len(str(max(Game2048.getValues(board))))

    @staticmethod
    def MaxValue(board):
        """
        Return the highest value on the board

        Args:
            board (list): the game board


        Returns:
            (int): the max value on the board
        """
        return max(Game2048.getValues(board))

    @staticmethod
    def getEmptySquarePosition(board):
        """
        Return a random couple of coordinates that
        gives the location of an empty square on the
        game board

        Args:
            board (list): the game board

        Returns:
            (tuple) the coordinates of a random empty square
        """
        return random.choice(Game2048.getLocationEmptySpaces(board))

    @staticmethod
    def addValue(board):
        """
        Add a new value on an empty square of the board

        Args:
            board(list): the game board
        """

        x, y = Game2048.getEmptySquarePosition(board)
        board[x][y] = random.choices([2, 4], [0.7, 0.3])[0]

    @staticmethod
    def getTransposedBoard(board):
        """
        Get the transposed game board

        Args:
            board(list): the game board

        Returns:
            The game board transposed (rows becomes columns
            and columns become rows)
        """
        return list(map(list, zip(*board)))

    @staticmethod
    def moveLeft(row):
        """
        Move all squares to the left

        Args:
            row(list): one row of the board

        Returns:
            (list): the ligne moved to the left
        """
        updated_row, k, previous = [0 for i in range(len(row))], 0, None
        for i in range(len(row)):
            if row[i] != 0:
                if previous is None:
                    previous = row[i]
                else:
                    if previous == row[i]:
                        updated_row[k] = row[i]*2
                        k += 1
                        previous = None
                    else:
                        updated_row[k] = previous
                        k += 1
                        previous = row[i]
        if previous is not None:
            updated_row[k] = previous
        return updated_row

    @staticmethod
    def swipeBoardInGivenDirection(board, direction):
        """
        Perform the swipe of the board in the given direction

        Args:
            board (list): the game board contained in a list
            direction (str): the direction in which to swipe the board
        """
        updated_board = board.copy()
        direction = Game2048.directions[direction]
        x, y = direction[0], direction[1]

        # left swipe
        if y == -1:
            for i in range(len(board)):
                updated_board[i] = Game2048.moveLeft(board[i])

        # right swipe
        elif y == 1:
            for i in range(len(board)):
                row = board[i].copy()
                row.reverse()
                row = Game2048.moveLeft(row)
                row.reverse()
                updated_board[i] = row

        # up swipe
        elif x == -1:
            updated_board = Game2048.getTransposedBoard(board)
            for i in range(len(updated_board)):
                updated_board[i] = Game2048.moveLeft(updated_board[i])
            updated_board = Game2048.getTransposedBoard(updated_board)

        # down swipe
        elif x == -1:
            updated_board = Game2048.getTransposedBoard(board)
            for i in range(len(updated_board)):
                row = updated_board[i].copy()
                row.reverse()
                row = Game2048.moveLeft(row)
                row.reverse()
                updated_board[i] = row
            updated_board = Game2048.getTransposedBoard(updated_board)

        return updated_board

    @staticmethod
    def possibleSwipe(board):
        """
        Return the list of available swipes

        Args:
             board(list): the game board

        Returns:
            (list) : the list of available swipe
        """
        directions_available = []
        for direction in Game2048.directions.keys():
            new = Game2048.swipeBoardInGivenDirection(board, direction)
            if new == board:
                directions_available.append(False)
            else:
                directions_available.append(True)
        return directions_available

    @staticmethod
    def displayBoard(board, n=4):
        """
        Display the game board

        Args:
            board (list): the game board
            n(int): the size of the board
        """
        size_square = Game2048.lengthMaxValue(board)
        print('-'*((n + 1) + size_square * n))
        for row in board:
            for square in row:
                print('|{:{align}{width}}'.format(
                    square, align='^',
                    width=size_square),
                    end="")
            print('|')
            print('-'*((n+1) + size_square * n))

    @staticmethod
    def getScore(board):
        """
        Return the total score of the game board

        Args:
            board (list): the game board

        Returns:
            (int) : total sum of the squares
        """
        score = 0
        for tuile in Game2048.getValues(board):
            score += tuile
        return score

    @staticmethod
    def saveBoard(board, folder):
        """
        Save the game board

        Args:
            board (list): the game board
            folder (str): the name of the folder to save the game
        """
        with open(folder, 'wb') as f:
            pickle.dump(board, f)

    @staticmethod
    def loadBoard(folder):
        """
        Load a previous game

        Args:
            folder (str): fichier where the game is saved

        Returns:
            (list): the board saved as a list
        """
        print(os.listdir())
        print(folder in os.listdir())

        with open(folder, 'rb') as f:
            list_board = pickle.load(f)
        return list_board

    @staticmethod
    def getRanking(folder):
        """
        Load the folder containing the ranking and return this ranking
        under the form of a list

        If the ranking doesn't exist, the function initialize the ranking

        Args:
            folder (str): the name of the folder

        Returns:
            (list): a list of tuples (score, pseudo, size)
        """
        try:
            with open(folder, 'rb') as f:
                ranking = pickle.load(f)
        except FileNotFoundError:
            ranking = []
            Game2048.saveRanking(folder, ranking)
        return ranking

    @staticmethod
    def saveRanking(folder, ranking):
        """
        Save the list of rankings in a folder

        Args:
            folder (str): the name of the folder containing the rankings
            ranking (list): a list of tuple (pseudo, score)
        """
        with open(folder, 'wb') as f:
            pickle.dump(ranking, f)

    @staticmethod
    def updateRanking(folder, pseudo, score, size):
        """
        Add a new player in the ranking

        Args:
            folder (str): the folder containing the rankings
            pseudo (str): the pseudo of the player
            score (int): the score of the game
            size (int): the size of the game board
        """
        ranking = Game2048.getRanking(folder)
        if len(ranking) >= 10:
            i = 0
            while i < len(ranking):
                if ranking[i][0] <= score:
                    ranking[i] = (score, pseudo, size)
                    break
        else:
            ranking.append((score, pseudo, size))
            ranking.sort(reverse=True)
        Game2048.saveRanking(folder, ranking)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
