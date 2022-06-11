import copy
import random

#Global variables
boardSize = 5
komi =2.5
maxMove= boardSize * boardSize - 1
myDeadPieces = 0
oppDeadPieces = 0
# pieceType
# 'X' pieces marked as 1 which indicates Black Stone which goes first
# 'O' pieces marked as 2 which indicates White Stone

def readInput():
    lines = []
    with open('input.txt', 'r') as f:
        for eachLine in f.readlines():
            lines.append(eachLine.strip())
    pieceType = int(lines[0])
    previousBoard = [[int(x) for x in eachLine] for eachLine in lines[1:boardSize + 1]]
    board = [[int(x) for x in eachLine] for eachLine in lines[boardSize + 1: 2 * boardSize + 1]]
    return pieceType, previousBoard, board

    # with open('input.txt', 'r') as f:
    #     lines = f.readlines()
    #
    #     pieceType = int(lines[0])
    #
    #     previousBoard = [[int(x) for x in line.rstrip('\n')] for line in lines[1:boardSize + 1]]
    #     board = [[int(x) for x in line.rstrip('\n')] for line in lines[boardSize + 1: 2 * boardSize + 1]]
    #
    # return pieceType, previousBoard, board

def opponent_pieceType():
    if pieceType == 1:
        return 2
    elif pieceType == 2:
        return 1

def copy_board(board):
    '''
    Copy the current board for potential testing.

    :param: None.
    :return: the copied board instance.
    '''
    return copy.deepcopy(board)

def printOutput(result):
    with open('output.txt', 'w') as f:
        if result == "PASS":
            f.write(result)
        else:
            f.write(str(result[0])+','+str(result[1]))

def compare_board(previousBoard, board):
    for i in range(5):
        for j in range(5):
            if board[i][j] != previousBoard[i][j]:
                return False
    return True

def heuristic(board,myPiece):

    max =0
    min = 0
    heuristicMax = 0
    heuristicMin = 0
    myCorner = 0
    oppCorner = 0
    myEdges = 0
    oppEdges = 0
    myEndangered = 0
    oppEndangered =0
    for row in range(5):
        for col in range(5):
            if board[row][col] == pieceType:
                max = max + 1
                heuristicMax = heuristicMax + (max + 0.5*liberty_count(board, row, col))
                if liberty_count(board, row, col) <= 1:
                    myEndangered += 1
            elif board[row][col] == 3 - pieceType:
                min = min + 1
                heuristicMin = heuristicMin + (min + 0.5*liberty_count(board, row, col))
                if liberty_count(board, row, col) <= 1:
                    oppEndangered += 1
            if (board[0][0] == pieceType or board[0][4] == pieceType or board[4][0] == pieceType or board[4][4] == pieceType):
                myCorner += 1
            elif (board[0][0] == 3 - pieceType or board[0][4] == 3 - pieceType or board[4][0] == 3 - pieceType or board[4][4] == 3 - pieceType):
                oppCorner += 1
            if(board[0][1] == pieceType or board[0][2] == pieceType or board[0][3] == pieceType or board[4][1] == pieceType or board[4][2] == pieceType or board[4][3] == pieceType or board[1][4] == pieceType or board[2][4] == pieceType or board[3][4] == pieceType or board[1][0] == pieceType or board[2][0] == pieceType or board[3][0] == pieceType):
                myEdges += 1
            elif (board[0][1] == 3 - pieceType or board[0][2] == 3 - pieceType or board[0][3] == 3 - pieceType or board[4][1] == 3 - pieceType or board[4][2] == 3 - pieceType or board[4][3] == 3 - pieceType or board[1][4] == 3 - pieceType or board[2][4] == 3 - pieceType or board[3][4] == 3 - pieceType or board[1][0] == 3 - pieceType or board[2][0] == 3 - pieceType or board[3][0] == 3 - pieceType):
                oppEdges += 1



    if myPiece == pieceType:
        return 3*heuristicMax -0.5*myEdges - myCorner - 3*heuristicMin + 0.5*oppEdges + oppCorner - 0.5*myEndangered + 0.5*oppEndangered
    else:
        return 3*heuristicMin - 3*heuristicMax  +0.5*myEdges + myCorner - 0.5*oppEdges - oppCorner + 0.5*myEndangered - 0.5*oppEndangered

def find_died_pieces(board, pieceType):
    '''
    Find the died stones that has no liberty in the board for a given piece type.

    :param piece_type: 1('X') or 2('O').
    :return: a list containing the dead pieces row and column(row, column).
    '''
    died_pieces = []
    # global myDeadPieces = 0
    # global oppDeadPieces = 0
    for i in range(boardSize): #row
        for j in range(boardSize): #col
            # Check if there is a piece at this position:
            if board[i][j] == pieceType:
                # The piece die if it has no liberty
                if not liberty_count(board, i, j) and (i, j) not in died_pieces:
                    died_pieces.append((i, j))
                    myDeadPieces += 1
            else:
                # The piece die if it has no liberty
                if not liberty_count(board, i, j) and (i, j) not in died_pieces:
                    oppDeadPieces += 1


    return died_pieces

def remove_certain_pieces(board, positions):
    '''
    Remove the stones of certain locations.

    :param positions: a list containing the pieces to be removed row and column(row, column)
    :return: None.
    '''
    for piece in positions:
        board[piece[0]][piece[1]] = 0
    return board

def remove_died_pieces(board, pieceType):
    '''
    Remove the dead stones in the board.

    :param piece_type: 1('X') or 2('O').
    :return: locations of dead pieces.
    '''

    died_pieces = find_died_pieces(board, pieceType)
    if not died_pieces:
        return board
    newBoard = remove_certain_pieces(board, died_pieces)
    return newBoard

def detect_neighbor(board, i, j):  ###
    '''
    Detect all the neighbors of a given stone.
    :param i: row number of the board.
    :param j: column number of the board.
    :return: a list containing the neighbors row and column (row, column) of position (i, j).
    '''

    #currentBoard = copy_board(board)
    board = remove_died_pieces(board, (i,j))
    neighbors = []  #[(i-1,j),(i+1,j),(i,j-1),(i,j+1)]
    # Detect borders and add neighbor coordinates
    if i > 0:
        neighbors.append((i - 1, j))
    if i < len(board) - 1:
        neighbors.append((i + 1, j))
    if j > 0:
        neighbors.append((i, j - 1))
    if j < len(board) - 1:
        neighbors.append((i, j + 1))
    #return ([piece for piece in neighbors if 0 <= piece[0] < boardSize and 0 <= piece[1] < boardSize])
    return neighbors

def detect_neighbor_ally(i, j, board):
    '''
    Detect the neighbor allies of a given stone.
    :param i: row number of the board.
    :param j: column number of the board.
    :return: a list containing the neighbored allies row and column (row, column) of position (i, j).
    '''
    dneighbors = detect_neighbor(board, i, j)  # call the function Detect neighbors and store the value in neighbors to find ally
    groupAllies = []
    # Iterate through neighbors
    for piece in dneighbors:
        if board[piece[0]][piece[1]] == board[i][j]: # Add to allies list if having the same color
            groupAllies.append(piece)
    return groupAllies

def ally_dfs(board, i, j):
    '''
    Using DFS to search for all allies of a given stone.
    :param i: row number of the board.
    :param j: column number of the board.
    :return: a list containing the all allies row and column (row, column) of position (i, j).
    '''
    stack = [(i, j)]  # stack for DFS serach
    allyMembers = []  # record allies positions during the search
    while stack:
        piece = stack.pop()
        allyMembers.append(piece)
        #neighborAllies = detect_neighbor_ally(piece[0], piece[1],board)
        for ally in detect_neighbor_ally(piece[0], piece[1],board):
            if ally not in stack and ally not in allyMembers:
                stack.append(ally)
    return allyMembers

def find_liberty(currentBoard, i, j):
    '''
    Find liberty of a given stone. If a group of allied stones has no liberty, they all die.
    :param i: row number of the board.
    :param j: column number of the board.
    :return: boolean indicating whether the given stone still has liberty.
    '''

    allyMembers = ally_dfs(currentBoard, i, j)
    for member in allyMembers:
        neighbors = detect_neighbor(currentBoard, member[0], member[1]) #member[0] has value of i , that is row number and member[1] has value of j,column number
        for piece in neighbors:
            if currentBoard[piece[0]][piece[1]] == 0:       # If there is empty space around a piece, it has liberty
                return True
    # If none of the pieces in allied group has an empty space, it has no liberty
    return False

def liberty_count(board, i, j):
    liberty_count = 0
    allyMembers = ally_dfs(board, i, j)
    for member in allyMembers:
        neighbors = detect_neighbor(board, member[0], member[1])  # member[0] has value of i , that is row number and member[1] has value of j,column number
        for piece in neighbors:
            if board[piece[0]][piece[1]] == 0:  # If there is empty space around a piece, it has liberty
                liberty_count = liberty_count + 1

    return liberty_count

def valid_place_check(board, previousBoard, i, j, myPiece):
    '''
    Check whether a placement is valid.
    :param i: row number of the board.
    :param j: column number of the board.
    :param piece_type: 1(white piece) or 2(black piece).
    :param test_check: boolean if it's a test check.
    :return: boolean indicating whether the placement is valid.
    '''
    if board[i][j] != 0: # Condition to check whether the piece
        return False
    currentBoard = copy_board(board)
    # Check if the place has liberty
    currentBoard[i][j] = myPiece
    died_pieces = find_died_pieces(currentBoard,3-myPiece)
    currentBoard = remove_died_pieces(currentBoard,3-myPiece)
    if liberty_count(currentBoard, i, j) >=1 and not (died_pieces and compare_board(previousBoard,currentBoard)):  # Check special case: repeat placement causing the repeat board state (KO rule)
        return True

def newMove(board,currentMove, myPiece):
    copyBoard = copy_board(board)
    copyBoard[currentMove[0]][currentMove[1]] = myPiece
    copyBoard = remove_died_pieces(copyBoard,3-myPiece)

    return copyBoard

def find_valid_positions(board,previousBoard,myPiece):
    validPositions=[]
    for row in range(5):
        for col in range(5):
            if valid_place_check(board,previousBoard,row,col,myPiece):
                validPositions.append((row,col))

    return validPositions

def newAction(currBoardState,prevBoardState,pieceType,maxDepth,alpha,beta):

    movements = []
    final = 0
    #countEyes = []
    copyCurrentState = copy_board(currBoardState)

    for currentMove in find_valid_positions(currBoardState,prevBoardState,pieceType):
        newState = newMove(currBoardState, currentMove, pieceType)
        minScore = minimizer(newState,copyCurrentState,maxDepth,alpha,beta, 3 - pieceType)
        score = -1 * minScore # minimizer(newState,copyCurrentState,maxDepth,alpha,beta, 3 - pieceType)

        if score > final or not movements:
            final = score
            alpha = final
            movements = [currentMove]
        elif score == final:
            movements.append(currentMove)
    return movements

def minimizer (currBoardState,prevBoardState,maxDepth,alpha,beta,opponent):

    final = heuristic(currBoardState,opponent)
    if maxDepth == 0:
        return final

    copyCurrentState = copy_board(currBoardState)

    for currentMove in find_valid_positions(currBoardState,prevBoardState,opponent):
        newState = newMove(currBoardState, currentMove, opponent)
        maxScore = maximizer(newState, copyCurrentState,maxDepth -1, alpha , beta, 3 - opponent)
        currentScore = -1 * maxScore    ##maximizer(newState, copyCurrentState,maxDepth -1, alpha , beta, 3 - opponent)

        if currentScore > final:
            final = currentScore

        myPiece = -1 * final

        if myPiece < alpha:
            return final
        if final > beta:
            beta = final
    return final

def maximizer(currBoardState, prevBoardState, maxDepth, alpha, beta, opponent):
    final = heuristic(currBoardState,opponent)
    if maxDepth == 0:
        return final

    copyCurrentState = copy_board(currBoardState)

    for currentMove in find_valid_positions(currBoardState, prevBoardState, opponent):
        newState = newMove(currBoardState, currentMove, opponent)
        minScore = minimizer(newState, copyCurrentState, maxDepth, alpha, beta, 3 - pieceType)
        currentScore = -1 *  minScore   #minimizer(newState, copyCurrentState, maxDepth - 1, alpha, beta, 3 - opponent)


        if currentScore > final:
            final = currentScore

        opponent = -1 * final

        if opponent < beta:
            return final
        if final > alpha:
            alpha = final
    return final

if __name__ == "__main__":

    global previousBoard
    global board
    global pieceType
    global check
    #global maxDepth=2
    check=0
    placeCheck = False

    pieceType, prevBoard, currBoard = readInput()
    for i in range(boardSize):
        for j in range(boardSize):
            if currBoard[i][j] != 0:
                if i == 2 and j == 2:
                    placeCheck = True
                check = check + 1
    if (check == 1 and pieceType == 2 and placeCheck == False) or (check == 0 and pieceType == 1): #first move
        action =[(2,2)]
    else:
        action = newAction(currBoard,prevBoard,pieceType, 2 , -1000, 1000)

    if action == []:
        val=['PASS']
    else:
        val=random.choice(action)

    printOutput(val)
