
import heapq
import math
import time

#Defining Global Variables
algo=""
dimension=()
entranceLoc=()
exitLoc=()
totalGrid=0
routes={}
actionPoint=[]

#Creating a function to get next action movement nodes
def action_movement(point,action):
	successor=()
	if action == 1:
		successor = (point[0] + 1,point[1],point[2])
	elif action == 2:
		successor = (point[0] - 1, point[1], point[2])
	elif action == 3:
		successor = (point[0], point[1] + 1, point[2])
	elif action == 4:
		successor = (point[0], point[1] - 1, point[2])
	elif action == 5:
		successor = (point[0], point[1], point[2] + 1)
	elif action == 6:
		successor = (point[0], point[1], point[2] - 1)
	elif action == 7:
		successor = (point[0] + 1, point[1] + 1, point[2])
	elif action == 8:
		successor = (point[0] + 1, point[1] - 1, point[2])
	elif action == 9:
		successor = (point[0] - 1, point[1] + 1, point[2])
	elif action == 10:
		successor = (point[0] - 1, point[1] - 1, point[2])
	elif action == 11:
		successor = (point[0] + 1, point[1], point[2] + 1)
	elif action == 12:
		successor = (point[0] + 1, point[1], point[2] - 1)
	elif action == 13:
		successor = (point[0] - 1, point[1], point[2] + 1)
	elif action == 14:
		successor = (point[0] - 1, point[1], point[2] - 1)
	elif action == 15:
		successor = (point[0], point[1] + 1, point[2] + 1)
	elif action == 16:
		successor = (point[0], point[1] + 1, point[2] - 1)
	elif action == 17:
		successor = (point[0], point[1] - 1, point[2] + 1)
	elif action == 18:
		successor = (point[0], point[1] - 1, point[2] - 1)
	else:
		successor = (point[0], point[1], point[2])

	# To validate the boundary constraint
	if (successor[0]>=0 and successor[1]>=0 and successor[2]>=0) and (successor[0]<dimension[0] and successor[1]<dimension[1] and successor[2]<dimension[2]):
		if(algo == 'BFS'):
			return successor
		if(algo == 'UCS' or algo == 'A*'):
			if (action>0 and action<7):
				successor = (successor,10)
				return successor
			elif(action>6 and action<19):
				successor = (successor,14)
				return successor
			else:
				successor = (successor,0)
				return successor


def path_bfs_traversal(algo, parent, entranceLoc, exitLoc):
	nodes = []
	currentPoint = exitLoc
	nodes.append(exitLoc)
	while True:
		if (currentPoint[0] == entranceLoc[0] and currentPoint[1] == entranceLoc[1] and currentPoint[2] == entranceLoc[2]): #to check if the entrance and exit grid location are same or not
			break
		currentPoint = parent.get(currentPoint) #To retrieve the parent node from the parent dictionary
		nodes.append(currentPoint)

	length = len(nodes)
	currentPoint = entranceLoc
	cost = 0
	currentCost=0
	printNodes = []

	for i in range(0, length):
		coordinates = nodes[length - 1 - i]
		distance = (abs(coordinates[0] - currentPoint[0]) + abs(coordinates[1] - currentPoint[1]) + abs(coordinates[2] - currentPoint[2]))

		if algo == 'BFS':
			if distance == 0:
				currentCost=0
			else:
				currentCost= 1

		cost = cost + currentCost
		currentPoint = coordinates
		newNodes = str(coordinates[0]) + " " + str(coordinates[1]) + " " + str(coordinates[2]) + " " + str(currentCost)
		printNodes.append(newNodes)

	f = open("output.txt", "w")
	f.write(str(cost))
	f.write("\n" + str(length))
	for line in printNodes:
		f.write("\n" +  line)
	f.close()


def path_traversal(algo, exitLoc , parentLoc ,parent):

	nodes = []
	currentPoint = exitLoc
	point = parentLoc[0]
	nodes.append(exitLoc)

	while True:
		if (currentPoint[0] == entranceLoc[0] and currentPoint[1] == entranceLoc[1] and currentPoint[2] == entranceLoc[2]):
			break

		currentPoint = parent.get(currentPoint)[0]
		nodes.append(currentPoint)


	length = len(nodes)
	currentPoint = entranceLoc
	cost = 0
	currentCost=0
	printNodes = []

	for i in range(0, length):
		coordinates = nodes[length - 1 - i]
		distance = (abs(coordinates[0] - currentPoint[0]) + abs(coordinates[1] - currentPoint[1]) + abs(coordinates[2] - currentPoint[2]))
		if (algo == 'UCS' or algo =='A*'):
			if distance == 0:
				currentCost=0
			elif distance == 1:
				currentCost = 10
			elif distance == 2:
				currentCost = 14

		cost = cost + currentCost
		currentPoint = coordinates
		newNodes = str(coordinates[0]) + " " + str(coordinates[1]) + " " + str(coordinates[2]) + " " + str(currentCost)
		printNodes.append(newNodes)

	f = open("output.txt", "w")
	f.write(str(cost))
	f.write("\n" + str(length))
	for line in printNodes:
		f.write("\n" +  line)
	f.close()


def get_heuristic(neighbors,exitLoc):
	heuristicValue = math.floor(math.sqrt((exitLoc[0] - neighbors[0])**2 + (exitLoc[1] - neighbors[1])**2 + (exitLoc[2] - neighbors[2])**2)*9)
	return heuristicValue

#BFS Algorithm
def bfs():
	boolResult=False
	bfs_queue = []  #queue created to store the accessing nodes
	visited={}
	parent={}
	dist={}

	for i in routes:
		visited[i] = False
	visited[entranceLoc] = True;
	dist[entranceLoc] = 0;
	bfs_queue.append(entranceLoc);

	neighbors = ()
	if(entranceLoc[0]>=0 and entranceLoc[1]>=0 and entranceLoc[2]>=0 and exitLoc[0]<dimension[0] and exitLoc[1]<dimension[1] and exitLoc[2]<dimension[2]):
		while (len(bfs_queue) != 0):
			point = bfs_queue.pop(0)

			for action in routes[point]:

				neighbors = action_movement(point, action)
				if(neighbors is None ):
					continue


				if visited[neighbors] == False:
					visited[neighbors] = True
					dist[neighbors] = 1 #dist[point]+1
					parent[neighbors] = point

					bfs_queue.append(neighbors)

					if (neighbors == exitLoc):
						boolResult=True
						break

	if boolResult:
		path_bfs_traversal(algo, parent, entranceLoc, exitLoc)
	else:
		f = open("output.txt", "w")
		f.write("FAIL")
		f.close()

#UCS Algorithm
def ucs():
	visited = {}
	parent = {}
	visitedStatus={}
	frontier = []
	isVisited = False

	heapq.heappush(frontier, (0, entranceLoc))
	visitedStatus [entranceLoc] = 0
	parent[entranceLoc] = (None,0)

	if (entranceLoc[0] >= 0 and entranceLoc[1] >= 0 and entranceLoc[2] >= 0 and exitLoc[0] < dimension[0] and exitLoc[1] < dimension[1] and exitLoc[2] < dimension[2]):
		while (len(frontier) != 0) and isVisited == False:
			popVal = heapq.heappop(frontier)
			cost = popVal[0]
			point = popVal[1]


			if(point == exitLoc):
				isVisited = True
				break
			else:
				if routes[point] != None:

					for action in routes[point]:
						neighbors = action_movement(point, action)
						if (neighbors is None):
							continue
						nextVal = neighbors[0]
						currentCost = cost + neighbors[1]


						if (nextVal in visitedStatus and visitedStatus.get(nextVal) > currentCost):
							heapq.heappush(frontier,(currentCost,nextVal))
							parent[nextVal] = (point, neighbors[1])
							visitedStatus[nextVal] = currentCost

						elif( nextVal in visited and visited.get(nextVal)>currentCost):
							heapq.heappush(frontier , (currentCost,nextVal))
							visitedStatus[nextVal]=currentCost
							visited.pop(nextVal)

						elif( nextVal not in visitedStatus and nextVal not in visited):
							heapq.heappush(frontier , (currentCost,nextVal))
							parent[nextVal]=(point , neighbors[1])
							visitedStatus[nextVal]=currentCost
			visited[point]=cost

	if isVisited:
			path_traversal(algo, exitLoc, parent[exitLoc], parent)
	else:
		f = open("output.txt", "w")
		f.write("FAIL")
		f.close()

#Astar Algorithm
def Astar():
	visited = {}
	parent = {}
	visitedStatus={}
	heuristic ={}
	frontier = []
	isVisited = False

	heapq.heappush(frontier, (0, entranceLoc))
	visitedStatus [entranceLoc] = 0
	heuristic[entranceLoc]=0
	parent[entranceLoc] = (None,0)

	if(entranceLoc[0] >= 0 and entranceLoc[1] >= 0 and entranceLoc[2] >= 0 and exitLoc[0] < dimension[0] and exitLoc[1] < dimension[1] and exitLoc[2] < dimension[2]):

		while (len(frontier) != 0):
			popVal = heapq.heappop(frontier)
			cost = popVal[0]
			point = popVal[1]
			parentHeuristic = heuristic.get(point)

			if(point == exitLoc):
				isVisited = True
				break
			else:
				if routes[point] != None:

					for action in routes[point]:
						neighbors = action_movement(point, action)
						if (neighbors is None):
							continue
						nextVal = neighbors[0]
						heuristic[nextVal] = get_heuristic(nextVal,exitLoc)

						#currentCost = cost + neighbors[1] + get_heuristic(nextVal,exitLoc) - parentHeuristic
						currentCost = cost + neighbors[1] + get_heuristic(nextVal,exitLoc) - parentHeuristic


						if ((nextVal in visitedStatus and visitedStatus.get(nextVal) > currentCost) or( nextVal not in visitedStatus and nextVal not in visited)) :

							heapq.heappush(frontier,(currentCost,nextVal))
							parent[nextVal] = (point, neighbors[1])
							visitedStatus[nextVal] = currentCost


						elif( nextVal in visited and visited.get(nextVal)>currentCost):
							heapq.heappush(frontier , (currentCost,nextVal))
							visitedStatus[nextVal]=currentCost
							visited.pop(nextVal)

			visited[point]=cost

	if isVisited:
			path_traversal(algo, exitLoc, parent[exitLoc], parent)
	else:
		f = open("output.txt", "w")
		f.write("FAIL")
		f.close()

#Creating a function to choose which algorithm to run
def choose_algo():
	if algo == 'BFS':
		bfs()
	elif algo == "UCS":
		ucs()
	elif algo == "A*":
		Astar()
	else:
		print("Invalid Input")

#Creating a function to create a input file
def input_read():
	global algo
	global dimension
	global entranceLoc
	global exitLoc
	global totalGrid
		lines = file.readline()
		algo = lines.rstrip('\n')
		lines = file.readline()
		dimension = tuple(map(int, lines.split()))  	#total maze size
		lines = file.readline()
		entranceLoc = tuple(map(int, lines.split()))
		lines = file.readline()
		exitLoc = tuple(map(int, lines.split()))
		lines = file.readline()
		totalGrid = int(lines)  #total routes

		for i in range (0,totalGrid):
			lines = file.readline()
			linesArray = lines.strip().split(" ")
			actionPoint=[int(routeNumber) for routeNumber in linesArray[3:]]
			routeLoc = (int(linesArray[0]), int(linesArray[1]), int(linesArray[2]))
			routes[routeLoc] = actionPoint

	file.close()
	choose_algo()

if __name__== '__main__':
	start_time = time.time()
	input_read()

	print("--- %s seconds ---" % (time.time() - start_time))















