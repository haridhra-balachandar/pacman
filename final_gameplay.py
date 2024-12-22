import pygame, sys, os
import math
from queue import PriorityQueue
from queue import Queue


boards = [
[6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5],
[3, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 3],
[3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3],
[3, 3, 1, 6, 4, 4, 5, 1, 6, 4, 4, 4, 5, 1, 3, 3, 1, 6, 4, 4, 4, 5, 1, 6, 4, 4, 5, 1, 3, 3],
[3, 3, 2, 3, 0, 0, 3, 1, 3, 0, 0, 0, 3, 1, 3, 3, 1, 3, 0, 0, 0, 3, 1, 3, 0, 0, 3, 2, 3, 3],
[3, 3, 1, 7, 4, 4, 8, 1, 7, 4, 4, 4, 8, 1, 7, 8, 1, 7, 4, 4, 4, 8, 1, 7, 4, 4, 8, 1, 3, 3],
[3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3],
[3, 3, 1, 6, 4, 4, 5, 1, 6, 5, 1, 6, 4, 4, 4, 4, 4, 4, 5, 1, 6, 5, 1, 6, 4, 4, 5, 1, 3, 3],
[3, 3, 1, 7, 4, 4, 8, 1, 3, 3, 1, 7, 4, 4, 5, 6, 4, 4, 8, 1, 3, 3, 1, 7, 4, 4, 8, 1, 3, 3],
[3, 3, 1, 1, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1, 1, 1, 3, 3],
[3, 7, 4, 4, 4, 4, 5, 1, 3, 7, 4, 4, 5, 0, 3, 3, 0, 6, 4, 4, 8, 3, 1, 6, 4, 4, 4, 4, 8, 3],
[3, 0, 0, 0, 0, 0, 3, 1, 3, 6, 4, 4, 8, 0, 7, 8, 0, 7, 4, 4, 5, 3, 1, 3, 0, 0, 0, 0, 0, 3],
[3, 0, 0, 0, 0, 0, 3, 1, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 1, 3, 0, 0, 0, 0, 0, 3],
[8, 0, 0, 0, 0, 0, 3, 1, 3, 3, 0, 6, 4, 4, 9, 9, 4, 4, 5, 0, 3, 3, 1, 3, 0, 0, 0, 0, 0, 7],
[4, 4, 4, 4, 4, 4, 8, 1, 7, 8, 0, 3, 0, 0, 0, 0, 0, 0, 3, 0, 7, 8, 1, 7, 4, 4, 4, 4, 4, 4],
[3, 0, 0, 0, 0, 0, 3, 1, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 1, 3, 0, 0, 0, 0, 0, 3],
[4, 4, 4, 4, 4, 4, 5, 1, 6, 5, 0, 3, 0, 0, 0, 0, 0, 0, 3, 0, 6, 5, 1, 6, 4, 4, 4, 4, 4, 4],
[5, 0, 0, 0, 0, 0, 3, 1, 3, 3, 0, 7, 4, 4, 4, 4, 4, 4, 8, 0, 3, 3, 1, 3, 0, 0, 0, 0, 0, 6],
[3, 0, 0, 0, 0, 0, 3, 1, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 1, 3, 0, 0, 0, 0, 0, 3],
[3, 0, 0, 0, 0, 0, 3, 1, 3, 3, 0, 6, 4, 4, 4, 4, 4, 4, 5, 0, 3, 3, 1, 3, 0, 0, 0, 0, 0, 3],
[3, 6, 4, 4, 4, 4, 8, 1, 7, 8, 0, 7, 4, 4, 5, 6, 4, 4, 8, 0, 7, 8, 1, 7, 4, 4, 4, 4, 5, 3],
[3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3],
[3, 3, 1, 6, 4, 4, 5, 1, 6, 4, 4, 4, 5, 1, 3, 3, 1, 6, 4, 4, 4, 5, 1, 6, 4, 4, 5, 1, 3, 3],
[3, 3, 1, 7, 4, 5, 3, 1, 7, 4, 4, 4, 8, 1, 7, 8, 1, 7, 4, 4, 4, 8, 1, 3, 6, 4, 8, 1, 3, 3],
[3, 3, 2, 1, 1, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 1, 1, 2, 3, 3],
[3, 7, 4, 5, 1, 3, 3, 1, 6, 5, 1, 6, 4, 4, 4, 4, 4, 4, 5, 1, 6, 5, 1, 3, 3, 1, 6, 4, 8, 3],
[3, 6, 4, 8, 1, 7, 8, 1, 3, 3, 1, 7, 4, 4, 5, 6, 4, 4, 8, 1, 3, 3, 1, 7, 8, 1, 7, 4, 5, 3],
[3, 3, 1, 1, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1, 1, 1, 3, 3],
[3, 3, 1, 6, 4, 4, 4, 4, 8, 7, 4, 4, 5, 1, 3, 3, 1, 6, 4, 4, 8, 7, 4, 4, 4, 4, 5, 1, 3, 3],
[3, 3, 1, 7, 4, 4, 4, 4, 4, 4, 4, 4, 8, 1, 7, 8, 1, 7, 4, 4, 4, 4, 4, 4, 4, 4, 8, 1, 3, 3],
[3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3],
[3, 7, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8, 3],
[7, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8]
         ]


height = 900
width = 950
heightScale = ((height - 50) // 32)
widthScale = (((width - 50) // 30))

pacman_folder = os.path.join('pacman_icons', 'frame1.png')
pacman = pygame.image.load('frame1.png')
size = (35, 35)
pacman = pygame.transform.scale(pacman, size)
direction = 'right'
ghost_mode = 'CHASE'

pygame.init()
screen = pygame.display.set_mode((height, width))

pygame.display.set_caption("PACMAN")
clock = pygame.time.Clock()
fps = 60

# pacman starting postion
x, y = 435, 620

points = 0

def drawBoard():
    global heightScale, widthScale, points
    centre_x = x + 17.5
    centre_y = y + 17.5
    b_colour = (0,0,255) # background colour
    heightScale = ((height - 50) // 32)
    widthScale = (((width - 50) // 30))
    for i in range(len(boards)):
        for j in range(len(boards[i])):
            if boards[i][j] == 1:
                colour = (243, 207, 198)
                mid_dot_x = int(j * widthScale + (widthScale / 2))
                mid_dot_y = int(i * heightScale + (heightScale / 2))
                dot = pygame.Rect(mid_dot_x, mid_dot_y, 4, 4)
                pygame.draw.rect(screen, colour, dot)
                
                if hitbox.colliderect(dot):
                    boards[i][j] = 0
                    points += 10
                
            if boards[i][j] == 2:
                colour = (243, 207, 198)
                mid_pellet_x = int(j * widthScale + (widthScale / 2))
                mid_pellet_y = int(i * heightScale + (heightScale / 2))
                pygame.draw.circle(screen, colour, (mid_pellet_x, mid_pellet_y) , 10)
                pellet = pygame.Rect(mid_pellet_x, mid_pellet_y, 4, 4)
                
                if hitbox.colliderect(pellet):
                    boards[i][j] = 0
                    points += 50
                    
            if boards[i][j] == 3:
                pygame.draw.line(screen, b_colour, (int(j * widthScale + (widthScale / 2)), i * heightScale), (int(j * widthScale + (widthScale / 2)), i * heightScale + heightScale), 3)
            if boards[i][j] == 4:
                pygame.draw.line(screen, b_colour, (j * widthScale , int(i * heightScale + (heightScale / 2))), (j * widthScale + widthScale, int(i * heightScale + (heightScale / 2))), 3)
            if boards[i][j] == 5:
                    # treat arc within boundary rectangle --> draw from top left corner
                    # 2 pi radians = 360 degree. so 90 degrees or quarter of circle = pi/2.
                    # arcs are measured anticlockwise
                    # 0 radian points to right
                    # we we measure the grid from top left point ( + in height scale moves it down)
                pygame.draw.arc(screen, b_colour, [(j * widthScale - (widthScale  * 0.45)) - 2, (i * heightScale + int(heightScale/2)), widthScale, heightScale], 0, (math.pi) / 2, 3)
            if boards[i][j] == 6:
                pygame.draw.arc(screen, b_colour, [(j * widthScale + (widthScale  * 0.45)) + 2, (i * heightScale + int(heightScale/2)), widthScale, heightScale], (math.pi) / 2, (math.pi), 3)
            if boards[i][j] == 7:
                pygame.draw.arc(screen, b_colour, [(j * widthScale + (widthScale  * 0.45)) + 2, (i * heightScale - int(heightScale/2)), widthScale, heightScale], (math.pi), (math.pi) * 1.5, 3)
            if boards[i][j] == 8:
                pygame.draw.arc(screen, b_colour, [(j * widthScale - (widthScale  * 0.45)) + 2, (i * heightScale - int(heightScale/2)), widthScale, heightScale], (math.pi) * 1.5, 0, 3)
            if boards[i][j] == 9:
                pygame.draw.line(screen, (255,255,255), (j * widthScale , int(i * heightScale + (heightScale / 2))), (j * widthScale + widthScale, int(i * heightScale + (heightScale / 2))), 3)
    


def check_env(x, y, direction):
    
   heightScale = (height - 50) // 32
   widthScale = (width - 50) // 30
   centreX = x + 17.5
   centreY = y + 17.5
   buffer = 10
   if direction == 'right':
      if boards[int(centreY // heightScale)][int((centreX + buffer) // widthScale)]<= 2:
         return True
   elif direction == 'left':
      if boards[int(centreY // heightScale)][int((centreX - buffer)  // widthScale)]<= 2:
         return True
   elif direction == 'up':
      if boards[int((centreY - buffer) // heightScale)][int(centreX  // widthScale)]<= 2:
         return True
   elif direction == 'down':
      if boards[int((centreY + buffer)// heightScale)][int(centreX  // widthScale)]<= 2:
         return True 
   
   
   return False
   

def redraw():
   global direction
   global x, y
   screen.blit(pacman, (x,y))
   pacman_speed = 2
   if direction == 'right' and check_env(x,y,'right'):
      x += pacman_speed
      screen.blit(pacman, (x,y))
   if direction == 'down' and check_env(x,y,'down'):
      y += pacman_speed
      screen.blit(pygame.transform.rotate(pacman, 270), (x,y))
   if direction == 'left' and check_env(x,y,'left'):
      x -= pacman_speed
      screen.blit(pygame.transform.rotate(pacman, 180), (x,y))
   if direction == 'up' and check_env(x,y,'up'):
      y -= pacman_speed
      screen.blit(pygame.transform.rotate(pacman, 90), (x,y))


def arrToCo(arrPos):
    row, col = arrPos
    return col * widthScale , row*heightScale

def CoToArr(Coor):
    x,y = Coor
    return int((y + 17.5)// heightScale) , int((x + 17.5) // widthScale)


def heuristic(node, goal_node):
    """ Simple Manhattan distance heuristic. """
    return abs(node.position[0] - goal_node.position[0]) + abs(node.position[1] - goal_node.position[1])


class Node:
    def __init__(self, position):
        """
        Initialize a new node.
        
        :param position: Tuple (x, y) coordinates of the node.
        :param g_cost: Actual cost from start node to this node.
        :param h_cost: Heuristic cost (estimated) from this node to the goal.
        """
        g_cost=float('inf')
        h_cost=0
        self.position = position  # (array i, array j) coordinates
        self.g_cost = g_cost  # Actual cost from start node
        self.h_cost = h_cost  # Heuristic cost to the goal
        self.f_cost = g_cost + h_cost  # Total cost
        self.parent = None  # Used to trace the path
        self.neighbours = {}  # dictionary of neighbour nodes with the node of the postion and then the value of distance from the current node.
        self.starts = [] # adjacent paths which are moveable
    
    def update_costs(self, g_cost, h_cost):
        """ Update g_cost, h_cost, and f_cost. """
        self.g_cost = g_cost
        self.h_cost = h_cost
        self.f_cost = g_cost + h_cost
    
    def find_reverse_direction(self, start, end):
        if start[0] == end[0] and start[1] == end[1] + 1:
            reverse_direction = 'R'
        if start[0] == end[0] and start[1] == end[1] - 1:
            reverse_direction = 'L'
        if start[0] == end[0] + 1 and start[1] == end[1]:
            reverse_direction = 'D'
        if start[0] == end[0] - 1 and start[1] == end[1]:
            reverse_direction = 'U'

        return reverse_direction


    def adj_moves(self, point, prev_point, node_pos, reverse_direction, count):
        # position of all the nodes in the graph
        global positions
        
        i, j = point
        moves = []
        
        # returns if starts is the adj node 
        if point in positions:
            return point
        
        if boards[i][j] == 0 or boards[i][j] == 1 or boards[i][j] == 2:
            # d
            if (reverse_direction != 'D') and (boards[i + 1][j] == 0 or boards[i + 1][j] == 1 or boards[i + 1][j] == 2):
                moves.append((i + 1, j))
            
            # u
            if (reverse_direction != 'U') and (boards[i - 1][j] == 0 or boards[i - 1][j] == 1 or boards[i - 1][j] == 2):
                moves.append((i - 1, j))
            
            # r
            if (reverse_direction != 'R') and (boards[i][j + 1] == 0 or boards[i][j + 1] == 1 or boards[i][j + 1] == 2):
                moves.append((i, j + 1))

            # l
            if (reverse_direction != 'L') and (boards[i][j - 1] == 0 or boards[i][j - 1] == 1 or boards[i][j - 1] == 2):
                moves.append((i, j - 1))

        if prev_point in moves:
            moves.remove(prev_point)
        
        if point in moves:
            moves.remove(point)
        
        reverse_direction = self.find_reverse_direction(point, moves[0])

        if moves[0] in positions:
            return moves[0], count + 1 # increment count before returning it
        
        else:
            count += 1
            return self.adj_moves(moves[0], point, positions, reverse_direction, count)

    def path_to_adj_node(self, point, prev_point, reverse_direction, path_to_node, count):
        # position of all the nodes in the graph
        global positions
        
        i, j = point
        moves = []
        
        # returns if starts is the adj node 
        if point in positions:
            path_to_node.append(point)
            return point, path_to_node, count
        
        if boards[i][j] == 0 or boards[i][j] == 1 or boards[i][j] == 2:
            # d
            if (reverse_direction != 'D') and (boards[i + 1][j] == 0 or boards[i + 1][j] == 1 or boards[i + 1][j] == 2):
                moves.append((i + 1, j))
            
            # u
            if (reverse_direction != 'U') and (boards[i - 1][j] == 0 or boards[i - 1][j] == 1 or boards[i - 1][j] == 2):
                moves.append((i - 1, j))
            
            # r
            if (reverse_direction != 'R') and (boards[i][j + 1] == 0 or boards[i][j + 1] == 1 or boards[i][j + 1] == 2):
                moves.append((i, j + 1))

            # l
            if (reverse_direction != 'L') and (boards[i][j - 1] == 0 or boards[i][j - 1] == 1 or boards[i][j - 1] == 2):
                moves.append((i, j - 1))
    
        if prev_point in moves:
            moves.remove(prev_point)
        
        if point in moves:
            moves.remove(point)
        
        reverse_direction = self.find_reverse_direction(point, moves[0])
        
        path_to_node.append(point)
        
        if moves[0] in positions:
            return moves[0], path_to_node, count + 1 # increment count before returning it
        
        else:
            count += 1
            return self.path_to_adj_node(moves[0], point, reverse_direction, path_to_node, count)
    
    def add_neighbour(self, graph):

        departs_queue = Queue(maxsize = 4)

        for depart_point in self.starts:
            departs_queue.put(depart_point)
        
        while not departs_queue.empty():
        # for each element in the queue testing = works
            start = departs_queue.get()
            
            r_direction = self.find_reverse_direction(node.position, start)
            # fixes the deficit of 1 error
            count = 1
            
            positions = [node.position for node in graph_of_nodes]
            
            neighbour, final_count = self.adj_moves(start, node, positions, r_direction, count)
            # now it appends the node
            self.neighbours.update({(Node(neighbour)): final_count})

    def find_nearest_node(self):
        # Get the positions of all nodes in the graph
        positions = [node.position for node in graph_of_nodes]
    
        # Check if the current position is already a graph node
        if self.position in positions:
            return self.position 
        
        # Extract the current position as (row, column)
        i, j = self.position
        path_starts = []  # List to store valid starting points for paths
        near_nodes = []  # List to store nearest nodes along with their paths and distances


        # Check if the current cell is valid for movement and determine path starts
        if boards[i][j] == 0 or boards[i][j] == 1 or boards[i][j] == 2:
            # Check adjacent cells and add valid ones to path_starts
            if boards[i + 1][j] == 0 or boards[i + 1][j] == 1 or boards[i + 1][j] == 2:
                path_starts.append((i + 1, j))  # Down
            if boards[i - 1][j] == 0 or boards[i - 1][j] == 1 or boards[i - 1][j] == 2:
                path_starts.append((i - 1, j))  # Up
            if boards[i][j + 1] == 0 or boards[i][j + 1] == 1 or boards[i][j + 1] == 2:
                path_starts.append((i, j + 1))  # Right
            if boards[i][j - 1] == 0 or boards[i][j - 1] == 1 or boards[i][j - 1] == 2:
                path_starts.append((i, j - 1))  # Left
        
        # Check path_start is one of position in graph_of_nodes
        for path_start in path_starts:
            if path_start in positions:
                return path_start
        
        departs_queue = Queue(maxsize=4)
        
        # Add all valid path starts to the queue
        for depart_point in path_starts:
            departs_queue.put(depart_point)
        
        while not departs_queue.empty():
            # Get the next path start from the queue
            path_start = departs_queue.get()
            # Find the reverse direction to avoid backtracking
            r_direction = self.find_reverse_direction((i, j), path_start)
            count = 1  # Initialize step count
            path_to_node = []  # List to store the path to the node
            
            near_node, path_to_node, final_count = self.path_to_adj_node(
                path_start, (i, j), r_direction, path_to_node, count)
            # Add the result to the list of nearest nodes
            near_nodes.append((near_node, path_to_node, final_count))
        
        # Initialize with the first node in the near_nodes list
        lowest_count = near_nodes[0][2]  # Smallest step count
        nearest_node = near_nodes[0][0]  # Nearest node
        path_to_node = near_nodes[0][1]  # Path to the nearest node


        # Compare all nodes to find the one with the lowest step count
        for node in near_nodes:
            if int(node[2]) < lowest_count:
                lowest_count = node[2]  # Update the lowest step count
                nearest_node = node[0]  # Update the nearest node
                path_to_node = node[1]  # Update the path to the nearest node
        
        # Add the nearest node to the path and return it
        path_to_node.append(nearest_node)
        return path_to_node, nearest_node  # Return the path and the nearest node

    def node_to_node(self, point, prev_point, end_point, reverse_direction, path_to_node):
        # Global variable to track positions of all nodes in the graph
        global positions
        # Extract the current point's coordinates
        i, j = point
        moves = []  # List to store possible moves from the current point

        # Check if the current point is already a known node and is not the endpoint
        if point in positions and point != end_point:
            return None 

        # Check if the current point is the endpoint
        if point == end_point:
            path_to_node.append(point)  # Add the endpoint to the path
            return path_to_node  

        # Only proceed if the current cell is traversable (e.g., value 0, 1, or 2)
        if boards[i][j] == 0 or boards[i][j] == 1 or boards[i][j] == 2:
            # Down unless moving back in reverse direction
            if (reverse_direction != 'D') and (boards[i + 1][j] == 0 or boards[i + 1][j] == 1 or boards[i + 1][j] == 2):
                moves.append((i + 1, j))
            
            # Up unless moving back in reverse direction
            if (reverse_direction != 'U') and (boards[i - 1][j] == 0 or boards[i - 1][j] == 1 or boards[i - 1][j] == 2):
                moves.append((i - 1, j))
            
            # Right unless moving back in reverse direction
            if (reverse_direction != 'R') and (boards[i][j + 1] == 0 or boards[i][j + 1] == 1 or boards[i][j + 1] == 2):
                moves.append((i, j + 1))
            
            # Left unless moving back in reverse direction
            if (reverse_direction != 'L') and (boards[i][j - 1] == 0 or boards[i][j - 1] == 1 or boards[i][j - 1] == 2):
                moves.append((i, j - 1))

        # Remove the previous point from possible moves to avoid going back
        if prev_point in moves:
            moves.remove(prev_point)

        # Remove the current point from possible moves to prevent looping
        if point in moves:
            moves.remove(point)

        # Determine the new reverse direction based on the first move
        reverse_direction = self.find_reverse_direction(point, moves[0])

        # Add the current point to the path
        path_to_node.append(point)

        # If the next move is already processed and is not the endpoint, terminate
        if moves[0] in positions and moves[0] != end_point:
            return None

        # If the next move is the endpoint, return the path
        if moves[0] == end_point:
            return path_to_node

        # Recursively continue the search with the updated parameters
        else:
            return self.node_to_node(moves[0], point, end_point, reverse_direction, path_to_node)
    
graph_of_nodes = []

for i in range(1, len(boards) - 1):
    for j in range(1, len(boards[i]) - 1):
        path_D = 0
        path_U = 0
        path_R = 0
        path_L = 0
        
        # if tile is 9 it is not a node
        if boards[i][j] == 0 or boards[i][j] == 1 or boards[i][j] == 2:
            if boards[i + 1][j] == 0 or boards[i + 1][j] == 1 or boards[i + 1][j] == 2:
                path_D += 1
            if boards[i - 1][j] == 0 or boards[i - 1][j] == 1 or boards[i - 1][j] == 2:
                path_U += 1
            if boards[i][j + 1] == 0 or boards[i][j + 1] == 1 or boards[i][j + 1] == 2:
                path_R += 1
            if boards[i][j - 1] == 0 or boards[i][j - 1] == 1 or boards[i][j - 1] == 2:
                path_L += 1
        
        # new path start point are stored in starts attribute for the node 
        if (path_D + path_U + path_R + path_L) >= 3:
            new_node = Node((i,j))
            paths = []
            if path_D == 1:
                paths.append((i + 1, j))
            if path_U == 1:
                paths.append((i - 1, j))
            if path_R == 1:
                paths.append((i, j + 1))
            if path_L == 1:
                paths.append((i, j - 1))
            
            new_node.starts = paths

            graph_of_nodes.append(new_node)
    
# prematurely add all the neighbours
            
positions = [node.position for node in graph_of_nodes]        

for node in graph_of_nodes:
    node.add_neighbour(graph_of_nodes)


def reconstruct_path(goal_node):
    path = []
    for node in graph_of_nodes:
        if node.position == goal_node.position:
            current = node

    while current is not None:         
        path.append(current.position)
        
        current = current.parent
                
    return path[::-1]


def A_star(graph, start_node, goal_node):
    openQueue = PriorityQueue()  # Priority queue to manage nodes to explore
    closed_list = []  # List to keep track of visited nodes

    for node in graph:  # Initialise all nodes with infinite costs and no parent
        node.g_cost = float('inf')
        node.f_cost = float('inf')
        node.parent = None
    
    for node in graph_of_nodes:  # Find and set the actual start_node
        if node.position == start_node.position:
            start_node = node
            
    start_node.g_cost = 0  # Start node has a g_cost of 0
    openQueue.put((0, 0, start_node))  # Creates a tuple for priority queue sorting
    
    count = 0

    while openQueue.qsize() != 0:  # Continue until there are no more nodes to explore
        # find node with the lowest f_cost
        current_node =  (openQueue.get()[2])  # Get node with the lowest f_cost


        if current_node.position == goal_node.position:  # Check if we reached the goal
            return(reconstruct_path(current_node))  # Return the reconstructed path
        
        temp_f_costs = [] 
        
        closed_list.append(current_node.position)  # Mark the current node as visited
                
        for node in graph_of_nodes:  # Update the current_node reference
            if node.position == current_node.position:
                current_node = node
        

        # neighbour is a tuple of (key value) where the key is the node and value is the distance
        for neighbour in current_node.neighbours.items():
            # we start at the start_node with a g_cost of 0
            if neighbour[0].position not in [node for node in closed_list]:
                
                temp_f_cost = current_node.g_cost + neighbour[1] + heuristic(neighbour[0], goal_node)
                if temp_f_cost < neighbour[0].f_cost:
                    count += 1
                    
                    for node in graph_of_nodes:
                        if node.position == neighbour[0].position:
                            nextNode = node

                    nextNode.g_cost = current_node.g_cost + neighbour[1]
                    nextNode.f_cost = temp_f_cost
                    nextNode.parent = current_node
                    # add the node to the queue which is in order of ascending f_value
                    f_cost = nextNode.f_cost
                    
                    openQueue.put((nextNode.f_cost, count, nextNode))
                      # Add neighbour to priority queue


def path_node_to_node(first_node, second_node):
    positions = [node.position for node in graph_of_nodes]

    step_path = []
    path_to_node = []
    starts = []
    
    start_origin = first_node
    end_step = second_node
    
    i,j = start_origin
    
    # find all starts
    if boards[i][j] == 0 or boards[i][j] == 1 or boards[i][j] == 2:
        if boards[i + 1][j] == 0 or boards[i + 1][j] == 1 or boards[i + 1][j] == 2:
            starts.append((i + 1, j))
        if boards[i - 1][j] == 0 or boards[i - 1][j] == 1 or boards[i - 1][j] == 2:
            starts.append((i - 1, j))
        if boards[i][j + 1] == 0 or boards[i][j + 1] == 1 or boards[i][j + 1] == 2:
            starts.append((i, j + 1))
        if boards[i][j - 1] == 0 or boards[i][j - 1] == 1 or boards[i][j - 1] == 2:
            starts.append((i, j - 1))
    
    for start in starts:
        if start in positions:
            return start

    if start_origin == end_step:
        return start_origin
    
    # add all the starts into priotity queue - prioritising the distance
    departs_queue = PriorityQueue()
    
    # if the depart point change i have to delete the preious list --> reason for the error
    for depart_point in starts:
        # distance from start to the node to see most likely optimum path
        distance = heuristic(Node(depart_point), Node(end_step))
        
        departs_queue.put((distance, depart_point))
    
    # take each start and return the path 
    while departs_queue.qsize() != 0:
        start = (departs_queue.get()[1])
        
        
        reverse_direction = Node(start).find_reverse_direction(start_origin, start)
        # clear the list if the correct node is not reached
        path_to_node = []
                    
        step_path = Node(start).node_to_node(start, start_origin, end_step, reverse_direction, path_to_node)
        
        if step_path is not None:
            step_path.append(end_step)
            return step_path
        
    # find path by modifying previous algorithm.
     

def final_path(start_point, end_point):
    
    # if nodes are adjacent
    if (start_point.position[0] == end_point.position[0]) and (abs(start_point.position[1] - end_point.position[1]) == 1) or (start_point.position[1] == end_point.position[1]) and (abs(start_point.position[0] - end_point.position[0]) == 1):
        return[start_point.position, end_point.position]
    
    # if ghost is in cage
    s_row, s_col = start_point.position
    
    n_row, n_col = s_row, s_col
    
    start_path = []
    if 14 <= s_row <= 16 and 12 <= s_col <= 17:
        if ((s_row, s_col) != (14,14)) and (12 <= s_col <= 14):

            complete = False
            while not complete:
                if n_row < 14:
                    n_row += 1
                    start_path.append((n_row, n_col))
                if n_col < 14:
                    n_col += 1
                    start_path.append((n_row, n_col))
                if (n_row, n_col) == (14,14):
                    complete = True
                    
            [start_path.append(point) for point in [(13,14), (12,14), (12,13)]]
            start_node = (12,13)
            
        if (s_row, s_col) == (14,14):
            [start_path.append(point) for point in [(13,14), (12,14), (12,13)]]
            start_node = (12,13)

        if (s_row, s_col) != (14,15) and 15 <= s_col <= 17:
            complete = False
            while not complete:
                if n_row > 14:
                    n_row -= 1
                    start_path.append((n_row, n_col))
                if n_col > 15:
                    n_col -= 1
                    start_path.append((n_row, n_col))
                if (n_row, n_col) == (14,15):
                    complete = True
                    
            [start_path.append(point) for point in [(13,15), (12,15), (12,16)]]
            start_node = (12,16)
        
        if (s_row, s_col) == (14,15):
            [start_path.append(point) for point in [(13,15), (12,15), (12,16)]]
            start_node = (12,16)

    else:
        path_and_node = start_point.find_nearest_node()
        print(start_point.position, "start", path_and_node)
        if isinstance(path_and_node[0], tuple) or isinstance(path_and_node[0], list):
            start_path, start_node = path_and_node
            # need to reverse end node
            
        elif isinstance(path_and_node[0], int):
            start_path = []
            start_node = path_and_node
    
    final_path = []
    
    
    if end_point.position in positions:
        end_path = [end_point.position]
        end_node = end_point.position
    else:
        # chech if point is adj to a node in which case only the node is returned. Check if tuple (path to node) or int(no path just the node)
        path_and_node = end_point.find_nearest_node()
        print(end_point.position, "end", path_and_node)
        if isinstance(path_and_node[0], tuple) or isinstance(path_and_node[0], list):
            end_path, end_node = path_and_node
            # need to reverse end node
            end_path = end_path[::-1]
            end_path = end_path[1:]
            end_path.append(end_point.position)
            
        elif isinstance(path_and_node[0], int):
            end_path = []
            end_node = path_and_node
            
    middle_path = A_star(graph_of_nodes, Node(start_node), Node(end_node))

    final_middle_path = []
    
    for i in range(len(middle_path) - 1):
        mini_path = (path_node_to_node(middle_path[i] , middle_path[i+1]))
        [final_middle_path.append(node) for node in mini_path]
    
    #return final path
    for point in start_path:
        final_path.append(point)
    
    for point in final_middle_path:
        final_path.append(point)
    
    for point in end_path:
        final_path.append(point)

    for i in range(len(final_path) - 1 ):
        if final_path[i] == final_path[i+1]:
            final_path.pop(i)
    
    for i in range(len(final_path)):

        if final_path[i] in final_path[i+1:]:
            # Find the index of the next occurrence
            next_points = final_path[i+1:]

            next_repeat = next_points.index(final_path[i])

            j = next_repeat + i + 1
            
            del final_path[i:j] 
            break

    return final_path

# contained all of ghost information in ghost class using OOP
class ghost():
    
    def __init__(self, name, position):
        
        global x,y
        global ghost_mode
        self.position = position  # (array i, array j) coordinates
        # if blinky = red ghost moving towards player
        self.name = name
        self.ghost_x , self.ghost_y = arrToCo(position)
        self.route = []
        self.mode_type()
        
        self.exact_track = self.moveToSpot(self.position, self.end_pos, self.route)
        self.next_point = self.return_step(self.exact_track)
        
        
        size = (35, 35)
    
        if self.name == 'Blinky':
            pacman_folder = os.path.join('pacman_icons', 'blinky.png')
            ghost = pygame.image.load('blinky.png')
            ghost = pygame.transform.scale(ghost, size)
            
            self.image = ghost
        
    def moveToSpot(self, start_pos, end_pos, route):
        global x, y  # Pac-Man's position

        self.ghost_x, self.ghost_y = arrToCo(start_pos)

        while True:
            # Calculate the route

            for i in range(len(route) - 1):
                start = arrToCo(route[i])
                end = arrToCo(route[i + 1])
                reached = False
                print("start",route[i])
                print("end", route[i + 1])
                
                while not reached:
                    # Move along x-axis
                    
                    if round(end[0]) == round(self.ghost_x):
                        step = abs(start[1] - end[1]) / 17
                        if self.ghost_y < end[1]:
                            self.ghost_y += step
                        elif self.ghost_y > end[1]:
                            self.ghost_y -= step

                    # Move along y-axis
                    if round(end[1]) == round(self.ghost_y):
                        step = abs(start[0] - end[0]) / 17
                        if self.ghost_x < end[0]:
                            self.ghost_x += step
                        elif self.ghost_x > end[0]:
                            self.ghost_x -= step

                    yield (self.ghost_x, self.ghost_y)

                    # If ghost reaches the destination node
                    if (round(self.ghost_x) == round(end[0])) and (round(self.ghost_y) == round(end[1])):
                        reached = True

            # Update positions for the next iteration
            
            start_pos = route[-1]  # Last node in the route
            end_pos = CoToArr((x, y))  # Pac-Man's current position
            route = final_path(Node(start_pos), Node(end_pos))
            

            
            if start_pos == end_pos:
                print("Target reached or no movement needed.")
                break  # Break if no further route recalculation is needed
    
    def return_step(self, path):
        return next(self.exact_track)

    def mode_type(self):
        if ghost_mode == 'CHASE':
            self.end_pos = CoToArr((x,y))
            self.route = final_path(Node(self.position), Node(self.end_pos))
        



run = True
x, y = 435, 620

blinky = ghost('Blinky', (14,12))


centreX = x + 17.5
centreY = y + 17.5

font = pygame.font.Font("fonts/PressStart2P.ttf", 40)




while run:
    hitbox = pygame.Rect(x, y, 35, 35)

    clock.tick(fps)
    screen.fill('black')
    drawBoard()
    screen.blit(blinky.image, (next(blinky.exact_track)))
    screen.blit(pacman, (x,y))
    redraw()

    text = font.render(str(points), True, (255, 255, 255))
    screen.blit(text, (25, 850))
   
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RIGHT:
                direction = 'right'
            if event.key == pygame.K_DOWN:
                direction = 'down'
            if event.key == pygame.K_LEFT:
                direction = 'left'
            if event.key == pygame.K_UP:
                direction = 'up'

    
    pygame.display.flip()

