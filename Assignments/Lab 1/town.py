'''

The idea is simple 

 s = (i,j,k,l);
 A = (u,d,l,r,s);
 T(s'|s,a=action of player) = 
 R(s,a) = Average utility value for taking this action or expected value.  Sum(Utility of a state * probability)
 
'''
import numpy as np
import matplotlib.pyplot as plt
import time
from IPython import display
import random

# Implemented methods
methods = ['DynProg', 'ValIter'];

# Some colours
LIGHT_RED    = '#FFC4CC';
LIGHT_GREEN  = '#95FD99';
BLACK        = '#000000';
WHITE        = '#FFFFFF';
LIGHT_PURPLE = '#E8D0FF';
LIGHT_ORANGE = '#FAE0C3';

class Town:

    # Actions
    STAY       = 0
    MOVE_LEFT  = 1
    MOVE_RIGHT = 2
    MOVE_UP    = 3
    MOVE_DOWN  = 4

    # Give names to actions
    actions_names = {
        STAY: "stay",
        MOVE_LEFT: "move left",
        MOVE_RIGHT: "move right",
        MOVE_UP: "move up",
        MOVE_DOWN: "move down"
    }

    # Reward values
    STEP_REWARD = -1
    GOAL_REWARD = 1
    CAUGHT_REWARD = -10# Set Reward for getting caught
    IMPOSSIBLE_REWARD = -100


    def __init__(self, maze, weights=None, random_rewards=False):
        """ Constructor of the environment Maze.
        """
        self.maze                     = maze;
        self.actions                  = self.__actions();
        self.states, self.map         = self.__states();
        self.n_actions                = len(self.actions);
        self.n_states                 = len(self.states);
        self.rewards                  = self.__rewards(weights=weights,random_rewards=random_rewards);

# ACTIONS AND STATES ARE PERFECT
    def __actions(self):
        actions = dict();
        actions[self.STAY]       = (0, 0);
        actions[self.MOVE_LEFT]  = (0,-1);
        actions[self.MOVE_RIGHT] = (0, 1);
        actions[self.MOVE_UP]    = (-1,0);
        actions[self.MOVE_DOWN]  = (1,0);
        return actions;

    def __states(self):
        states = dict();
        map = dict();
        end = False;
        s = 0;
        for i in range(self.maze.shape[0]): # X?
            for j in range(self.maze.shape[1]): # Y?
                for k in range(self.maze.shape[0]):
                    for l in range(self.maze.shape[1]):
                        if True:
                            states[s] = (i,j,k,l); # make the state as a quadraple
                            map[(i,j,k,l)] = s;
                            s += 1;
        return states, map
# Move is also perfect
    def __move(self, state, action):
        row = self.states[state][0] + self.actions[action][0];
        
        col = self.states[state][1] + self.actions[action][1];
        row_m = self.states[state][2]
        col_m = self.states[state][3]
        hitting_maze_walls =  (row == -1) or (row == self.maze.shape[0]) or \
                              (col == -1) or (col == self.maze.shape[1])
        moved_state = (row,col,row_m,col_m)
        if hitting_maze_walls:
           return state;
        else:
           return self.map[(row, col,row_m,col_m)];   

    def random_walk(self,state,action):
        row = self.states[state][0] ;
        col = self.states[state][1] ;
        row_m = self.states[state][2]+ self.actions[action][0];
        col_m = self.states[state][3]+ self.actions[action][1];
        
        hitting_walls =  (row_m == -1) or (row_m == self.maze.shape[0]) or \
                              (col_m == -1) or (col_m == self.maze.shape[1]) 
        moved_state = (row,col,row_m,col_m)
        if hitting_walls:
           return state;
        else:
            return self.map[(row, col,row_m,col_m)];     
    def __find_valid_moves(self,current_state,agent,isStayAllowed):
        valid_action =[];
        number_of_actions =0;
        i =1;
        if(isStayAllowed):
            i=0;
        for action in range(i,self.n_actions):
            if(agent=='police'):
                next_state = self.random_walk(current_state,action);
                
                if(next_state == current_state):
                    if(action!=self.STAY):
                        continue;
                    else:
                        number_of_actions = number_of_actions+1;
                        valid_action.append(action)
                elif(next_state!=current_state):
                    number_of_actions = number_of_actions +1;
                    valid_action.append(action)
                            
                    
            elif agent =='player':
                next_state = self.__move(current_state,action);
                if next_state == current_state and action == self.STAY:
                    if(isStayAllowed):
                        number_of_actions = number_of_actions+1;
                        valid_action.append(action);
                elif next_state!=current_state:
                    number_of_actions = number_of_actions+1;
                    valid_action.append(action);
        return valid_action,number_of_actions                
    

    def simulate(self, start, policy, method):
        if method not in methods:
            error = 'ERROR: the argument method must be in {}'.format(methods);
            raise NameError(error);

        path = list();
        if method == 'DynProg':
            # Deduce the horizon from the policy shape
            horizon = policy.shape[1];
            # Initialize current state and time
            t = 0;
            s = self.map[start];
            # Add the starting position in the maze to the path
            path.append(start);
            print(path)
            print(policy[s,t])
            while t < horizon-1:
                # Move to next state given the policy and the current state
                temp_s = self.__move(s,policy[s,t]);
                
                if(temp_s==s and policy[s,t]!=self.STAY):
                    Exception('Wrong policy');
                print('Current state is ',s,'-> policy',policy[s,t],temp_s);
                
                while True:
                    a_m = random.randint(1,4);
                    next_s = self.random_walk(temp_s,a_m);
                    if(next_s != temp_s):
                        break;
                #policy determines best action for player but not for minotaur which is random    
                # Confused about how to account for the minotaur's move
                
                
                # Add the position in the maze corresponding to the next state
                # to the path
                path.append(self.states[next_s])
                # Update time and state for next iteration
                t +=1;
                s = next_s;
        if method == 'ValIter':
            # Initialize current state, next state and time
            t = 1;
            s = self.map[start];
            # Add the starting position in the maze to the path
            path.append(start);
            
            print(' The action chosen is ',policy[s])
            # Move to next state given the policy and the current state
            temp_s = self.__move(s,policy[s]);
            if(temp_s==s and policy[s]!=self.STAY):
                Exception ('cant happen something wrong with policy');
            
            print('The result is ',s,'->',temp_s)
            while True:
                    a_m = random.randint(1,4)
                    next_s = self.__move(s,a_m);
                    if(next_s!=temp_s):
                        break;
            # Add the position in the maze corresponding to the next state
            # to the path
            path.append(self.states[next_s]);
            # Loop while state is not the goal state
            while s != next_s:
                # Update state
                s = next_s;
                # Move to next state given the policy and the current state
                temp_s = self.__move(s,policy[s]);
                while True:
                    a_m = random.randint(1,4)
                    next_s = self.__move(s,policy[s]);
                    if(next_s!=temp_s):
                        break;
                # Add the position in the maze corresponding to the next state
                # to the path
                path.append(self.states[next_s])
                # Update time and state for next iteration
                t +=1;
        return path


    def show(self):
        print('The states are :')
        print(self.states)
        print('The actions are:')
        print(self.actions)
        print('The mapping of the states:')
        print(self.map)
        print('The rewards:')
        print(self.rewards)

def dynamic_programming(env, horizon):
    """ Solves the shortest path problem using dynamic programming
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input int horizon        : The time T up to which we solve the problem.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """

    # The dynamic prgramming requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    p         = env.transition_probabilities;
    r         = env.rewards;
    n_states  = env.n_states;
    n_actions = env.n_actions;
    T         = horizon;

    # The variables involved in the dynamic programming backwards recursions
    V      = np.zeros((n_states, T+1));
    policy = np.zeros((n_states, T+1));
    Q      = np.zeros((n_states, n_actions));


    # Initialization
    Q            = np.copy(r);
    V[:, T]      = np.max(Q,1);
    policy[:, T] = np.argmax(Q,1);

    # The dynamic programming bakwards recursion
    for t in range(T-1,-1,-1):
        # Update the value function acccording to the bellman equation
        for s in range(n_states):
            for a in range(n_actions):
                # Update of the temporary Q values
                Q[s,a] = r[s,a] + np.dot(p[:,s,a],V[:,t+1])
        # Update by taking the maximum Q value w.r.t the action a
        V[:,t] = np.max(Q,1);
        # The optimal action is the one that maximizes the Q function
        policy[:,t] = np.argmax(Q,1);
    return V, policy;

def value_iteration(env, gamma, epsilon):
    """ Solves the shortest path problem using value iteration
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input float gamma        : The discount factor.
        :input float epsilon      : accuracy of the value iteration procedure.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """
    # The value itearation algorithm requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    p         = env.transition_probabilities;
    r         = env.rewards;
    n_states  = env.n_states;
    n_actions = env.n_actions;

    # Required variables and temporary ones for the VI to run
    V   = np.zeros(n_states);
    Q   = np.zeros((n_states, n_actions));
    BV  = np.zeros(n_states);
    # Iteration counter
    n   = 0;
    # Tolerance error
    tol = (1 - gamma)* epsilon/gamma;

    # Initialization of the VI
    for s in range(n_states):
        for a in range(n_actions):
            Q[s, a] = r[s, a] + gamma*np.dot(p[:,s,a],V);
    BV = np.max(Q, 1);

    # Iterate until convergence
    while np.linalg.norm(V - BV) >= tol and n < 200:
        # Increment by one the numbers of iteration
        n += 1;
        # Update the value function
        V = np.copy(BV);
        # Compute the new BV
        for s in range(n_states):
            for a in range(n_actions):
                Q[s, a] = r[s, a] + gamma*np.dot(p[:,s,a],V);
        BV = np.max(Q, 1);
        # Show error
        #print(np.linalg.norm(V - BV))

    # Compute policy
    policy = np.argmax(Q,1);
    # Return the obtained policy
    return V, policy;

def draw_maze(maze):

    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED};

    # Give a color to each cell
    rows,cols    = maze.shape;
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows));

    # Remove the axis ticks and add title title
    ax = plt.gca();
    ax.set_title('The Maze');
    ax.set_xticks([]);
    ax.set_yticks([]);

    # Give a color to each cell
    rows,cols    = maze.shape;
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                            cellColours=colored_maze,
                            cellLoc='center',
                            loc=(0,0),
                            edges='closed');
    # Modify the hight and width of the cells in the table
    tc = grid.properties()['child_artists']
    for cell in tc:
        cell.set_height(1.0/rows);
        cell.set_width(1.0/cols);

def animate_solution(maze, path):

    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED};

    # Size of the maze
    rows,cols = maze.shape;

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows));

    # Remove the axis ticks and add title title
    ax = plt.gca();
    ax.set_title('Policy simulation');
    ax.set_xticks([]);
    ax.set_yticks([]);

    # Give a color to each cell
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                     cellColours=colored_maze,
                     cellLoc='center',
                     loc=(0,0),
                     edges='closed');

    # Modify the hight and width of the cells in the table
    tc = grid.properties()['child_artists']
    for cell in tc:
        cell.set_height(1.0/rows);
        cell.set_width(1.0/cols);


    # Update the color at each frame

    for i in range(len(path)):
        print(path[i]);
        grid.get_celld()[(path[i][0],path[i][1])].set_facecolor(LIGHT_ORANGE)
        grid.get_celld()[(path[i][0],path[i][1])].get_text().set_text('Player')
        grid.get_celld()[(path[i][2],path[i][3])].set_facecolor(LIGHT_RED)
        grid.get_celld()[(path[i][2],path[i][3])].get_text().set_text('Minotaur')
        
        if i > 0:
            if path[i] == path[i-1]:
                grid.get_celld()[(path[i][0],path[i][1])].set_facecolor(LIGHT_GREEN)
                grid.get_celld()[(path[i][0],path[i][1])].get_text().set_text('Player is out')
            else:
                # grid.get_celld()[(path[i-1][0],path[i-1][1])].set_facecolor(col_map[maze[path[i-1][0],path[i-1][1]]])
                # grid.get_celld()[(path[i-1][0],path[i-1][1])].get_text().set_text('')
                grid.get_celld()[(path[i-1][2],path[i-1][3])].set_facecolor(col_map[maze[path[i-1][2],path[i-1][3]]])
                grid.get_celld()[(path[i-1][2],path[i-1][3])].get_text().set_text('')
                
                
        display.display(fig)
        display.clear_output(wait=True)
        time.sleep(1)
