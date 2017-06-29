import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F #all functions from pytorch for neuron activation
import torch.optim as optim #optimizer
from torch.autograd import Variable

#Creating the architecture of the neural network

class Network(nn.Module):
    
    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()

        """input_size is of input variables. Corresponds to the number of vectors that fully describe the state of the agent (in the
        self driving car's case, the input_size is 5, as there are five vectors that describe the car's state: the three
        sensors, positive orientation, and negative orientation"""
        self.input_size = input_size
        self.nb_action = nb_action #number of output variables
        
        """fc stands for full connection, and represents a full connection between two layers of neurons. fc1 is connection 
        between the input layer and the first hidden layer. All neurons in input layer are connected  to all neurons in 
        hidden layer (full connection). nn.Linear and our Network class both draw from nn.Module, which has a __call__ 
        magic method defined to essentially run forward(). Our program makes use of polymorphism as well"""
        self.fc1 = nn.Linear(input_size, 30)
        self.fc2 = nn.Linear(30, nb_action)
            #first parameter is number of neurons in the first layer. Second parameter is number of neurons in the second layer
        
    def forward(self, state):
        """returns the q-values from a forward propagation of the Network with the initial variables state"""
        x = F.relu(self.fc1(state)) #rectifier function that activates the hidden neurons. X represents hidden layer. State represents input neurons for fc1
        q_values = self.fc2(x) #output q values. X represents input neurons for fc2
        return q_values
        
#Implementing Experience Replay to remove bias from similar consecutive states

class ReplayMemory(object):

    def __init__(self, capacity): #Experience replay with the last "capacity" transitions
        self.capacity = capacity #Max number of transitions in memory of events
        self.memory = [] #Here be events
    
    def push(self, event):
        """events come with four values - the state, the action, the reward, and the next state"""
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]


            
    def sample(self, batch_size):
        
        """samples will take a random sample of events from our memory and zip them so that they are arranged in tuples
        that group by column instead of row (i.e. (action 1, action 2, action 3) instead of (action 1, reward 1, state 1)."""
        samples = zip(*random.sample(self.memory, batch_size))        
        """map will make every tuple in samples a torch Variable, with each value in the tuple concatenated by their first
        dimension. 
        
        The return value will be one torch Variable that contains every value of the samples tuples concatenated together
        into one Tensor. torch Variables contain both a Tensor and a Gradient."""
        return map(lambda x: Variable(torch.cat(x, 0)), samples)
    
#Implementing Deep Q Learning. This is the whole Deep Q Learning model

class Dqn():
    
    def __init__(self, input_size, nb_action, gamma):
        """input_size is the number of dimensions in the vectors included in the input state. nb_action is the number of 
        actions the car can take (go left, straight, right). Gamma is the delay coefficient in the dqn model."""
        
        self.gamma = gamma
        self.reward_window = [] #evolving mean of the last 100 rewards. Will contain the last 100 rewards and be constantly upddated
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(100000) #One hundred thousand events in the memory
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001) #lr = learning rate
        
        #5 dimensions of our Dqn are the three car sensors, positive orientation, and negative orientation
        self.last_state = torch.Tensor(input_size).unsqueeze(0) #the network can only accept batches of input observations
        """Normally our Dqn has 5 elements, but we need a fake dimension corresponds to batch size, since the Dqn
        mostly works in batches. The unsqueeze(0) will add in that fake dimension in the [0] position of the
        Tensor (it has to be first).        
        
        This fake dimension isn't a dimension in the same sense of the 5 Dqn dimensions (sensor inputs & orientations).
        Rather, the fake dimension corresponds to another axes within the tensor.
        
        You can think of a Tensor in this context as an array of one type. Most tensors are also wrapped in a variable
        that contains a gradient."""            
        
        self.last_action = 0  #corresponds to the index of the possible actions in action2rotation (maps.py line 33)           
        self.last_reward = 0
        
    def select_action(self, state):
        
        """probs is softmax probabilities of taking each possible action, based on their q-values' proportion to total. 
        Our neural network, represented by self.model, will pass output values to the softmax. The model gets passed the 
        variable state, which is a Torch Tensor, but in a Variable form with volatile = True, so that the gradient of that 
        Tensor isn't included. Volatile also makes it so that the gradient associated with state is not included in the 
        graph of all computations of nn.Module. This is to make the algorithm lighter and save memory.
        7 represents the temperature. If the temperature is higher, softmax will weigh the higher probability actions even
        more and the lower probability actions even less. As the temperature approaches 1, the network is more uncertain
        about which action to play.

        probs.multinomial() returns a pytorch value with a fake batch (fake dimension correspon to the batch size) so
        using action.data[0, 0] will return the valid actions, which are actions 0, 1, or 2"""

        probs = F.softmax(self.model(Variable(state, volatile = True)) * 7)         
        action = probs.multinomial() #random draw of probs distribution
        return action.data[0, 0] 

        
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        
        """Taken in batches because we took a sample of events with our ReplayMemory. The gather method will gather all 
        values in the 1st dimension with an index of batch_action.unsqueeze(1). Squeeze reverts the batch_action back
        to normal without the fake dimension."""
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        
        """next_outputs is required to compute the future value portion of Q-target. this gets the maximum of the Q-values
        of the next state (represented by index 0) according to all actions that are represented by index 1
        
        detach will detach all the outputs from running batch_next_state through our neural network. max will make it so
        that we only return the maximum of the q_values of running batch_next_state thorugh the nn (max is a pytorch method
        that returns the maximum of all elements within the calling tensor; it is not the standard python max). The 1
        in the max parameters represents the index of the action we are getting."""
        next_outputs = self.model(batch_next_state).detach().max(1)[0]        
        target = self.gamma * next_outputs + batch_reward #this is the target equation
        
        """smooth_l1_loss is the best loss function for Deep-Q learning. outputs is the neural network's prediction of
        Q-values and targets is the target which the prediction is going to be compared against."""
        td_loss = F.smooth_l1_loss(outputs, target)

        #here on out is back propagation
        
        """optimizer is used on loss error to perform stochastic gradient descent and update the neural network weights.
        optimizer has to be re-initialized at each iteration of the loop of stochastic gradient descent. zero_grad does that
        reinitialization."""
        self.optimizer.zero_grad()
        
        """Does the back propagation that sends signals back through the neural network and runs stochastic gradient 
        descent and adjust weights."""
        td_loss.backward(retain_variables=True) #retain_variables frees memory
        
        """step updates the weights"""
        self.optimizer.step()      
        
    def update(self, reward, new_signal):
        
        """to update our memory with a new transition"""
        
        """new signal will initially be the last_signal variable in map.py line 130. Remember that the state of the Dqn is 
        the signal composed of the three signals from the car as well as orientation and negative orientation. reward is
        also updated from map.py in its update function, and is what actually happens to our car in the Game.
        
        Remember inputs of the neural network should be Torch tensors, and all new Tensors should have the fake dimension
        corresponding to the batch added in."""
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        
        """At any time t, the transition is composed of current state st, next state st+1, reward rt, and action at. Right
        nowe we have st, rt, and at, and we just received st+1 in new_state. Because we just got a brand new transition, we
        have to append that transition into the memory. The event can be replayed a bunch of times with ReplayMemory.
        self.last_state, new_state, self.last_action, and reward are all components of the transition to be pushed (as a 
        tuples of Tensors)."""
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        
        """play the new action after reaching the new state"""
        action = self.select_action(new_state)
          
        if len(self.memory.memory) > 100:         
            """make batch_state, batch_next_state, batch_reward, and batch_action equal to their respective values within
            100 transitions in the memory"""
            batch_state, batch_next_state, batch_reward, batch_action = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action) #learn from that size 100 batch
        
        """update all the variables on the current transition"""
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        
        """update the last reward to the sliding reward window"""
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        
        """return the action that was just played to reach the new state for map.py line 131"""
        return action
        
    def score(self):
        """mean of all rewards in reward window"""
        return sum(self.reward_window)/(len(self.reward_window) + 1)
        
    def save(self):
        """save your neural network and optimizer so you can use your trained dqn even after you close your application. 
        state_dict is a nn.Module method that returns a dict that represents the state of the whole Module that is calling
        that method.
        
        Neural network and optimizer will be saved to 'last_brain.pth'"""
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict()
                    }, 'last_brain.pth')
        
    def load(self):
        """retrieve your model that you saved"""
        if os.path.isfile('last_brain.pth'): #checks if last_brain.pth exists in your working directory
            print('=> loading checkpoint...')
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict']) #updates your model's state_dict with the loaded one
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print('Done!')
        else:
            print('Error: No checkpoint found')
        
        
        
        
        
        
        
        
        
        
        
        
        