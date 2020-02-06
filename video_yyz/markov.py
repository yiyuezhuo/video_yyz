import numpy as np

class MarkovModel:
    def __init__(self, *, prior, transition_matrix, likelihood_list):
        self.prior = prior
        self.transition_matrix = transition_matrix
        self.likelihood_list = likelihood_list
        
        self.n = len(self.prior)
        self.p = self.prior
    
    def predict(self):
        self.p = self.p @ self.transition_matrix
    
    def update(self, prob):
        joint_arr = np.zeros(self.n)
        for i, (p, likelihood) in enumerate(zip(self.p, self.likelihood_list)):
            joint_arr[i] = p * likelihood(prob)
        self.p = joint_arr / joint_arr.sum()
