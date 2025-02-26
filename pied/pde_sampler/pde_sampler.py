class PDESampler:
    
    def __init__(self):
        self.inv_params = None
        self.exp_params = None
    
    def set_inv_params(self, inv_params):
        self.inv_params = inv_params
    
    def set_exp_params(self, exp_params):
        self.exp_params = exp_params
    
    def prep_simulator(self):
        raise NotImplementedError
    
    def log_likelihood(self, xs, ys):
        raise NotImplementedError
    
    def sample(self, xs, rng):
        raise NotImplementedError
    
    def generate_intermediate_info(self):
        raise NotImplementedError
    
    def reset(self):
        raise NotImplementedError
