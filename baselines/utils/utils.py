
'''
'''
class ReasongingTypeAccCalculator():
    def __init__(self, reasoning_types):
        '''
        params: reasoning_types: list of strings
        '''
        self.reasoning_types = reasoning_types
        self.true_count_dct = {}
        self.all_count_dct = {}
        self.acc_dct = {}
        for reasoning_type in self.reasoning_types:
            self.true_count_dct[reasoning_type] = 0
            self.all_count_dct[reasoning_type] = 0
            self.acc_dct[reasoning_type] = 0
    
    def update(self, reasoning_type_lst, pred, label):
        '''
        params: reasoning_type_lst: list of list of strings
        '''
        res = (pred == label)
        for i, q_reasoning_types in enumerate(reasoning_type_lst):
            for reasoning_type in q_reasoning_types:
                if res[i]:
                    self.true_count_dct[reasoning_type] += 1
                self.all_count_dct[reasoning_type] += 1
    
    def reset(self):
        for reasoning_type in self.reasoning_types:
            self.true_count_dct[reasoning_type] = 0
            self.all_count_dct[reasoning_type] = 0
            self.acc_dct[reasoning_type] = 0
    
    def get_acc(self):
        for reasoning_type in self.reasoning_types:
            if self.all_count_dct[reasoning_type] == 0:
                self.acc_dct[reasoning_type] = 0
            else:
                self.acc_dct[reasoning_type] = self.true_count_dct[reasoning_type] / self.all_count_dct[reasoning_type]
        return self.acc_dct