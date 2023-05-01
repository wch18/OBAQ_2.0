import json

class Q_Scheme:
    def __init__(self, update_period=1,
                 target_bit_W=2, target_bit_bA=2,
                 K_update_mode = 'BinarySearch', bwmap_smooth=0.5) -> None:
        self.q_type = 'BFP'
        self.update_period = update_period
        self.target_bit_W = target_bit_W
        self.target_bit_bA = target_bit_bA
        self.K_update_mode = K_update_mode
        self.bwmap_smooth = bwmap_smooth
        self.init_bwmap = {
            'A':8,
            'W':4,
            'G':8,
            'bA':4,
        }
        self.block_size = {
            'A':[4,4,1,1],
            'W':[4,4,1,1],
            'G':[4,4,1,1],
            'bA':[4,4,1,1],
        }

