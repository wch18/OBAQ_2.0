import json

class Q_Scheme:
    def __init__(self) -> None:
        self.update_period = 1
        self.target_bit_W = 4
        self.target_bit_bA = 4
        self.K_update_mode = 'BinarySearch'
        self.model_type = 'BFP'
        self.bwmap_smooth = 0.5

