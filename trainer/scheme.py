import json

class Scheme:
    def __init__(self, init_lr=0.1, 
                 lr_tuning_method = 'step',  # Step/Exp/Specify
                 lr_tuning_points = [60, 120, 160], lr_tuning_rates = [0.2, 0.2, 0.2], # config of StepLR
                 warm_up_epoch=1,
                 lr_func = None) -> None:
        self.init_lr = init_lr
        self.warm_up_epoch = warm_up_epoch

        self.lr_tuning_method = lr_tuning_method
        self.lr_tuning_points = lr_tuning_points
        self.lr_tuning_rates = lr_tuning_rates

        self.lr_func = lr_func

