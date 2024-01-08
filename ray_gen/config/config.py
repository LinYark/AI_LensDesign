class TrainCfg:
    def __init__(self):
        self.sys_num = 3  # EPD,FILED,NA,F_NUM
        self.input_scale = [[15, 25], [15, 15], [0.2, 0.25], [0.15, 0.3]]
        self.output_scale = [[10, 13], [100, 110]]
