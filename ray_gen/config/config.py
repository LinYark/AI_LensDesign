class TrainCfg:
    def __init__(self):
        self.sys_num = 3  # EPD,FILED,NA,F_NUM
        self.input_scale = [[15, 25], [12, 12], [0.2, 0.4], [0.15, 0.3]]
        self.output_scale = [[25, 30], [150, 155]]  # g_thick,a_thick
