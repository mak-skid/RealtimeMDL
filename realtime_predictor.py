from training.mdl_model import get_MDL_model


class RealTimePredictor:
    def __init__(self):
        self.bootstrap_servers = 'localhost:9092'    

        self.start = 2229.5 # after spliting the test and train data 80%:20%
        self.end = 2772 # max_elapsed_time=2772000 ms = 46.2 minutes if hour is 7-8am
        self.predict_len = 1

        self.timewindow = 10 #0.5
        self.num_section_splits = 9
        self.num_lanes = 5
        self.history_len = 6 #20
        self.num_skip = 1 #10
        self.max_dist = 2224.6633212502065
        self.num_sections = self.num_section_splits + 1

        self.with_ramp = False
        self.with_ramp_sign = "w" if self.with_ramp else "wo"
        self.num_features = 1

        self.model = get_MDL_model(self.history_len, self.num_lanes, self.num_sections)