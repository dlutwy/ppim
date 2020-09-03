from torch.utils.tensorboard import SummaryWriter

class LogWriter():
    def __init__(self, log_dir=None):
        if log_dir is None:
            self.writer = SummaryWriter()
        else:
            self.writer = SummaryWriter(log_dir=log_dir)
        self.__stepDict = {}

    def __step(self, phase):
        step = self.__stepDict.get(phase)
        if step is not None:
            self.__stepDict[phase] += 1
            return self.__stepDict.get(phase)
        else:
            self.__stepDict[phase] = 0
            return 0

    def write_log(self, log_name, value):
        phase = log_name.split('/')[-1]
        step = self.__step(phase)
        # print(log_name, value, step)
        self.writer.add_scalar(log_name, value, step)
