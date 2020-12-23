from torch.utils.tensorboard import SummaryWriter

class SummaryHelper(object):
    def __init__(self, distributed_rank=0, log_dir=None):
        if distributed_rank == 0:
            self.w = SummaryWriter(log_dir=log_dir)
        else:
            self.w = None
        self.step = 0

    def add_scalar(self, key, value):
        if self.w is None: return
        self.w.add_scalar(key, value, self.step)

    def add_text(self, key, value):
        if self.w is None: return
        self.w.add_text(key, value)

    def add_images(self, key, value):
        if self.w is None: return
        self.w.add_images(key, value, self.step)

    def set_step(self, step):
        self.step = step
