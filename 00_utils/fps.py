# -*- coding: utf-8 -*-
import time

class FPS(object):

    """FPS Counter"""

    def __init__(self, moving_average=30):
        """
        Args:
            moving_average (int): recent N frames moving average
        """
        self.moving_average = moving_average
        self.prev_time = time.time()
        self.dtimes = []

    def update(self):
        """
        Update FPS.
        Returns:
            fps: current fps
        """
        cur_time = time.time()
        dtime = cur_time - self.prev_time
        self.prev_time = cur_time
        self.dtimes.append(dtime)
        if len(self.dtimes) > self.moving_average:
            self.dtimes.pop(0)
        return self.get()

    def get(self):
        """
        Get FPS.
        Returns:
            fps: current fps
        """
        if len(self.dtimes) == 0:
            return None
        else:
            return len(self.dtimes) / sum(self.dtimes)
    
    def to_string(self):
        return 'fps: %.2f' % self.get()
