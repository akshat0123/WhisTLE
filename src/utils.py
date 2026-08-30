from collections import deque


class RollingCounter:

    def __init__(self, limit: int=1000):
        """ Counter keeping track of total and rolling average
        Args:
            limit: total size of "rolling" count
        """
        self.rolling_values = deque()
        self.rolling_total = 0
        self.limit = limit
        self.total = 0
        self.count = 0

    def add(self, x: float) -> None:
        """ Add number to rolling counts
        Args:
           x: number to add 
        """
        self.rolling_values.append(x)
        self.rolling_total += x

        if len(self.rolling_values) > self.limit:
            remove = self.rolling_values.popleft()
            self.rolling_total -= remove
            self.total += remove
            self.count += 1

    def total_average(self) -> float:
        count = self.count + len(self.rolling_values)
        total = self.total + self.rolling_total
        avg = total / count if count > 0 else 0
        return avg

    def rolling_average(self) -> float:
        total = self.rolling_total
        count = len(self.rolling_values)
        avg = total / count if count > 0 else 0
        return avg


