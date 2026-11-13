"""Kinematic 2-D rover with visible heading and correct episode reset."""
import math


class Rover:
    ACTIONS = ("left", "forward", "right")

    def __init__(self, x=50.0, y=50.0, theta=0.0,
                 step_size=3.0, turn_angle_deg=15.0):
        self.start_pose = (float(x), float(y), float(theta))
        self.step_size = float(step_size)
        self.turn_angle = math.radians(turn_angle_deg)
        self.reset()

    def reset(self):
        self.x, self.y, self.theta = self.start_pose
        self.theta %= 2 * math.pi
        self.x_list = [self.x]
        self.y_list = [self.y]
        self.theta_list = [self.theta]
        self.action_history = []
        return self.get_state()

    def get_state(self):
        return (self.x, self.y, self.theta)

    def move(self, action, bounds=None):
        """Execute a discrete action; return False if forward hits map bounds.

        A turn changes heading in place and therefore doesn't change x or y.
        Append to history for EVERY action, including turns, so plotting
        can show orientation even when the position remains unchanged.
        """
        if action not in self.ACTIONS:
            raise ValueError(f"Unknown rover action: {action}")

        moved_ok = True
        if action == "left":
            self.theta += self.turn_angle
        elif action == "right":
            self.theta -= self.turn_angle
        else:
            x_new = self.x + self.step_size * math.cos(self.theta)
            y_new = self.y + self.step_size * math.sin(self.theta)
            if bounds is not None:
                xmin, xmax, ymin, ymax = bounds
                if not (xmin <= x_new <= xmax and ymin <= y_new <= ymax):
                    moved_ok = False
            if moved_ok:
                self.x, self.y = x_new, y_new

        self.theta %= 2 * math.pi
        self.x_list.append(self.x)
        self.y_list.append(self.y)
        self.theta_list.append(self.theta)
        self.action_history.append(action)
        return moved_ok
