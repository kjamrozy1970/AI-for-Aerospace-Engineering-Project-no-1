import cmath
import math

class Rover:
    def __init__(self, x, y, theta, laser_on):
        self.x_list = [x]
        self.y_list = [y]
        self.theta_list = [theta]
        self.title = "rover"
        self.time = 0
        self.time_laser = 0
        self.velocity = 0.5
        self.pos = []

    def move(self, x_dest, y_dest):
        rover_pos = complex(self.x_list[-1], self.y_list[-1])
        new_pos = complex(x_dest, y_dest)
        rover_pos += new_pos
        self.pos.append(new_pos)
        self.time += cmath.sqrt(self.pos[-1].real**2 + self.pos[-1].imag**2) * self.velocity

        self.x_list.append(rover_pos.real)
        self.y_list.append(rover_pos.imag)
        self.theta_list.append((-cmath.phase(self.pos[-1])) * 180/math.pi + 90)

