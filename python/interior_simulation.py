# plots the jacobian and minimum-maximum forces over the workspace

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import fwd_kinematics
import inv_kinematics

# clear
# clc
# close all
# set(0,'defaultfigurewindowstyle','docked')

# plot the linkage at each step?
plot_linkage = False

# force angles
angle_number = 20
angle_range = 2 * np.pi / angle_number

# max current per motor
kt = 0.0234

# based on stall torue (max peak)
Tmax = 0.129
Imax = Tmax / kt

# based on maximum continuous
# Imax = 1.2; #A
# Tmax = kt*Imax;

# capstan mechanial advantage ratio
#ratio = 0.07303 / (0.0131 / 2)
ratio = 1

# link lengths
a1 = 0.25
a2 = 0.25
a3 = a2
a4 = a1
a5 = 0.1

# origin
x1 = 0
y1 = 0

# the rectangle
xcenter = -(x1 + a5) / 2
ycenter = 0.34
w = 0.2794
h = 0.2159


# define how many points to solve for
number = 10
xpoints = np.linspace(xcenter + w / 2, xcenter - w / 2, number)
ypoints = np.linspace(ycenter + h / 2, ycenter - h / 2, number)

# preallocate
condition = np.zeros([len(xpoints), len(ypoints)])
theta = np.zeros([2, len(xpoints)])
mean_force = np.zeros([len(xpoints), len(ypoints)])
min_force = np.zeros([len(xpoints), len(ypoints)])


# at every point
for i in range(len(xpoints)):
    for j in range(len(ypoints)):
        # find joint angles
        x3 = xpoints[i].item()
        y3 = ypoints[j].item()

        theta = inv_kinematics.inv_kinematics(a1, a2, a3, a4, a5, x3, y3)
        t1 = theta[0].item()
        t5 = theta[1].item()

        # jacobian
        J = fwd_kinematics.fwd_kinematics(
            a1, a2, a3, a4, a5, t1, t5, None, plot_linkage
        )

        Jinv = np.linalg.inv(J)

        # condition of jacobian
        condition[i][j] = np.linalg.cond(J)

        # solve torque needed for 10 force directions evenly spaced
        # increase the magnitue of the force until T1 or T2 > Tmax
        # report the largest and the smallest

        # for each point
        max_dir_force = np.zeros([angle_number])
        for k in range(angle_number):
            mag = 0
            max_force_reached = False
            # while we have not exceeded the maximum toruqe
            while not max_force_reached:
                # find the force in a particular direction
                # TODO: no need for this multiplication
                # i think this is 2x1
                f = (
                    mag
                    * np.array(
                        [
                            [np.cos(k * angle_range), -np.sin(k * angle_range)],
                            #[np.cos(angle_range), -np.sin(k * angle_range)],
                            [np.sin(k * angle_range), np.cos(k * angle_range)],
                        ]
                    )
                    @ np.array([[1], [0]])
                )
                # compute the needed motor torque
                # so this is also 2x1
                torque = np.transpose(J) @ f
                # if we have  exceeded spec (90% of max current)
                if abs(torque[0].item()) >= (Tmax) * ratio or abs(torque[1].item()) >= (Tmax) * ratio:
                    # then we have reached the max force in that direction
                    max_force_reached = True
                else:
                    # otherwise increment the magnitude
                    mag = mag + 0.01

            max_dir_force[k] = mag

        mean_force[i, j] = np.mean(max_dir_force)
        min_force[i, j] = np.min(max_dir_force)


# FIGURE 1
# figure set up
# xlabel('-x');ylabel('-y')
plt.figure()
ax = plt.gca()
# set(gcf,'color','white');
ax.axis("equal")
# axis equal
# axis([-.2 .35 -.45 0.03])
#ax.set_xlim(-0.2, 0.35)
#ax.set_ylim(-0.45, 0.03)
# grid on
ax.grid()
# hold on

# plot one linkage configuration
t1iso = 0.8719
t5iso = 2.2697
J = fwd_kinematics.fwd_kinematics(a1, a2, a3, a4, a5, t1iso, t5iso, ax, True)

# plot the rectangle
# rectangle('Position',[-(xcenter+w/2) -(ycenter+h/2) w h])
ax.add_patch(Rectangle((-(xcenter + w / 2), -(ycenter + h / 2)), w, h, fill=False))

#print(xpoints) # 1xn
#print(ypoints) # 1xn
#print(condition) # nxn
# plot the condition number contour
# v = [1:.5:4]
# [c handle] = contour(-xpoints,-ypoints,condition', v);
# clabel(c,handle,v)

CS = ax.contour(-xpoints, -ypoints, np.transpose(condition))
ax.clabel(CS)

ax.set_title("condition")

#plt.show()


# FIGURE 2
# figure set up
# figure
# hold on
# xlabel('-x [m]');ylabel('-y [m]')
# set(gcf,'color','white');
plt.figure()
ax = plt.gca()
ax.axis("equal")
# axis equal
# axis([-.2 .35 -.45 0.03])
#ax.set_xlim(-0.2, 0.35)
#ax.set_ylim(-0.45, 0.03)
# grid on
ax.grid()

# plot one linkage configuration
t1iso = 0.8719
t5iso = 2.2697
J = fwd_kinematics.fwd_kinematics(a1, a2, a3, a4, a5, t1iso, t5iso, ax, True)

# plot the rectangle
# rectangle('Position',[-(xcenter+w/2) -(ycenter+h/2) w h])
ax.add_patch(Rectangle((-(xcenter + w / 2), -(ycenter + h / 2)), w, h, fill=False))

# plot the min force contour
# v = 0:.5:5;
# [c handle] = contour(-xpoints,-ypoints,min_force', v);
# clabel(c,handle,v)

CS = ax.contour(-xpoints, -ypoints, np.transpose(min_force))
ax.clabel(CS)

ax.set_title("min force")


plt.show()
