# simulates the device moving around the edges of the workspace.  Plots the
# device at the 4 corners

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import fwd_kinematics
import inv_kinematics

# clear
# clc
# close all
# set(0,'defaultfigurewindowstyle','docked')

# plot?
plot_linkage = True

# max current per motor
Imax = 6  # A
kt = 0.0234  # Nm/A
Tmax = Imax * kt * np.ones([2, 1])  # Nm

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

# find solve inverse kinematics at the boundary of the paper
number = 300
xpoints1 = np.linspace(xcenter, xcenter + w / 2, number)
xpoints2 = np.linspace(xcenter + w / 2, xcenter + w / 2, number)
xpoints3 = np.linspace(xcenter + w / 2, xcenter - w / 2, number)
xpoints4 = np.linspace(xcenter - w / 2, xcenter - w / 2, number)
xpoints5 = np.linspace(xcenter - w / 2, xcenter + w / 2, number)

xpoints = np.concatenate((xpoints1, xpoints2, xpoints3, xpoints4, xpoints5))

ypoints1 = np.linspace(ycenter, ycenter + h / 2, number)
ypoints2 = np.linspace(ycenter + h / 2, ycenter - h / 2, number)
ypoints3 = np.linspace(ycenter - h / 2, ycenter - h / 2, number)
ypoints4 = np.linspace(ycenter - h / 2, ycenter + h / 2, number)
ypoints5 = np.linspace(ycenter + h / 2, ycenter + h / 2, number)

ypoints = np.concatenate((ypoints1, ypoints2, ypoints3, ypoints4, ypoints5))

theta = np.zeros([2, len(xpoints)])
for i in range(len(xpoints)):
    # #find joint angles
    x3 = xpoints[i].item()
    y3 = ypoints[i].item()

    theta = inv_kinematics.inv_kinematics(a1, a2, a3, a4, a5, x3, y3)
    t1 = theta[0].item()
    t5 = theta[1].item()

    # jacobian
    # J(:,:,i) = fwd_kinematics.fwd_kinematics(a1,a2,a3,a4,a5,t1,t5,plot_linkage)

    # figure set up
    # xlabel('-x');ylabel('-y')
    # set(gcf,'color','white');
    # axis equal
# axis([-.25 .3 -.45 0.03])
# grid on

# plot the rectangle
# rectangle('Position',[-(xcenter+w/2) -(ycenter+h/2) w h])

# frame(i) = getframe;


# -----------------------------Plot 4 positions---------------------
fig, axs = plt.subplots(2, 2)

# figure
# subplot(2,2,1)
theta = inv_kinematics.inv_kinematics(a1, a2, a3, a4, a5, max(xpoints), max(ypoints))
t1 = theta[0]
t5 = theta[1]
fwd_kinematics.fwd_kinematics(a1, a2, a3, a4, a5, t1, t5, axs[0, 0], plot_linkage)
# figure set up
# xlabel('-x');ylabel('-y')
# set(gcf,'color','white');
axs[0, 0].axis("equal")
# axis([-.25 .35 -.45 0.03])
axs[0, 0].set_xlim(-0.25, 0.35)
axs[0, 0].set_ylim(-0.45, 0.03)
axs[0, 0].grid()
# grid on
axs[0, 0].add_patch(
    Rectangle((-(xcenter + w / 2), -(ycenter + h / 2)), w, h, fill=False)
)
# rectangle('Position',[-(xcenter+w/2) -(ycenter+h/2) w h])


# subplot(2,2,2)
theta = inv_kinematics.inv_kinematics(a1, a2, a3, a4, a5, min(xpoints), max(ypoints))
t1 = theta[0]
t5 = theta[1]
fwd_kinematics.fwd_kinematics(a1, a2, a3, a4, a5, t1, t5, axs[0, 1], plot_linkage)
# figure set up
# xlabel('-x');ylabel('-y')
# set(gcf,'color','white');
axs[0, 1].axis("equal")
# axis equal
# axis([-.25 .35 -.45 0.03])
axs[0, 1].set_xlim(-0.25, 0.35)
axs[0, 1].set_ylim(-0.45, 0.03)
axs[0, 1].grid()
# grid on
axs[0, 1].add_patch(
    Rectangle((-(xcenter + w / 2), -(ycenter + h / 2)), w, h, fill=False)
)
# rectangle('Position',[-(xcenter+w/2) -(ycenter+h/2) w h])

# subplot(2,2,3)
theta = inv_kinematics.inv_kinematics(a1, a2, a3, a4, a5, max(xpoints), min(ypoints))
t1 = theta[0]
t5 = theta[1]
fwd_kinematics.fwd_kinematics(a1, a2, a3, a4, a5, t1, t5, axs[1, 0], plot_linkage)
# figure set up
# xlabel('-x');ylabel('-y')
# set(gcf,'color','white');
axs[1, 0].axis("equal")
# axis equal
# axis([-.25 .35 -.45 0.03])
axs[1, 0].set_xlim(-0.25, 0.35)
axs[1, 0].set_ylim(-0.45, 0.03)
axs[1, 0].grid()
# grid on
axs[1, 0].add_patch(
    Rectangle((-(xcenter + w / 2), -(ycenter + h / 2)), w, h, fill=False)
)
# rectangle('Position',[-(xcenter+w/2) -(ycenter+h/2) w h])

# subplot(2,2,4)
theta = inv_kinematics.inv_kinematics(a1, a2, a3, a4, a5, min(xpoints), min(ypoints))
t1 = theta[0]
t5 = theta[1]
fwd_kinematics.fwd_kinematics(a1, a2, a3, a4, a5, t1, t5, axs[1, 1], plot_linkage)
# figure set up
# xlabel('-x');ylabel('-y')
# set(gcf,'color','white');
axs[1, 1].axis("equal")
# axis equal
# axis([-.25 .35 -.45 0.03])
axs[1, 1].set_xlim(-0.25, 0.35)
axs[1, 1].set_ylim(-0.45, 0.03)
axs[1, 1].grid()
# grid on
axs[1, 1].add_patch(
    Rectangle((-(xcenter + w / 2), -(ycenter + h / 2)), w, h, fill=False)
)
# rectangle('Position',[-(xcenter+w/2) -(ycenter+h/2) w h])


plt.show()
