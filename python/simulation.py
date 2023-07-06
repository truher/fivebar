import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import kinematics


def side():
    """Simulates the device moving around the edges of the workspace.
    Plots the device at the 4 corners."""

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

        theta = kinematics.inverse(a1, a2, a3, a4, a5, x3, y3)
        t1 = theta[0].item()
        t5 = theta[1].item()

    fig, axs = plt.subplots(2, 2)

    theta = kinematics.inverse(a1, a2, a3, a4, a5, max(xpoints), max(ypoints))
    t1 = theta[0]
    t5 = theta[1]
    kinematics.forward(a1, a2, a3, a4, a5, t1, t5, axs[0, 0], plot_linkage)
    axs[0, 0].axis("equal")
    axs[0, 0].set_xlim(-0.25, 0.35)
    axs[0, 0].set_ylim(-0.45, 0.03)
    axs[0, 0].grid()
    axs[0, 0].add_patch(
        Rectangle((-(xcenter + w / 2), -(ycenter + h / 2)), w, h, fill=False)
    )

    theta = kinematics.inverse(a1, a2, a3, a4, a5, min(xpoints), max(ypoints))
    t1 = theta[0]
    t5 = theta[1]
    kinematics.forward(a1, a2, a3, a4, a5, t1, t5, axs[0, 1], plot_linkage)
    axs[0, 1].axis("equal")
    axs[0, 1].set_xlim(-0.25, 0.35)
    axs[0, 1].set_ylim(-0.45, 0.03)
    axs[0, 1].grid()
    axs[0, 1].add_patch(
        Rectangle((-(xcenter + w / 2), -(ycenter + h / 2)), w, h, fill=False)
    )

    theta = kinematics.inverse(a1, a2, a3, a4, a5, max(xpoints), min(ypoints))
    t1 = theta[0]
    t5 = theta[1]
    kinematics.forward(a1, a2, a3, a4, a5, t1, t5, axs[1, 0], plot_linkage)
    axs[1, 0].axis("equal")
    axs[1, 0].set_xlim(-0.25, 0.35)
    axs[1, 0].set_ylim(-0.45, 0.03)
    axs[1, 0].grid()
    axs[1, 0].add_patch(
        Rectangle((-(xcenter + w / 2), -(ycenter + h / 2)), w, h, fill=False)
    )

    theta = kinematics.inverse(a1, a2, a3, a4, a5, min(xpoints), min(ypoints))
    t1 = theta[0]
    t5 = theta[1]
    kinematics.forward(a1, a2, a3, a4, a5, t1, t5, axs[1, 1], plot_linkage)
    axs[1, 1].axis("equal")
    axs[1, 1].set_xlim(-0.25, 0.35)
    axs[1, 1].set_ylim(-0.45, 0.03)
    axs[1, 1].grid()
    axs[1, 1].add_patch(
        Rectangle((-(xcenter + w / 2), -(ycenter + h / 2)), w, h, fill=False)
    )


def interior():
    """Plots the jacobian and minimum-maximum forces over the workspace."""

    # plot the linkage at each step
    plot_linkage = False

    # force angles
    angle_number = 20
    angle_range = 2 * np.pi / angle_number

    # max current per motor
    kt = 0.0234

    # based on stall torue (max peak)
    Tmax = 0.129
    Imax = Tmax / kt

    # capstan mechanial advantage ratio
    # ratio = 0.07303 / (0.0131 / 2)
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

            theta = kinematics.inverse(a1, a2, a3, a4, a5, x3, y3)
            t1 = theta[0].item()
            t5 = theta[1].item()

            # jacobian
            J = kinematics.forward(a1, a2, a3, a4, a5, t1, t5, None, plot_linkage)

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
                                # [np.cos(angle_range), -np.sin(k * angle_range)],
                                [np.sin(k * angle_range), np.cos(k * angle_range)],
                            ]
                        )
                        @ np.array([[1], [0]])
                    )
                    # compute the needed motor torque
                    # so this is also 2x1
                    torque = np.transpose(J) @ f
                    # if we have  exceeded spec (90% of max current)
                    if (
                        abs(torque[0].item()) >= (Tmax) * ratio
                        or abs(torque[1].item()) >= (Tmax) * ratio
                    ):
                        # then we have reached the max force in that direction
                        max_force_reached = True
                    else:
                        # otherwise increment the magnitude
                        mag = mag + 0.01

                max_dir_force[k] = mag

            mean_force[i, j] = np.mean(max_dir_force)
            min_force[i, j] = np.min(max_dir_force)

    plt.figure()
    ax = plt.gca()
    ax.axis("equal")
    ax.set_xlim(-0.2, 0.35)
    ax.set_ylim(-0.45, 0.03)
    ax.grid()

    t1iso = 0.8719
    t5iso = 2.2697
    J = kinematics.forward(a1, a2, a3, a4, a5, t1iso, t5iso, ax, True)

    ax.add_patch(Rectangle((-(xcenter + w / 2), -(ycenter + h / 2)), w, h, fill=False))

    CS = ax.contour(-xpoints, -ypoints, np.transpose(condition))
    ax.clabel(CS)

    ax.set_title("condition")

    plt.figure()
    ax = plt.gca()
    ax.axis("equal")
    ax.set_xlim(-0.2, 0.35)
    ax.set_ylim(-0.45, 0.03)
    ax.grid()

    # plot one linkage configuration
    t1iso = 0.8719
    t5iso = 2.2697
    J = kinematics.forward(a1, a2, a3, a4, a5, t1iso, t5iso, ax, True)

    ax.add_patch(Rectangle((-(xcenter + w / 2), -(ycenter + h / 2)), w, h, fill=False))

    CS = ax.contour(-xpoints, -ypoints, np.transpose(min_force))
    ax.clabel(CS)

    ax.set_title("min force")


def main():
    side()
    interior()
    plt.show()


if __name__ == "__main__":
    main()
