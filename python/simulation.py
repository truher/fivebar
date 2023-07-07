import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import kinematics
from scenario import Scenario


def side(scenario: Scenario):
    """Simulates the device moving around the edges of the workspace.

    Verifies the end-effector can reach the whole envelope
    (inverse kinematics will fail otherwise).
    Plots the device at the corners and midpoints.
    """

    number = 20
    xpoints2 = np.linspace(
        scenario.xcenter + scenario.w / 2, scenario.xcenter + scenario.w / 2, number
    )
    xpoints3 = np.linspace(
        scenario.xcenter + scenario.w / 2, scenario.xcenter - scenario.w / 2, number
    )
    xpoints4 = np.linspace(
        scenario.xcenter - scenario.w / 2, scenario.xcenter - scenario.w / 2, number
    )
    xpoints5 = np.linspace(
        scenario.xcenter - scenario.w / 2, scenario.xcenter + scenario.w / 2, number
    )

    xpoints = np.concatenate((xpoints2, xpoints3, xpoints4, xpoints5))

    ypoints2 = np.linspace(
        scenario.ycenter + scenario.h / 2, scenario.ycenter - scenario.h / 2, number
    )
    ypoints3 = np.linspace(
        scenario.ycenter - scenario.h / 2, scenario.ycenter - scenario.h / 2, number
    )
    ypoints4 = np.linspace(
        scenario.ycenter - scenario.h / 2, scenario.ycenter + scenario.h / 2, number
    )
    ypoints5 = np.linspace(
        scenario.ycenter + scenario.h / 2, scenario.ycenter + scenario.h / 2, number
    )

    ypoints = np.concatenate((ypoints2, ypoints3, ypoints4, ypoints5))

    # if you can reach the boundary, you can reach the interior
    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    for i in range(len(xpoints)):
        x3 = xpoints[i].item()
        y3 = ypoints[i].item()
        plot_linkage(scenario, ax, x3, y3)
        # kinematics.inverse(scenario, x3, y3)

    # show the key positions
    fig, axs = plt.subplots(3, 3, figsize=(8, 8))
    plot_linkage(scenario, axs[0, 0], min(xpoints), max(ypoints))
    plot_linkage(scenario, axs[0, 1], scenario.xcenter, max(ypoints))
    plot_linkage(scenario, axs[0, 2], max(xpoints), max(ypoints))
    plot_linkage(scenario, axs[1, 0], min(xpoints), scenario.ycenter)
    plot_linkage(scenario, axs[1, 1], scenario.xcenter, scenario.ycenter)
    plot_linkage(scenario, axs[1, 2], max(xpoints), scenario.ycenter)
    plot_linkage(scenario, axs[2, 0], min(xpoints), min(ypoints))
    plot_linkage(scenario, axs[2, 1], scenario.xcenter, min(ypoints))
    plot_linkage(scenario, axs[2, 2], max(xpoints), min(ypoints))


def plot_linkage(scenario, ax, x3, y3):
    """Plots the linkage at the specified end-effector position.

    TODO: fix the reversal here
    """
    t1, t5 = kinematics.inverse(scenario, x3, y3)
    P1, P2, P3, P4, P5, Ph = kinematics.forward(scenario, t1, t5)
    x1 = P1[0].item()
    y1 = P1[1].item()
    x2 = P2[0].item()
    y2 = P2[1].item()
    x3 = P3[0].item()
    y3 = P3[1].item()
    x4 = P4[0].item()
    y4 = P4[1].item()
    x5 = P5[0].item()
    y5 = P5[1].item()
    ax.axis("equal")
    ax.set_xlim(scenario.xmin, scenario.xmax)
    ax.set_ylim(scenario.ymin, scenario.ymax)
    ax.grid()
    ax.add_patch(
        Rectangle(
            (
                -(scenario.xcenter + scenario.w / 2),
                -(scenario.ycenter + scenario.h / 2),
            ),
            scenario.w,
            scenario.h,
            fill=False,
        )
    )
    ax.plot(-x1, -y1, marker="o")
    ax.plot(-x2, -y2, marker="o")
    ax.plot(-x3, -y3, marker="o")
    ax.plot(-x4, -y4, marker="o")
    ax.plot(-x5, -y5, marker="o")
    ax.plot([-x1, -x2], [-x1, -y2])
    ax.plot([-x2, -x3], [-y2, -y3])
    ax.plot([-x3, -x4], [-y3, -y4])
    ax.plot([-x4, -x5], [-y4, -y5])
    ax.plot([-x5, -x1], [-y5, -y1])


def interior(scenario):
    """Plots the jacobian and minimum-maximum forces over the workspace."""

    # force angles
    angle_number = 20
    angle_range = 2 * np.pi / angle_number

    # define how many points to solve for
    number = 10
    xpoints = np.linspace(
        scenario.xcenter + scenario.w / 2, scenario.xcenter - scenario.w / 2, number
    )
    ypoints = np.linspace(
        scenario.ycenter + scenario.h / 2, scenario.ycenter - scenario.h / 2, number
    )

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

            theta = kinematics.inverse(scenario, x3, y3)
            t1 = theta[0].item()
            t5 = theta[1].item()

            # jacobian
            P1, P2, P3, P4, P5, Ph = kinematics.forward(scenario, t1, t5)
            J = kinematics.jacobian(scenario, t1, t5, P1, P2, P3, P4, P5, Ph)

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
                        abs(torque[0].item()) >= (scenario.Tmax) * scenario.ratio
                        or abs(torque[1].item()) >= (scenario.Tmax) * scenario.ratio
                    ):
                        # then we have reached the max force in that direction
                        max_force_reached = True
                    else:
                        # otherwise increment the magnitude
                        mag = mag + 0.01

                max_dir_force[k] = mag

            mean_force[i, j] = np.mean(max_dir_force)
            min_force[i, j] = np.min(max_dir_force)

    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    ax.axis("equal")
    ax.set_xlim(scenario.xmin, scenario.xmax)
    ax.set_ylim(scenario.ymin, scenario.ymax)
    ax.grid()

    ax.add_patch(
        Rectangle(
            (
                -(scenario.xcenter + scenario.w / 2),
                -(scenario.ycenter + scenario.h / 2),
            ),
            scenario.w,
            scenario.h,
            fill=False,
        )
    )

    CS = ax.contourf(-xpoints, -ypoints, np.transpose(condition), cmap="summer")
    CS = ax.contour(-xpoints, -ypoints, np.transpose(condition), colors="k")
    ax.clabel(CS)

    ax.set_title("condition")

    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    ax.axis("equal")
    ax.set_xlim(scenario.xmin, scenario.xmax)
    ax.set_ylim(scenario.ymin, scenario.ymax)
    ax.grid()

    ax.add_patch(
        Rectangle(
            (
                -(scenario.xcenter + scenario.w / 2),
                -(scenario.ycenter + scenario.h / 2),
            ),
            scenario.w,
            scenario.h,
            fill=False,
        )
    )
    # ax.autoscale_view()

    CS = ax.contourf(-xpoints, -ypoints, np.transpose(min_force), cmap="summer")
    CS = ax.contour(-xpoints, -ypoints, np.transpose(min_force), colors="k")
    ax.clabel(CS)

    ax.set_title("min force (N)")


def main():
    scenario = Scenario(
        name="bigger",
        a1=0.15,
        a2=0.2,
        a3=0.2,
        a4=0.15,
        a5=0.05,
        x1=0,
        y1=0,
        ratio=1,
        Tmax=0.38,  # dual motors
        w=0.30,
        h=0.15,
        xcenter=-0.025,
        ycenter=0.2,
        xmin=-0.2,
        xmax=0.25,
        ymin=-0.30,
        ymax=0.05,
    )

    small_scenario = Scenario(
        name="small",
        a1=0.065,
        a2=0.1,
        a3=0.1,
        a4=0.065,
        a5=0.05,
        x1=0,
        y1=0,
        ratio=1,
        Tmax=0.19,
        w=0.15,
        h=0.075,
        xcenter=-0.025,
        ycenter=0.08,
        xmin=-0.1,
        xmax=0.15,
        ymin=-0.3,
        ymax=0.05,
    )

    # TODO theres some wierd reversing going on here
    # original_scenario = Scenario(
    #    name="original",
    #    a1=0.25,
    #    a2=0.25,
    #    a3=0.25,
    #    a4=0.25,
    #    a5=0.1,
    #    x1=0,
    #    y1=0,
    #    # capstan mechanial advantage ratio
    #    # ratio = 0.07303 / (0.0131 / 2)
    #    ratio=1,
    #    # stall torue Nm
    #    Tmax=0.129,
    #    w=0.2794,
    #    h=0.2159,
    #    xcenter=-0.05,
    #    ycenter=0.34,
    # )
    side(scenario)
    interior(scenario)
    plt.show()


if __name__ == "__main__":
    main()
