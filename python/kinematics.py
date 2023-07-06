import numpy as np


def inverse(a1, a2, a3, a4, a5, x3, y3):
    """Inverse kinematics of 2-DOF 5-bar linkage

    a1-a5: link lengths, meters
    x3,y3: position of end effector, meters
    """

    P13 = np.sqrt((x3**2) + (y3**2))
    P53 = np.sqrt(((x3 + a5) ** 2) + (y3**2))

    alphaOne = np.arccos(((a1**2) + (P13**2) - (a2**2)) / (2 * a1 * P13))
    betaOne = np.arctan2(y3, -x3)
    thetaOne = np.pi - alphaOne - betaOne

    alphaFive = np.arctan2(y3, x3 + a5)
    betaFive = np.arccos(((P53**2) + (a4**2) - (a3**2)) / (2 * P53 * a4))
    thetaFive = alphaFive + betaFive

    return thetaOne, thetaFive


def forward(a1, a2, a3, a4, a5, t1, t5, ax, plot_linkage):
    """Forward kinematics of 2-DOF 5-bar planar linkage

    Following the work of: "The Pantograph Mk-II: A Haptic Instrument"
    Hayward, 2005 (note two negative signs in the Jacobian are fixed here)

    a1-a5: link lengths, meters
    t1: ?
    ax: matplotlib axes to plot on
    plot_linkage (boolean): do the plotting
    """

    # by definition
    x1 = 0
    y1 = 0

    x2 = a1 * np.cos(t1)
    y2 = a1 * np.sin(t1)

    x4 = a4 * np.cos(t5) - a5
    y4 = a4 * np.sin(t5)

    x5 = -a5
    y5 = 0

    P2 = np.array([[x2], [y2]])
    P4 = np.array([[x4], [y4]])

    P2Ph = (a2**2 - a3**2 + np.linalg.norm(P4 - P2) ** 2) / (
        2 * np.linalg.norm(P4 - P2)
    )
    Ph = P2 + (P2Ph / np.linalg.norm(P2 - P4)) * (P4 - P2)
    P3Ph = np.sqrt(a2**2 - P2Ph**2)

    x3 = Ph[0].item() + (P3Ph / np.linalg.norm(P2 - P4)) * (y4 - y2)
    y3 = Ph[1].item() - (P3Ph / np.linalg.norm(P2 - P4)) * (x4 - x2)

    P3 = np.array([[x3], [y3]])

    if plot_linkage:
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

    # Jacobian
    d = np.linalg.norm(P2 - P4)
    b = np.linalg.norm(P2 - Ph)
    h = np.linalg.norm(P3 - Ph)

    del1_x2 = -a1 * np.sin(t1)  # NOTE: THE AUTHOR FORGOT NEGATIVE SIGN IN THE PAPER
    del1_y2 = a1 * np.cos(t1)
    del5_x4 = -a4 * np.sin(t5)  # NOTE: THE AUTHOR FORGOT NEGATIVE SIGN IN THE PAPER
    del5_y4 = a4 * np.cos(t5)

    del1_y4 = 0
    del1_x4 = 0
    del5_y2 = 0
    del5_x2 = 0

    # joint 1
    del1_d = ((x4 - x2) * (del1_x4 - del1_x2) + (y4 - y2) * (del1_y4 - del1_y2)) / d
    del1_b = del1_d - (del1_d * (a2**2 - a3**2 + d**2)) / (2 * d**2)
    del1_h = -b * del1_b / h

    del1_yh = (
        del1_y2
        + (del1_b * d - del1_d * b) / d**2 * (y4 - y2)
        + b / d * (del1_y4 - del1_y2)
    )
    del1_xh = (
        del1_x2
        + (del1_b * d - del1_d * b) / d**2 * (x4 - x2)
        + b / d * (del1_x4 - del1_x2)
    )

    del1_y3 = (
        del1_yh
        - h / d * (del1_x4 - del1_x2)
        - (del1_h * d - del1_d * h) / d**2 * (x4 - x2)
    )
    del1_x3 = (
        del1_xh
        + h / d * (del1_y4 - del1_y2)
        + (del1_h * d - del1_d * h) / d**2 * (y4 - y2)
    )

    # joint 2
    del5_d = ((x4 - x2) * (del5_x4 - del5_x2) + (y4 - y2) * (del5_y4 - del5_y2)) / d
    del5_b = del5_d - (del5_d * (a2**2 - a3**2 + d**2)) / (2 * d**2)
    del5_h = -b * del5_b / h

    del5_yh = (
        del5_y2
        + (del5_b * d - del5_d * b) / d**2 * (y4 - y2)
        + b / d * (del5_y4 - del5_y2)
    )
    del5_xh = (
        del5_x2
        + (del5_b * d - del5_d * b) / d**2 * (x4 - x2)
        + b / d * (del5_x4 - del5_x2)
    )

    del5_y3 = (
        del5_yh
        - h / d * (del5_x4 - del5_x2)
        - (del5_h * d - del5_d * h) / d**2 * (x4 - x2)
    )
    del5_x3 = (
        del5_xh
        + h / d * (del5_y4 - del5_y2)
        + (del5_h * d - del5_d * h) / d**2 * (y4 - y2)
    )

    return np.array([[del1_x3, del5_x3], [del1_y3, del5_y3]])
