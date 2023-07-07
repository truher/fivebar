"""Kinematics of 2-dof 5-bar planar linkage with one grounded bar.

    Adapted from
    http://charm.stanford.edu/ME327/JaredAndSam

    Which is itself adapted from
    "The Pantograph Mk-II: A Haptic Instrument" Hayward, 2005
"""
import numpy as np
from scenario import Scenario


def inverse(scenario: Scenario, x3, y3):
    """Inverse kinematics

    scenario: simulation geometry
    x3,y3: position of end effector, meters
    """

    P13 = np.sqrt((x3**2) + (y3**2))
    P53 = np.sqrt(((x3 + scenario.a5) ** 2) + (y3**2))

    alphaOne = np.arccos(
        ((scenario.a1**2) + (P13**2) - (scenario.a2**2)) / (2 * scenario.a1 * P13)
    )
    betaOne = np.arctan2(y3, -x3)
    thetaOne = np.pi - alphaOne - betaOne

    alphaFive = np.arctan2(y3, x3 + scenario.a5)
    betaFive = np.arccos(
        ((P53**2) + (scenario.a4**2) - (scenario.a3**2)) / (2 * P53 * scenario.a4)
    )
    thetaFive = alphaFive + betaFive

    return thetaOne, thetaFive


def forward(scenario: Scenario, t1, t5, ax, plot_linkage):
    """Forward kinematics

    scenario: geometry
    t1: angle between a1 and a5
    t5: angle between a4 and a5
    ax: matplotlib axes to plot on
    plot_linkage (boolean): do the plotting

    returns the jacobian.  todo make that a separate function
    """

    # by definition
    x1 = scenario.x1
    y1 = scenario.y1

    x2 = scenario.a1 * np.cos(t1)
    y2 = scenario.a1 * np.sin(t1)

    x4 = scenario.a4 * np.cos(t5) - scenario.a5
    y4 = scenario.a4 * np.sin(t5)

    x5 = -scenario.a5
    y5 = 0

    P2 = np.array([[x2], [y2]])
    P4 = np.array([[x4], [y4]])

    P2Ph = (scenario.a2**2 - scenario.a3**2 + np.linalg.norm(P4 - P2) ** 2) / (
        2 * np.linalg.norm(P4 - P2)
    )
    Ph = P2 + (P2Ph / np.linalg.norm(P2 - P4)) * (P4 - P2)
    P3Ph = np.sqrt(scenario.a2**2 - P2Ph**2)

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

    # NOTE: THE AUTHOR FORGOT NEGATIVE SIGN IN THE PAPER
    del1_x2 = -scenario.a1 * np.sin(t1)
    del1_y2 = scenario.a1 * np.cos(t1)
    # NOTE: THE AUTHOR FORGOT NEGATIVE SIGN IN THE PAPER
    del5_x4 = -scenario.a4 * np.sin(t5)
    del5_y4 = scenario.a4 * np.cos(t5)

    del1_y4 = 0
    del1_x4 = 0
    del5_y2 = 0
    del5_x2 = 0

    # joint 1
    del1_d = ((x4 - x2) * (del1_x4 - del1_x2) + (y4 - y2) * (del1_y4 - del1_y2)) / d
    del1_b = del1_d - (del1_d * (scenario.a2**2 - scenario.a3**2 + d**2)) / (
        2 * d**2
    )
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
    del5_b = del5_d - (del5_d * (scenario.a2**2 - scenario.a3**2 + d**2)) / (
        2 * d**2
    )
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
