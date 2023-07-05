# inverse kinematics of the device based on link lengths and end-effector
# position

import numpy as np


def inv_kinematics(a1, a2, a3, a4, a5, x3, y3):
    """a arguments are .. angles?  x arguments are ... ?"""

    P13 = np.sqrt((x3**2) + (y3**2))
    P53 = np.sqrt(((x3 + a5) ** 2) + (y3**2))

    alphaOne = np.arccos(((a1**2) + (P13**2) - (a2**2)) / (2 * a1 * P13))
    betaOne = np.arctan2(y3, - x3)
    thetaOne = np.pi - alphaOne - betaOne

    alphaFive = np.arctan2(y3, x3 + a5)
    betaFive = np.arccos(((P53**2) + (a4**2) - (a3**2)) / (2 * P53 * a4))
    thetaFive = alphaFive + betaFive

    return thetaOne, thetaFive
