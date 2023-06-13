# -*- coding: utf-8 -*-
"""
Created on 2023.06.06
@author: ChallengeCup2023
"""

import math


def payback(EquipmentTiltAngle, Latitude, Area, LendingRates, LendingPeriod, TotalCost, LendingRatio, Allowance, H):
    Pmax = 0.45
    AreaPV = 2.25
    K = 0.78
    Es = 1000
    Kd = 0.996
    n = 0
    payback_1 = Area/AreaPV
    payback_2 = (0.707 * math.tan(Latitude) + 0.4338) / (0.707 - 0.4338 * math.tan(Latitude))
    payback_3 = (payback_1 * Pmax / (math.cos(EquipmentTiltAngle) + math.sin(EquipmentTiltAngle) * payback_2)) / Es
    cost = LendingRates * LendingPeriod * TotalCost * LendingRatio + TotalCost
    payback_4 = Allowance * H * payback_3 * K * math.pow(Kd, n)
    while (payback_4 < cost):
        n += 1
        payback_4 += Allowance * H * payback_3 * K * math.pow(Kd, n)
    return n

demo = payback(45, 31.0192, 50, 0.05, 10, 72000, 1, 0.85, )