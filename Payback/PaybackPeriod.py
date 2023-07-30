# -*- coding: utf-8 -*-
"""
Created on 2023.06.06
@author: ChallengeCup2023
"""

import math


def payback(EquipmentTiltAngle, Latitude, Area, LendingRates, LendingPeriod,
            TotalCost, LendingRatio, Allowance, H):
    Pmax = 0.45
    AreaPV = 2.25
    K = 0.78
    Es = 1
    Kd = 0.996
    n = 0
    payback_1 = Area / AreaPV
    payback_2 = (0.707 * math.tan(Latitude/180*math.pi) + 0.4338) / \
                (0.707 - 0.4338 * math.tan(Latitude/180*math.pi))
    payback_3 = (payback_1 * Pmax /
                 (math.cos(EquipmentTiltAngle/180*math.pi) +
                  math.sin(EquipmentTiltAngle/180*math.pi) * payback_2)) / Es
    cost = LendingRates * LendingPeriod * TotalCost * LendingRatio + TotalCost
    output = H * payback_3 * K
    payback_4 = output * math.pow(Kd, n)
    payback_total = Allowance * payback_4
    while (payback_total < cost):
        n += 1
        payback_4 = output * (1-math.pow(Kd, n)) / (1 - Kd)
        payback_total = Allowance * payback_4
        output_n = output * math.pow(0.996, n)
        print('第{}年，本年度发电量为{}度，总发电量为{}度，总收益为{}元'.format(n, output_n, payback_4, payback_total))
    return print('总成本为{}元，需要约{}年可回本'.format(cost, n))

payback(45, 31.0192, 50, 0.05, 10, 72000, 1, 0.85, 1233)