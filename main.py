import numpy as np
import random, math


def main():
    SL = 15
    SC = 5
    sequenceGenerator(SC, SL)
    print(sequenceGenerator(SC, SL));


def sequenceGenHelper(length):
    nucleotide = ['A', 'C', 'G', 'T']
    sequence = ''
    for i in range(length):
        sequence += random.choice(nucleotide)
    return sequence


def sequenceGenerator(sequenceCount, sequenceLength):
    SL = sequenceLength
    SC = sequenceCount
    list = [0 for j in range(SC)]
    for i in range(SC):
        list[i] = sequenceGenHelper(SL)


def motifGenerator(ml, icpc, sc):
    pwm = np.zeros((ml, 4))
    if icpc == 1:
        for r in range(ml):
            motifGeneratorIC(pwm, r, sc, 1)
        return pwm
    elif icpc == 1.5:
        for r in range(ml):
            rand_num = random.randint(0, 1)
            if rand_num == 0:
                motifGeneratorIC(pwm, r, sc, 1)
            else:
                motifGeneratorIC(pwm, r, sc, 2)
        return pwm
    elif icpc == 2:
        for r in range(ml):
            motifGeneratorIC(pwm, r, sc, 2)
        return pwm


def motifGeneratorIC(pwm, row, sc, col):
    if col == 1:
        cols = [0, 1, 2, 3]
        c1 = random.choice(cols)
        cols.remove(c1)
        c2 = random.choice(cols)
        pwm[row, c1] = sc / 2
        pwm[row, c2] = sc - pwm[row, c1]
    if col == 2:
        colNum = random.randint(0, 3)
        pwm[row, colNum] = sc
