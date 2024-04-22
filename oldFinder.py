import numpy as np
import random, math
import time, os
# import Bio as bp
# from numpy.core.multiarray import frombuffer
def readFASTA(filename):
    f = open(filename)
    curr = f.readline()
    lines = []
    while True:
        curr = f.readline()
        while True:
            if not curr:
                break
            if curr[0] == '>':
                break
            lines.append(curr.rstrip())
            curr = f.readline()
        if not curr:
            return lines


def readMotifLength(filename):
    f = open(filename)
    curr = int(f.readline())
    return curr


def writeMotif(pwm, ml, name, sc):
    new_name = name + 'predictedmotif.txt'
    f = open(new_name, 'w')
    f.write('>MOTIF1 %d \n' % ml)
    for row in pwm:
        rowText = ""
        for cell in row:
            rowText += str(cell * sc) + " "
        rowText = rowText.strip()
        f.write(rowText + '\n')
    f.write('<\n')
    f.close()


def writeSites(sites, name):
    new_name = name + 'predictedsites.txt'
    f = open(new_name, 'w')
    for site in sites:
        f.write(str(site) + '\n')
    f.close()


nucleotide = ['A', 'C', 'G', 'T']
ML = readMotifLength('testmotifLength.txt')
sequences = readFASTA('testsequences.fa')
seqCount = len(sequences)
seqLength = len(sequences[0])


# returns 4-element list of background frequencies of each nucleotide
def bgFreqCalc(sequences, seqCount, seqLength):
    bgFreq = np.zeros(4)
    for i in range(0, seqCount):
        for j in range(0, seqLength):
            if sequences[i][j] == 'A':
                bgFreq[0] += 1
            elif sequences[i][j] == 'C':
                bgFreq[1] += 1
            elif sequences[i][j] == 'G':
                bgFreq[2] += 1
            elif sequences[i][j] == 'T':
                bgFreq[3] += 1
    bgFreq = bgFreq / (seqLength * seqCount)
    return bgFreq


bgFreq = bgFreqCalc(sequences, seqCount, seqLength)


# returns initial, randomly-selected motifs
def initMotif(sequences, seqCount, seqLength, ML):
    motifs = ['' for i in range(seqCount)]
    potentialStarts = seqLength - ML  # Ensure the last possible start index fits the motif length

    for i in range(seqCount):
        startingPoint = random.randint(0, potentialStarts - 1)  # Fix index range
        motifs[i] = sequences[i][startingPoint:startingPoint + ML]
    return motifs


# returns a tuple of initial random PWM and ICPC.
# To access: initialPWM(x,y,z)[0] = PWM, initialPWM(x,y,z)[1] = ICPC
def firstPWM(seqCount, motifs, ML):
    PWM = np.zeros((4, ML), dtype=float)

    # PWM Calculation:
    for i in range(0, ML):
        for j in range(0, seqCount):
            if (motifs[j][i] == nucleotide[0]):
                PWM[0][i] += 1
            elif (motifs[j][i] == nucleotide[1]):
                PWM[1][i] += 1
            elif (motifs[j][i] == nucleotide[2]):
                PWM[2][i] += 1
            elif (motifs[j][i] == nucleotide[3]):
                PWM[3][i] += 1
    PWM = PWM / seqCount  # converts frequency table to probability distributions

    # calculating ICPC:
    IC = 0
    for i in range(0, ML):
        for j in range(0, len(PWM)):
            if (PWM[j][i] > 0):
                IC += (PWM[j][i] * math.log2(PWM[j][i] / bgFreq[j]))
    ICPC = IC / len(motifs[0])

    return PWM, ICPC


# Returns a new PWM given list of motifs and length of motifs
def motifPWM(motifs, ML):
    PWM = np.zeros((4, ML), dtype=float)
    # PWM Calculation:
    for i in range(ML):  # Loop over each position in the motif length
        for motif in motifs:  # Loop over each motif
            if motif[i] == nucleotide[0]:
                PWM[0][i] += 1
            elif motif[i] == nucleotide[1]:
                PWM[1][i] += 1
            elif motif[i] == nucleotide[2]:
                PWM[2][i] += 1
            elif motif[i] == nucleotide[3]:
                PWM[3][i] += 1
    PWM = PWM / seqCount  # converts frequency table to probability distributions
    return PWM


# Calculates a score by comparing PWM to subsequence
def scoreCalc(subSeq, PWM):
    Qx = 1
    Px = bgFreq[0] * bgFreq[1] * bgFreq[2] * bgFreq[3]

    for i in range(0, len(subSeq)):
        currColumn = 0
        for j in range(0, 3):
            if subSeq[i] == nucleotide[j]:
                currColumn = j
        Qx = Qx * PWM[currColumn][i]

    score = Qx / Px
    return score


# Softmax Function applied to normalize scores into probability distribution
def normScores(scores):
    eTemp = np.exp(scores - np.max(scores))
    probDist = eTemp / eTemp.sum()
    return probDist


# Chooses new motif based on probability distribution of scores
def motifSelection(sequence, ML, probDist):
    motifStarting = np.random.choice(len(probDist), p=probDist)
    return sequence[motifStarting:motifStarting + ML]


# Runs gibbs sampling algorithm and keeps track of best motifs
# Output Guide: [0] = PWM, [1] = motifs, [2] = high score, [3] = best motifs
def gibbs(sequences, seqCount, seqLength, ML, iterations):
    baseDict = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

    # creates first motif and PWM
    firstMotifs = initMotif(sequences, seqCount, seqLength, ML)
    motifs = firstMotifs.copy()
    initialPWM = firstPWM(seqCount, firstMotifs, ML)[0]

    highScore = -float('inf')

    # Array to store the starting indexes of each motif
    startingIndexes = [0] * seqCount

    for run in range(iterations):
        prevPWM = []
        for n in range(seqCount):
            excludedMotifs = motifs[:n] + motifs[n + 1:]
            currentPWM = motifPWM(excludedMotifs, ML)

            # calculate scores + convert to probability distribution
            scores = []
            for i in range((seqLength - ML) + 1):
                subSeq = sequences[n][i:i + ML]
                score = scoreCalc(subSeq, currentPWM)
                scores.append(score)
            probDist = normScores(scores)

            # Choose the next motif based on the calculated probability distribution
            nextStart = np.random.choice(range((seqLength - ML) + 1), p=probDist)
            motifs[n] = sequences[n][nextStart:nextStart + ML]
            startingIndexes[n] = nextStart  # Update the starting index for the current motif

    outputPWM = motifPWM(motifs, ML)
    return [outputPWM, motifs, startingIndexes]

def writeFreq(bgFreq, dir):
    f = open(os.path.join(dir, 'bgFreq.txt'), 'w')
    f.write(str(bgFreq[:]))
    f.close()
def writeRunTime(duration, dir):
    f = open(os.path.join(dir, 'runningTime.txt'), 'w')
    f.write(str(duration) + '\n')
    f.close()
def main():
    startTime = time.time()

    test1, test2, test3 = gibbs(sequences, seqCount, seqLength, ML, 200)

    duration = time.time() - startTime
    print(test1[:][:])
    print(test2[:][:])
    print(test3[:])
    timeDir = "ICPC_" + str(2) + "_ML_" + str(ML) + "_SL_" + str(seqLength) + "_SC_" + str(seqCount)
    if not os.path.exists(timeDir):
        os.makedirs(timeDir)
    writeRunTime(duration, timeDir)
    writeFreq(bgFreq, timeDir)

main()

