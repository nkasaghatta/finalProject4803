import numpy as np
import random, math
import time, os


# import Bio as bp
# from numpy.core.multiarray import frombuffer
def readFASTA(filename):
    with open(filename) as f:
        lines = []
        for line in f:
            if line.startswith('>'):
                continue
            lines.append(line.rstrip())
    return lines


def readMotifLength(filename):
    with open(filename) as f:
        return int(f.readline().strip())


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
ML = readMotifLength('motifLength.txt')
sequences = readFASTA('sequences.fa')
seqCount = len(sequences)
seqLength = len(sequences[0])


# returns 4-element list of background frequencies of each nucleotide
def bgFreqCalc(sequences):
    count = np.zeros(4)
    nucleotide_index = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

    for seq in sequences:
        for nucleotide in seq:
            count[nucleotide_index[nucleotide]] += 1

    total_nucleotides = len(sequences) * len(sequences[0])
    return count / total_nucleotides


bgFreq = bgFreqCalc(sequences)


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
    nucleotideIndex = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

    motif_indices = np.array([[nucleotideIndex[nuc] for nuc in motif]
                              for motif in motifs])
    PWM = np.zeros((4, ML))
    for nuc in range(4):
        PWM[nuc, :] = np.sum(motif_indices == nuc, axis=0)
    PWM /= len(motifs)
    return PWM


# Calculates a score by comparing PWM to subsequence
Px = np.prod(bgFreq)


def scoreCalc(subSeq, PWM, Px):
    Qx = 1
    for i in range(0, len(subSeq)):
        currColumn = 0
        for j in range(0, 4):
            if subSeq[i] == nucleotide[j]:
                currColumn = j
        Qx = Qx * PWM[currColumn][i]

    score = Qx / Px
    if score == 0.0:
        score = 1e-10
    return score


# Softmax Function applied to normalize scores into probability distribution
def normScores(scores):
    scoreArr = np.array(scores)
    probDist = scoreArr / (scoreArr.sum())
    return probDist


# Chooses new motif based on probability distribution of scores
def motifSelection(sequence, ML, probDist):
    motifStarting = np.random.choice(len(probDist), p=probDist)
    return sequence[motifStarting:motifStarting + ML]


# ICPC calculator given
def calcICPC(PWM, bgFreq):
    IC = 0
    q_b = len(PWM[0])
    for i in range(0, len(PWM[0])):
        for j in range(0, 4):
            value = PWM[j][i]
            if value > 0:
                IC += PWM[j][i] * math.log2(PWM[j][i] / bgFreq[j])
            else:
                IC += 0
    ICPC = IC / q_b
    return ICPC


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
    prevPWM = []
    prevStartingIndexes = []
    ICPCdata = [0.0 for i in range(iterations)]
    for run in range(0, iterations):
        for n in range(seqCount):
            excludedMotifs = motifs[:n] + motifs[n + 1:]
            currentPWM = motifPWM(excludedMotifs, ML)

            # calculate scores + convert to probability distribution
            scores = []
            for i in range((seqLength - ML) + 1):
                subSeq = sequences[n][i:i + ML]
                score = scoreCalc(subSeq, currentPWM, Px)
                scores.append(score)
            probDist = normScores(scores)
            # print("RUN:  " + str(run) + "  N: " + str(n) + "  SCORES:   " + str(probDist[:]))

            # Choose the next motif based on the calculated probability distribution
            nextStart = np.random.choice(range((seqLength - ML) + 1), p=probDist)
            motifs[n] = sequences[n][nextStart:nextStart + ML]
            startingIndexes[n] = nextStart  # Update the starting index for the current motif

        if len(prevPWM) == 0:
            outputPWM = motifPWM(motifs, ML)
            prevPWM = outputPWM
            prevStartingIndexes = startingIndexes
        elif calcICPC(prevPWM, bgFreq) > calcICPC(motifPWM(motifs, ML), bgFreq):
            outputPWM = prevPWM
            startingIndexes = prevStartingIndexes
        elif calcICPC(prevPWM, bgFreq) < calcICPC(motifPWM(motifs, ML), bgFreq):
            prevPWM = motifPWM(motifs, ML)
            prevStartingIndexes = startingIndexes
        ICPCdata[run] = calcICPC(prevPWM, bgFreq)
        print("RUN: " + str(run) + "       ICPC: " + str(ICPCdata[run]))
    outputPWM = motifPWM(motifs, ML)
    return [outputPWM, startingIndexes, ICPCdata]


def writeFreq(bgFreq, dir):
    f = open(os.path.join(dir, 'bgFreq.txt'), 'w')
    f.write(str(bgFreq[:]))
    f.close()


def writeRunTime(duration, dir):
    f = open(os.path.join(dir, 'runningTime.txt'), 'w')
    f.write(str(duration) + '\n')
    f.close()


def writeICPCData(data, dir):
    f = open(os.path.join(dir, 'icpcdata.txt'), 'w')
    for i in range(0, len(data)):
        f.write(str(data[i]) + '\n')
    f.close()


def writeOutPWM(data, dir):
    f = open(os.path.join(dir, 'outputPWM.txt'), 'w')
    for i in range(0, 4):
        f.write(str(data[i][:]) + '\n')
    f.close()


def writeStartingSites(data, dir):
    f = open(os.path.join(dir, 'outputStartingSites.txt'), 'w')
    for i in range(0, len(data)):
        f.write(str(data[i]) + '\n')
    f.close()


def main():
    startTime = time.time()

    test1, test2, test3 = gibbs(sequences, seqCount, seqLength, ML, 20000)

    duration = time.time() - startTime
    print(test1[:][:])
    print(test2[:])
    timeDir = "ICPC_" + str(2) + "_ML_" + str(ML) + "_SL_" + str(
        seqLength) + "_SC_" + str(seqCount)
    if not os.path.exists(timeDir):
        os.makedirs(timeDir)
    writeRunTime(duration, timeDir)
    writeICPCData(test3, timeDir)
    writeFreq(bgFreq, timeDir)
    writeOutPWM(test1, timeDir)
    writeStartingSites(test2, timeDir)


main()
