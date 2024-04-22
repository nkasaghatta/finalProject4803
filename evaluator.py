import os, math, json
import numpy as np
import matplotlib.pyplot as plt


def readBgFreq(f):
    sites = f.readlines()
    f.close()
    return sites


def getRunningTime(directories):
    times = {}
    for directory in directories:
        dirString = os.path.join('data', directory)
        time_file = open(os.path.join(dirString, 'runningTime.txt'), 'r')
        rt = float(time_file.readline())
        times[directory] = rt
        time_file.close()
    f = open('runningTimes.json', 'w')
    json.dump(times, f, indent=4, sort_keys=True)
    f.close()


def getEntropy(directories, SC):
    data = {}
    for directory in directories:
        dirString = os.path.join('data', directory)
        motif_file = open(os.path.join(dirString, 'motif.txt'), 'r')
        predicted_file = open(os.path.join(dirString, 'predictedmotif.txt'), 'r')
        bgFreq_file = open(os.path.join(dirString, 'bgFreq.txt'), 'r')
        motif = readMotif(motif_file)
        motif = np.array(
            motif) / SC  ##all elements in the benchmark motif must be divided by SC because it is not a probability matrix
        predicted = readMotif(predicted_file)
        bgFreq = readBgFreq(bgFreq_file)
        entropy = getEntropyHelper(motif, predicted, bgFreq)
        data[directory] = entropy
        motif_file.close()
        predicted_file.close()
        bgFreq_file.close()
    f = open('entropy.json', 'w')
    json.dump(data, f, indent=4, sort_keys=True)
    f.close()


def getEntropyHelper(motif, predicted, bgFreq):
    ##This has to be changed
    pe = []
    sc = bgFreq[0][0] + bgFreq[0][1] + bgFreq[0][2] + bgFreq[0][
        3]  # or just bgFreq[0] + bgFreq[0] + bgFreq[0] + bgFreq[0]
    for p in range(len(motif)):
        relative_entropy = 0.0
        for b in range(4):
            if motif[p][b] != 0 and predicted[p][b] != 0:
                relative_entropy += (motif[p][b] / sc) * math.log(((motif[p][b] / sc) / (predicted[p][b] / sc)), 2)
        relative_entropy = relative_entropy / len(motif)
        pe.append(relative_entropy)
    return pe


def readMotif(f):
    motif = []
    for line in f.readlines()[1:]:
        if line[0] != '<':
            sp = line.split()
            f_sp = []
            for s in sp:
                f_sp.append(float(s))
            motif.append(f_sp)
    f.close()
    return motif



def getOverlappingSites(directories):
    sites = {}
    for directory in directories:
        dirString = os.path.join('data', directory)
        site_file = open(os.path.join(dirString, 'sites.txt'))
        predicted_file = open(os.path.join(dirString, 'predictedsites.txt'), 'r')
        sites = readSites(site_file)
        predicted = readSites(predicted_file)
        overlaps = getOverlaps(sites, predicted)
        sites[directory] = overlaps
        site_file.close()
        predicted_file.close()
    f = open('siteOverlap.json', 'w')
    json.dump(sites, f, indent=4, sort_keys=True)
    f.close()


def readSites(f):
    sites = f.readlines()
    f.close()
    return sites


def getOverlaps(sites, predicted):
    overlaps = []
    for i in range(len(sites)):
        o = int(predicted[i]) - int(sites[i])
        overlaps.append(o)
    return overlaps


def getICPC(f):
    scores = []
    for line in f.readlines()[1:]:
        if line[0] != '<':
            currScore = line.split()
            scores.append(currScore)
    f.close()
    return scores


def createLinePlot(directories):
    for directory in directories:
        dirString = os.path.join('data', directory)
        icpc_file = open(os.path.join(dirString, 'ICPC_2_ML_6_SL_500_SC_10/icpcdata.txt'), 'r')
        y = getICPC(icpc_file)
        x = []
        x = [i for i in range(len(y))]

    plt.plot(x, y)
    plt.xlabel("Number of Iterations")  # add X-axis label
    plt.ylabel("Maximum ICPC Score Found")  # add Y-axis label
    plt.title("Maximum ICPC Score vs Number of Iterations")  # add title
    plt.show()


if __name__ == '__main__':
    SC = 10
    directories = os.listdir('./')
    getRunningTime(directories)
    getEntropy(directories, SC)
    getOverlappingSites(directories)
    createLinePlot(directories)