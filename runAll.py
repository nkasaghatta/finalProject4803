import numpy as np
import random, math, os, json, time
import Bio as bp

ICPC = 2
ML = 8
SL = 500
SC = 10

dir_name = "ICPC_" + str(ICPC) + "_ML_" + str(ML) + "_SL_" + str(SL) + "_SC_" + str(SC)

if not os.path.exists(dir_name):
  os.makedirs(dir_name)


##BENCHMARK
##

NUCLEOTIDE = ["A", "C", "G", "T"]

def sequenceGenHelper(SL):
  """
  This function generates a random sequence of nucleotides.
  Arguments:
    SL: sequence length
  Returns:
    One sequence of length SL
  """
  sequence = ''
  for _ in range(SL):
    sequence += random.choice(NUCLEOTIDE)
  return sequence


def sequenceGenerator(sequenceCount, sequenceLength):
  """
  This function generates a "sequenceCount" number of sequences of length "sequenceLength".
  Arguments:
    sequenceCount: number of sequences to generate
    sequenceLength: length of each sequence
  Returns:
    A list with "sequenceCount" number of sequences.
  """
  SL = sequenceLength
  SC = sequenceCount
  seq_list = ['' for x in range(SC)]
  for i in range(SC):
    seq_list[i] = sequenceGenHelper(SL)
  return seq_list


def motifGenerator(ML, ICPC):
  """
  This function generates a motif (PWM) of length ML and an information content per column of ICPC.
  Arguments:
    ML: motif length
    ICPC: information content per column
  Returns:
    A motif (PWM) of length ML and an information content per column of ICPC.
  """
  pwm = np.zeros((ML, 4))
  if ICPC == 1:
    result = motifGeneratorICPC(pwm, ML, 1)
    return result
  elif ICPC == 1.5:
    result = motifGeneratorICPC(pwm, ML, 1.5)
    return result
  elif ICPC == 2:
    result = motifGeneratorICPC(pwm, ML, 2)
    return result


def motifGeneratorICPC(pwm, ML, ICPC):
  """
  This helper function generates a motif (PWM) of length ML and an information content per column of ICPC.
  Arguments:
    pwm: a motif of dimensions (ML, 4) where 4 stands for the number of nucleotides.
    ML: motif length
    ICPC: information content per column
  """
  nucleotide_dict = {"A": 0, "C": 1, "G": 2, "T": 3}
  if ICPC == 1:
    p = 0.8105
  elif ICPC == 1.5:
    p = 0.9245
  elif ICPC == 2:
    p = 1.0

  for i in range(ML):  #The PWM has ML rows and 4 columns so iteration is across the 'row' variable.
    preferred_nucleotide = random.choice(NUCLEOTIDE)
    for key in nucleotide_dict:
      if key == preferred_nucleotide:
        pwm[i][nucleotide_dict[key]] = p
      else:
        pwm[i][nucleotide_dict[key]] = (1 - p) / 3

  return pwm


def bindingSiteGenerator(motif, ML, SC):
  """
  This function generates SC number of binding sites of length ML each.
  Arguments:
    motif: PWM, a matrix of dimensions (ML, 4) where 4 stands for the number of nucleotides.
    ML: motif length
    SC: sequence count
  Returns:
    A list of SC binding sites of length ML each
  """
  #New bindingsite method, works
  bindingSites = ["" for i in range(SC)]
  for i in range(SC):
    instance = ""
    for j in range(ML):
      chosen_nucleotide = random.choices(NUCLEOTIDE, weights=motif[j, :], k=1)
      instance += ''.join(chosen_nucleotide)
    bindingSites[i] += instance

  return bindingSites


def plantSite(sequences, bindingSites, SL, ML):
  """
  This function randomly selects a binding site from the list of binding sites and places it in a random location in each sequence, rewriting that specific portion of the sequence.
  Arguments:
    sequences: list of sequences
    bindingSites: list of binding sites
    SL: sequence length
    ML: motif length
  Returns:
    A list of indexes where the binding site was placed in each sequence.
  """
  maxLocation = SL - ML
  plantSites = []
  binding_ind = random.randrange(len(bindingSites))
  for i in range(len(sequences)):
    plantStart = random.randint(0, maxLocation)
    seq = sequences[i]
    if plantStart == maxLocation:
      sequences[i] = seq[:plantStart] + bindingSites[binding_ind]
    else:
      sequences[i] = seq[:plantStart] + bindingSites[binding_ind] + seq[plantStart + ML:]
    plantSites.append(plantStart)
  return plantSites


def writeSequences(sequences, name):
  new_name = os.path.join(name,'sequences.fa')
  f = open(new_name, 'w')
  lines = []
  for i in range(len(sequences)):
    f.write('>%d \n' % i)
    f.write(sequences[i] + '\n')
  f.writelines(lines)
  f.close()


def writeSites(sites, name):
  new_name = os.path.join(name,'sites.txt')
  f = open(new_name, 'w')
  for site in sites:
    f.write(str(site) + '\n')
  f.close()


def writeMotif(pwm, ml, name, sc):
  new_name = os.path.join(name,'motif.txt')
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


def writeMotifLength(ml, name):
  new_name = os.path.join(name,'motifLength.txt')
  f = open(new_name, 'w')
  f.write('%s\n' % str(ml))
  f.close()

# def readMotif(name):
#   new_name = os.path.join(name, 'motif.txt')
#   f = open(new_name, 'r')
#   motif = []
#   for line in f.readlines()[1:]:
#     if line[0] != '<':
#       sp = line.split()
#       f_sp = []
#       for s in sp:
#         f_sp.append(float(s))
#       motif.append(f_sp)
#   f.close()
#   return motif

# import Bio as bp
# from numpy.core.multiarray import frombuffer

##GIBBS
##

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


def writePredMotif(pwm, ml, name, sc):
    new_name = os.path.join(name,'predictedmotif.txt')
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


def writePredSites(sites, name):
    new_name = os.path.join(name,'predictedsites.txt')
    f = open(new_name, 'w')
    for site in sites:
        f.write(str(site) + '\n')
    f.close()

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
            if (motifs[j][i] == NUCLEOTIDE[0]):
                PWM[0][i] += 1
            elif (motifs[j][i] == NUCLEOTIDE[1]):
                PWM[1][i] += 1
            elif (motifs[j][i] == NUCLEOTIDE[2]):
                PWM[2][i] += 1
            elif (motifs[j][i] == NUCLEOTIDE[3]):
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
            if motif[i] == NUCLEOTIDE[0]:
                PWM[0][i] += 1
            elif motif[i] == NUCLEOTIDE[1]:
                PWM[1][i] += 1
            elif motif[i] == NUCLEOTIDE[2]:
                PWM[2][i] += 1
            elif motif[i] == NUCLEOTIDE[3]:
                PWM[3][i] += 1
    PWM = PWM / SC  # converts frequency table to probability distributions
    return PWM


# Calculates a score by comparing PWM to subsequence
def scoreCalc(subSeq, PWM):
    Qx = 1
    Px = bgFreq[0] * bgFreq[1] * bgFreq[2] * bgFreq[3]

    for i in range(0, len(subSeq)):
        currColumn = 0
        for j in range(0, 3):
            if subSeq[i] == NUCLEOTIDE[j]:
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
    initialPWM = firstPWM(seqCount, firstMotifs, ML)

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

##EVALUATOR
##

def readBgFreq(f):
  sites = f.readlines()
  f.close()
  return sites

def getRunningTime(directories):
  times = {}
  print("directories:",directories)
  for directory in directories:
    #dirString = os.path.join('data',directory)
    #time_file = open(os.path.join(dirString,'runningTime.txt'),'r')
    time_file = open(os.path.join(directory,'runningTime.txt'),'r')
    rt = float(time_file.readline())
    times[directory] = rt
    time_file.close()
  f = open('runningTimes.json','w')
  json.dump(times,f,indent=4,sort_keys=True)
  f.close()


def getEntropy(directories, SC, ML):
  data = {}
  for directory in directories:
    #dirString = os.path.join('data',directory)
    #motif_file = open(os.path.join(dirString,'motif.txt'),'r')
    #predicted_file = open(os.path.join(dirString,'predictedmotif.txt'),'r')
    #bgFreq_file = open(os.path.join(dirString, 'bgFreq.txt'), 'r')
    motif_file = open(os.path.join(directory,'motif.txt'),'r')
    predicted_file = open(os.path.join(directory,'predictedmotif.txt'),'r')
    bgFreq_file = open(os.path.join(directory, 'bgFreq.txt'), 'r')
    motif = readMotif(motif_file)
    motif = np.array(motif) / SC ##all elements in the benchmark motif must be divided by SC because it is not a probability matrix
    predicted = readMotif(predicted_file)
    print("predicted_before_np:", predicted)
    predicted = np.transpose(np.array(predicted))
    print("predicted_after_np:", predicted)
    #bgFreq = readBgFreq(bgFreq_file)
    entropy = getEntropyHelper(motif, predicted)
    data[directory] = entropy
    motif_file.close()
    predicted_file.close()
    bgFreq_file.close()
  f = open('entropy.json','w')
  json.dump(data,f,indent=4,sort_keys=True)
  f.close()


def getEntropyHelper(motif, predicted):
  ##This has to be changed
  pe = []
  #print("bgFreq[0].split():",bgFreq[0].split())
  #print("bgFreq float:",[float(i) for i in bgFreq[0].split()])
  #float_bgFreq = [float(i) for i in bgFreq[0].split()]
  #print("bgFreq float sum:",float_bgFreq[0]+float_bgFreq[1]+float_bgFreq[2]+float_bgFreq[3])
  #sc = bgFreq[0] + bgFreq[1] + bgFreq[2] + bgFreq[3] #or just bgFreq[0] + bgFreq[0] + bgFreq[0] + bgFreq[0]
  #print("sc:", sc)
  print("motif:", motif)
  print("predicted:", predicted)
  for p in range(len(motif)):
    relative_entropy = 0.0
    for b in range(4):
      if motif[p][b] != 0 and predicted[p][b] != 0:
        relative_entropy += (motif[p][b]) * math.log(((motif[p][b])/(predicted[p][b])), 2)
    relative_entropy = relative_entropy/len(motif)
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
    #dirString = os.path.join('data',directory)
    #site_file = open(os.path.join(dirString,'sites.txt'))
    #predicted_file = open(os.path.join(dirString,'predictedsites.txt'),'r')
    site_file = open(os.path.join(directory,'sites.txt'))
    predicted_file = open(os.path.join(directory,'predictedsites.txt'),'r')
    sites = readSites(site_file)
    predicted = readSites(predicted_file)
    overlaps = getOverlaps(sites,predicted)
    print("overlaps:",overlaps)
    print("directory:",directory)
    sites[directory] = overlaps
    site_file.close()
    predicted_file.close()
  f = open('siteOverlap.json','w')
  json.dump(sites,f,indent=4,sort_keys=True)
  f.close()


def readSites(f):
  sites = f.readlines()
  f.close()
  return sites


def getOverlaps(sites,predicted):
  overlaps = []
  for i in range(len(sites)):
    o = int(predicted[i]) - int(sites[i])
    overlaps.append(o)
  return overlaps

#Benchmark
sequences = sequenceGenerator(SC, SL)
motif = motifGenerator(ML, ICPC)
binding_sites = bindingSiteGenerator(motif, ML, SC)
planted_sites = plantSite(sequences, binding_sites, SL, ML)

writeSequences(sequences, dir_name)
writeSites(planted_sites, dir_name)
writeMotif(motif, ML, dir_name, SC)
writeMotifLength(ML, dir_name)

#Gibbs
#ML = readMotifLength('testmotifLength.txt')
#sequences = readFASTA('testsequences.fa')
#seqCount = len(sequences)
#seqLength = len(sequences[0])
#bgFreq = bgFreqCalc(sequences, seqCount, seqLength)
sequences = readFASTA(os.path.join(dir_name, "sequences.fa"))
bgFreq = bgFreqCalc(sequences, SC, SL)

startTime = time.time()
test1, test2, test3 = gibbs(sequences, SC, SL, ML, 10)

duration = time.time() - startTime
print(test1[:][:])
print(test2[:][:])
print(test3[:])
writePredMotif(test1[:][:], ML, dir_name, SC)
writePredSites(test3[:], dir_name)


if not os.path.exists(dir_name):
    os.makedirs(dir_name)
writeRunTime(duration, dir_name)
writeFreq(bgFreq, dir_name)

#Evaluator
#directories = os.listdir('./data/')
directories = os.listdir('./')
updated_directories = []
for i in range(len(directories)):
    if (directories[i][-4:] != ".git") and (directories[i][-3:] != ".py") and (directories[i][-5:] != ".json") and (directories[i][-5:] != ".idea"):
        updated_directories.append(directories[i])
print(updated_directories[:])
getRunningTime(updated_directories)
getEntropy(updated_directories, SC, ML)
getOverlappingSites(updated_directories)