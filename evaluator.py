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
    time_file = open(os.path.join(directory,'runningTime.txt'),'r')
    rt = float(time_file.readline())
    rt = round(rt, 3)
    times[directory] = rt
    time_file.close()
  f = open('runningTimes.json','w')
  json.dump(times,f,indent=4,sort_keys=True)
  f.close()


def getEntropy(directories):
  data = {}
  for directory in directories:
    sequence_count = int(directory.split('_')[7])
    motif_file = open(os.path.join(directory,'motif.txt'),'r')
    predicted_file = open(os.path.join(directory,'outputPWM.txt'),'r')
    motif = readMotif(motif_file)
    motif = np.array(motif) / sequence_count ##all elements in the benchmark motif must be divided by SC because it is not a probability matrix
    predicted = readPredMotif(predicted_file)
    predicted = np.transpose(np.array(predicted))
    entropy = getEntropyHelper(motif, predicted)
    data[directory] = entropy
    motif_file.close()
    predicted_file.close()
  f = open('entropy.json','w')
  json.dump(data,f,indent=4,sort_keys=True)
  f.close()


def getEntropyHelper(motif, predicted):
  pe = []
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

def readPredMotif(f):
  motif = []
  for line in f.readlines():
    read_line = line[1:-2].split(' ')
    updated_line = []
    for i in range(len(read_line)):
      if read_line[i] != '':
        updated_line.append(read_line[i])
    updated_line = [float(i) for i in updated_line]
    motif.append(updated_line)
  f.close()
  return motif


def getOverlappingSites(directories):
  overlapping_sites = {}
  for directory in directories:
    site_file = open(os.path.join(directory,'sites.txt'))
    predicted_file = open(os.path.join(directory,'outputStartingSites.txt'),'r')
    sites = readSites(site_file)
    predicted = readSites(predicted_file)
    motif_length = int(directory.split('_')[3])
    overlaps = getOverlaps(sites,predicted,motif_length)
    overlapping_sites[directory] = overlaps
    site_file.close()
    predicted_file.close()
  f = open('siteOverlap.json','w')
  json.dump(overlapping_sites,f,indent=4,sort_keys=True)
  f.close()


def readSites(f):
  sites = f.readlines()
  f.close()
  return sites


def getOverlaps(sites,predicted,ML):
  overlaps = []
  for i in range(len(sites)):
    o = ML - abs(int(predicted[i]) - int(sites[i]))
    o /= ML
    if o >= (1/ML) and o <= 1:
      overlaps.append(o)
    else:
      overlaps.append(0)
  return overlaps

  
##################################################################
##################################################################

NUCLEOTIDE = ["A", "C", "G", "T"]
directories = os.listdir('../../Downloads/FINAL SUBMISSION 2/')
updated_directories = []
for i in range(len(directories)):
  if (directories[i][:4] == "ICPC"):
    updated_directories.append(directories[i])
getRunningTime(updated_directories)
getEntropy(updated_directories)
getOverlappingSites(updated_directories)