import numpy as np
import matplotlib.pyplot as plt
import json, os


def readJson(entropy, overlap, runtime):
  """
  entropy = 'entropy.json'
  overlap = 'siteOverlap.json'
  runtime = 'runningTime.json'
  """
  entropy_file = open(entropy, 'r')
  entropies = json.load(entropy_file)
  running_file = open(runtime, 'r')
  runningTimes = json.load(running_file)
  overlap_file = open(overlap, 'r')
  overlaps = json.load(overlap_file)
  return [entropies, overlaps, runningTimes]


def ICPCvsEntropy(entropies):
  """
  ICPC vs runtime (Needs ML = 8 and SC = 10)
  ICPC vs entropy (Needs ML = 8 and SC = 10)
  ICPC vs overlap (Needs ML = 8 and SC = 10)
  """
  list1 = []
  list15 = []
  list2 = []
  for k in entropies:
    spl = k.split('_')
    icpc = float(spl[1])
    ml = int(spl[3])
    sc = int(spl[7])
    if ml == 8 and sc == 10:
      if icpc == 1:
        list1.append(sum(entropies[k]))
      elif icpc == 1.5:
        list15.append(sum(entropies[k]))
      elif icpc == 2:
        list2.append(sum(entropies[k]))
  
  list1 = np.array(list1)
  list15 = np.array(list15)
  list2 = np.array(list2)

  combined = np.stack((list1, list15, list2))
  combined = np.transpose(combined)

  plt.xlabel('ICPC')
  plt.ylabel('Entropy')
  plt.title('ICPC vs Entropy')
  
  plt.boxplot(combined, vert=True, patch_artist=True, labels=[1.0, 1.5, 2.0])
  plt.savefig('ICPCentropy.png')
  plt.clf()


def ICPCvsRuntime(runtimes):
  """
  ICPC vs runtime (Needs ML = 8 and SC = 10)
  ICPC vs entropy (Needs ML = 8 and SC = 10)
  ICPC vs overlap (Needs ML = 8 and SC = 10)
  """
  list1 = []
  list15 = []
  list2 = []
  for k in runtimes:
    spl = k.split('_')
    icpc = float(spl[1])
    ml = int(spl[3])
    sc = int(spl[7])
    if ml == 8 and sc == 10:
      if icpc == 1:
        list1.append(runtimes[k])
      elif icpc == 1.5:
        list15.append(runtimes[k])
      elif icpc == 2:
        list2.append(runtimes[k])

  list1 = np.array(list1)
  list15 = np.array(list15)
  list2 = np.array(list2)

  combined = np.stack((list1, list15, list2))
  combined = np.transpose(combined)

  plt.xlabel('ICPC')
  plt.ylabel('Runtime')
  plt.title('ICPC vs Runtime')

  plt.boxplot(combined, vert=True, patch_artist=True, labels=[1.0, 1.5, 2.0])
  plt.savefig('ICPCruntime.png')
  plt.clf()


def ICPCvsOverlaps(overlaps):
  """
  ICPC vs runtime (Needs ML = 8 and SC = 10)
  ICPC vs entropy (Needs ML = 8 and SC = 10)
  ICPC vs overlap (Needs ML = 8 and SC = 10)
  """
  list1 = []
  list15 = []
  list2 = []
  for k in overlaps:
    spl = k.split('_')
    icpc = float(spl[1])
    ml = int(spl[3])
    sc = int(spl[7])
    if ml == 8 and sc == 10:
      if icpc == 1:
        list1.append(sum(overlaps[k]) / sc)
      elif icpc == 1.5:
        list15.append(sum(overlaps[k]) / sc)
      elif icpc == 2:
        list2.append(sum(overlaps[k]) / sc)
      
  list1 = np.array(list1)
  list15 = np.array(list15)
  list2 = np.array(list2)

  combined = np.stack((list1, list15, list2))
  combined = np.transpose(combined)

  plt.xlabel('ICPC')
  plt.ylabel('Overlaps')
  plt.title('ICPC vs Overlaps')

  plt.boxplot(combined, vert=True, patch_artist=True, labels=[1.0, 1.5, 2.0])
  plt.savefig('ICPCoverlap.png')
  plt.clf()


def MLvsRuntime(runtimes):
  """
  Needs ICPC = 2 and SC = 10
  """
  list6 = []
  list7 = []
  list8 = []
  for k in runtimes:
    spl = k.split('_')
    icpc = float(spl[1])
    ml = int(spl[3])
    sc = int(spl[7])
    if icpc == 2 and sc == 10:
      if ml == 6:
        list6.append(runtimes[k])
      elif ml == 7:
        list7.append(runtimes[k])
      elif ml == 8:
        list8.append(runtimes[k])

  list6 = np.array(list6)
  list7 = np.array(list7)
  list8 = np.array(list8)

  combined = np.stack((list6, list7, list8))
  combined = np.transpose(combined)

  plt.xlabel('ML')
  plt.ylabel('Runtime')
  plt.title('ML vs Runtime')
  
  plt.boxplot(combined, vert=True, patch_artist=True, labels=[6, 7, 8])
  plt.savefig('MLruntime.png')
  plt.clf()


def MLvsEntropy(entropies):
  """
  Needs ICPC = 2 and SC = 10
  """
  list6 = []
  list7 = []
  list8 = []
  for k in entropies:
    spl = k.split('_')
    icpc = float(spl[1])
    ml = int(spl[3])
    sc = int(spl[7])
    if icpc == 2 and sc == 10:
      if ml == 6:
        list6.append(sum(entropies[k]))
      elif ml == 7:
        list7.append(sum(entropies[k]))
      elif ml == 8:
        list8.append(sum(entropies[k]))

  list6 = np.array(list6)
  list7 = np.array(list7)
  list8 = np.array(list8)

  combined = np.stack((list6, list7, list8))
  combined = np.transpose(combined)

  plt.xlabel('ML')
  plt.ylabel('Entropy')
  plt.title('ML vs Entropy')

  plt.boxplot(combined, vert=True, patch_artist=True, labels=[6, 7, 8])
  plt.savefig('MLentropy.png')
  plt.clf()


def MLvsOverlaps(overlaps):
  """
  Needs ICPC = 2 and SC = 10
  """
  list6 = []
  list7 = []
  list8 = []
  for k in overlaps:
    spl = k.split('_')
    icpc = float(spl[1])
    ml = int(spl[3])
    sc = int(spl[7])
    if icpc == 2 and sc == 10:
      if ml == 6:
        list6.append(sum(overlaps[k]) / sc)
      elif ml == 7:
        list7.append(sum(overlaps[k]) / sc)
      elif ml == 8:
        list8.append(sum(overlaps[k]) / sc)
      
  list6 = np.array(list6)
  list7 = np.array(list7)
  list8 = np.array(list8)

  combined = np.stack((list6, list7, list8))
  combined = np.transpose(combined)

  plt.xlabel('ML')
  plt.ylabel('Overlaps')
  plt.title('ML vs Overlaps')

  plt.boxplot(combined, vert=True, patch_artist=True, labels=[6, 7, 8])
  plt.savefig('MLoverlap.png')
  plt.clf()


def SCvsRuntime(runtimes):
  """
  Needs ICPC = 2 and ML = 8
  """
  list5 = []
  list10 = []
  list20 = []
  for k in runtimes:
    spl = k.split('_')
    icpc = float(spl[1])
    ml = int(spl[3])
    sc = int(spl[7])
    if icpc == 2 and  ml == 8:
      if sc == 5:
        list5.append(runtimes[k])
      elif sc == 10:
        list10.append(runtimes[k])
      elif sc == 20:
        list20.append(runtimes[k])

  list5 = np.array(list5)
  list10 = np.array(list10)
  list20 = np.array(list20)

  combined = np.stack((list5, list10, list20))
  combined = np.transpose(combined)

  plt.xlabel('SC')
  plt.ylabel('Runtime')
  plt.title('SC vs Runtime')

  plt.boxplot(combined, vert=True, patch_artist=True, labels=[5, 10, 20])
  plt.savefig('SCruntime.png')
  plt.clf()


def SCvsEntropy(entropies):
  """
  Needs ICPC = 2 and ML = 8
  """
  list5 = []
  list10 = []
  list20 = []
  for k in entropies:
    spl = k.split('_')
    icpc = float(spl[1])
    ml = int(spl[3])
    sc = int(spl[7])
    if icpc == 2 and ml == 8:
      if sc == 5:
        list5.append(sum(entropies[k]))
      elif sc == 10:
        list10.append(sum(entropies[k]))
      elif sc == 20:
        list20.append(sum(entropies[k]))

  list5 = np.array(list5)
  list10 = np.array(list10)
  list20 = np.array(list20)

  combined = np.stack((list5, list10, list20))
  combined = np.transpose(combined)

  plt.xlabel('SC')
  plt.ylabel('Entropy')
  plt.title('SC vs Entropy')

  plt.boxplot(combined, vert=True, patch_artist=True, labels=[5, 10, 20])
  plt.savefig('SCentropy.png')
  plt.clf()


def SCvsOverlaps(overlaps):
  """
  Needs ICPC = 2 and mL = 8
  """
  list5 = []
  list10 = []
  list20 = []
  for k in overlaps:
    spl = k.split('_')
    icpc = float(spl[1])
    ml = int(spl[3])
    sc = int(spl[7])
    if icpc == 2 and ml == 8:
      if sc == 5:
        list5.append(sum(overlaps[k]) / sc)
      elif sc == 10:
        list10.append(sum(overlaps[k]) / sc)
      elif sc == 20:
        list20.append(sum(overlaps[k]) / sc)
      
  list5 = np.array(list5)
  list10 = np.array(list10)
  list20 = np.array(list20)

  combined = np.stack((list5, list10, list20))
  combined = np.transpose(combined)

  plt.xlabel('SC')
  plt.ylabel('Overlaps')
  plt.title('SC vs Overlaps')
  
  plt.boxplot(combined, vert=True, patch_artist=True, labels=[5, 10, 20])
  plt.savefig('SCoverlap.png')
  plt.clf()



[entropies, overlaps, runningTimes] = readJson('entropy.json','siteOverlap.json','runningTimes.json')
ICPCvsEntropy(entropies)
ICPCvsOverlaps(overlaps)
ICPCvsRuntime(runningTimes)

MLvsEntropy(entropies)
MLvsOverlaps(overlaps)
MLvsRuntime(runningTimes)

SCvsEntropy(entropies)
SCvsOverlaps(overlaps)
SCvsRuntime(runningTimes)