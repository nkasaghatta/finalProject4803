import numpy as np
import random, math
import Bio as bp
import os

NUCLEOTIDE = ["A", "C", "G", "T"]

def main():
  ##########################################
  ##########################################
  ICPC = 2
  ML = 8
  SL = 500
  SC = 10

  dir_name = "ICPC_" + str(ICPC) + "_ML_" + str(ML) + "_SL_" + str(SL) + "_SC_" + str(SC)

  if not os.path.exists(dir_name):
    os.makedirs(dir_name)

  sequences = sequenceGenerator(SC, SL)
  motif = motifGenerator(ML, ICPC)
  binding_sites = bindingSiteGenerator(motif, ML, SC)
  planted_sites = plantSite(sequences, binding_sites, SL, ML)

  writeSequences(sequences, dir_name)
  writeSites(planted_sites, dir_name)
  writeMotif(motif, ML, dir_name, SC)
  writeMotifLength(ML, dir_name)


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
  for i in range(len(sequences)):
    binding_ind = random.randrange(len(bindingSites))
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

main()