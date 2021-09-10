import time
from recordtype import recordtype
import src.ExecMode as em

SP_Step = recordtype('StatPerfStep', ['step', 'begin', 'end'])
SP_StepList = []

def SP_BegStep(steplib):
  if em.ExecMode != em.MODE_TEST: return
  step = SP_Step(steplib, time.time(), -1)
  SP_StepList.append(step)
  
def SP_EndStep():
  if em.ExecMode != em.MODE_TEST: return
  global SP_StepList
  cnt = len(SP_StepList)
  if not cnt: return
  SP_StepList[cnt - 1].end = time.time()
  
def SP_DumpStep():
  if em.ExecMode != em.MODE_TEST: return
  global SP_StepList
  cnt = len(SP_StepList)
  if not cnt: return
  start = SP_StepList[0].begin
  idx = 0
  while idx < cnt:
    step = SP_StepList[idx]
    if step.end == -1:
      if idx == cnt - 1: step.end = time.time()
      else: step.end = SP_StepList[idx + 1].begin
    elapstime = step.end - start
    steptime = step.end - step.begin
    print("{:<4d}: {:.3f} {:.3f} {:s}".
            format(idx, elapstime, steptime, step.step))
    idx += 1
  SP_StepList = []

