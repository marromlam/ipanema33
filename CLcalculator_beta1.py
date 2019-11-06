from bisect import *
from toygen import generateBinned, poissonLL_b as poissonLL
from timeit import default_timer as timer

def TS(sample, null, test):
    chi2b = -2*poissonLL(sample, null)
    chi2sb = -2*poissonLL(sample, test)
    return chi2b - chi2sb

def catTS(samples, nulls, tests):
    ts = 0
    for name in samples.keys():
        ts += TS(samples[name], nulls[name], tests[name])
    return ts

def getCLs(ts0,tS,tB, msg = 1):
    tS.sort(), tB.sort()
    
    clsb = bisect(tS,ts0)*1./len(tS)
    clb = bisect(tB,ts0)*1./len(tB)
    if clsb: cls = clsb/clb
    else: cls = 0.
    if msg:
        print "CLb:",clb
        print "CLsb:", clsb
        print "CLs", cls
    return cls, clb, clsb

class CLcalculatorBasic:
    def __init__(self): pass

    def setModel(self, model): self.model = model
    def setExpected(self, f): self.Expected = f
    def createNull(self): self.null = self.Expected(0)

    def setData(self, dat):
        self.dataset = dat
        self.tmptoy = dat.copy()
        
    def CLs(self, Ns, toys = 10000):
        if not "null" in dir(self): self.createNull()
        dat = self.dataset
        tmptoy = self.tmptoy
        
        tB, tS = [],[]
        null = self.null
        test = self.Expected(Ns)
       
        ts0 = TS(dat, null, test)
        for i in range(toys):
            #start = timer()
            generateBinned(tmptoy, null)
            #print "GEN:", timer() - start
            #start = timer()
            tB.append(TS(tmptoy, null, test))
            #print "TS:", timer()-start
            generateBinned(tmptoy, test)
            tS.append(TS(tmptoy, null, test))
            
        tB.sort(), tS.sort()
        clsb = bisect(tS,ts0)*1./len(tS)
        clb = bisect(tB,ts0)*1./len(tB)
        print "CLb:",clb
        print "CLsb:", clsb
        print "CLs", clsb/clb
        return tS,tB,ts0

#class CLcalculator:
    
