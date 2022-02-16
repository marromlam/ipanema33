import numpy as np
dataonly = 1

__all__ = ['RooUnblindUniform']

class RooBlindTools(object):
    def __init__(self, stSeedIn:str, Mode, centralValue:float, sigmaOffset:float, s2bMode:bool=False):
        self._PrecisionOffsetScale = sigmaOffset
        self._PrecisionCentralValue = centralValue
        self._mode = Mode
        self._s2bMode = s2bMode
        self.setup(stSeedIn)


    def setup(self, stSeedIn:str):
      self._stSeed = stSeedIn;
      self._DeltaZScale = 1.56;
      self._DeltaZOffset = self._DeltaZScale*self.MakeOffset("abcdefghijklmnopqrstuvwxyz")
      self._DeltaZSignFlip = self.MakeSignFlip("ijklmnopqrstuvwxyzabcdefgh")
      self._AsymOffset = self.MakeGaussianOffset("opqrstuvwxyzabcdefghijklmn")
      self._AsymSignFlip = self.MakeSignFlip("zyxwvutsrqponmlkjihgfedcba")
      self._DeltaMScale = 0.1
      self._DeltaMOffset = self._DeltaMScale*self.MakeOffset("opqrstuvwxyzabcdefghijklmn")
      self._MysteryPhase = 3.14159 * self.MakeOffset("wxyzabcdefghijklmnopqrstuv")
      if (self._s2bMode) :
        self._PrecisionSignFlip = self.MakeSignFlip("zyxwvutsrqponmlkjihgfedcba")
      else:
        self._PrecisionSignFlip = self.MakeSignFlip("klmnopqrstuvwxyzabcdefghij")
      self._PrecisionOffset = self._PrecisionOffsetScale*self.MakeGaussianOffset("opqrstuvwxyzabcdefghijklmn")
      self._PrecisionUniform = self._PrecisionOffsetScale*self.MakeOffset("jihgfedcbazyxwvutsrqponmlk")
      self._STagConstant = self.Randomizer("fghijklmnopqrstuvwxyzabcde")


    def HideDeltaZ(self, DeltaZ:float, STag:float):
      sTag = int(SignOfTag(STag))
      DeltaZPrime = _DeltaZSignFlip*DeltaZ*sTag + _DeltaZOffset
      return DeltaZPrime


    def HiDelZPdG(self, DeltaZ, STag, PdG):
      sTag = int(SignOfTag(STag))
      DeltaZPrime = _DeltaZSignFlip*(DeltaZ - PdG)*sTag + _DeltaZOffset
      return DeltaZPrime
    

    def UnHideDeltaZ(self, DeltaZPrime, STag):
      sTag = int(SignOfTag(STag))
      DeltaZ = (DeltaZPrime - _DeltaZOffset)/(sTag*_DeltaZSignFlip)
      return DeltaZ
    

    def UnHiDelZPdG(self, DeltaZPrime, STag, PdG):
      sTag = int(SignOfTag(STag))
      DeltaZ = PdG + (DeltaZPrime - _DeltaZOffset)/(sTag*_DeltaZSignFlip)
      return DeltaZ
    

    def UnHideAsym(self, AsymPrime):
      if(self._mode == dataonly):
          return AsymPrime
      Asym = (AsymPrime - _AsymOffset)/_AsymSignFlip
      return Asym

    
    def HideAsym(self, Asym):
      if(self._mode == dataonly):
          return Asym
      AsymPrime = Asym*_AsymSignFlip + _AsymOffset
      return AsymPrime
    
    
    def UnHideDeltaM(self, DeltaMPrime):
      if(self._mode == dataonly):
          return DeltaMPrime
      DeltaM = DeltaMPrime - _DeltaMOffset
      return DeltaM
    
    def HideDeltaM(self, DeltaM):
      if(mode()==dataonly): return DeltaM
      DeltaMPrime = DeltaM + _DeltaMOffset
      return DeltaMPrime
    
    def UnHiAsPdG(AsymPrime, PdG):
      if(self._mode ==dataonly): return AsymPrime
      Asym = PdG + (AsymPrime - _AsymOffset)/_AsymSignFlip
      return Asym
    

    def MysteryPhase(self):
        if(self._mode ==dataonly): return 0.0
        return _MysteryPhase
    

    def HiAsPdG(self, Asym, PdG):
      if(self._mode ==dataonly): return Asym
      AsymPrime = (Asym - PdG)*_AsymSignFlip + _AsymOffset
      return AsymPrime
    
    def UnHidePrecision(self, PrecisionPrime):
      if(self._mode ==dataonly): return PrecisionPrime
      Precision = 0.0
      if (_PrecisionSignFlip>0):
        Precision = PrecisionPrime - self._PrecisionOffset
      else:
        Precision = 2.0*_PrecisionCentralValue - PrecisionPrime + self._PrecisionOffset
      return Precision
    

    def HidePrecision(self, Precision):
      if(self._mode ==dataonly):
          return Precision
      PrecisionPrime = 0.0
      if (_PrecisionSignFlip>0):
        PrecisionPrime = Precision + self._PrecisionOffset
      else:
        PrecisionPrime = 2.0*_PrecisionCentralValue - Precision + self._PrecisionOffset
      return PrecisionPrime
    

    def UnHideOffset(self, PrecisionPrime):
      if(self._mode ==dataonly): return PrecisionPrime
      return PrecisionPrime - self._PrecisionOffset
    
    
    def HideOffset(self, Precision):
      if(self._mode ==dataonly):
          return Precision
      return Precision + self._PrecisionOffset
    

    def UnHideUniform(self, PrecisionPrime):
      if(self._mode ==dataonly):
          return PrecisionPrime
      return PrecisionPrime - self._PrecisionUniform
    
    
    def HideUniform(self, Precision):
      if(self._mode ==dataonly):
          return Precision
      return Precision + self._PrecisionUniform
    
    
    def RandomizeTag(self, STag, EventNumber:int):
      Seed = EventNumber % 7997 + 2
      r = self.PseudoRandom(Seed)
      STagPrime =0.0
      if (r < self._STagConstant):
          STagPrime = STag
      else:
          STagPrime = -1.0 * STag 
      return STagPrime
    
       
    def Randomizer(self, string_seed:str):
      len_alphabet = 26
      lowerseed = ""
      lowerseed += self._stSeed
      lengthSeed = len(lowerseed)
      lowerseed = lowerseed.lower() 
      sumSeed:int = 0 # integer
    
      for i in range(0, lengthSeed):
          for j in range(0, 26):
              if lowerseed[i] == string_seed[j]:
                  if self._s2bMode:
                      sumSeed = (j<<(5*(i%3)))^sumSeed
                  else:
                      sumSeed = sumSeed + j
    
      if (lengthSeed<5 | ((sumSeed<1 | sumSeed>8000) and not self._s2bMode)):
        print("RooBlindTools::Randomizer: Your String Seed is Bad:")
      ia = 8121 # integer
      ic = 28411 # integer
      im = 134456 # integer
      jRan = (sumSeed*ia + ic) % im # unsigned int
      jRan = (jRan*ia + ic) % im
      jRan = (jRan*ia + ic) % im
      jRan = (jRan*ia + ic) % im
      theRan = float(jRan) / float(im)     
      return theRan    #theRan is between 0.0 - 1.0


    def PseudoRandom(self, Seed:int):
        if (Seed<1 or Seed>8000 ) :
            print("RooBlindTools::PseudoRandom: Your integer Seed is Bad")
        ia = 8121
        ic = 28411
        im = 134456
        jRan = (Seed*ia + ic) % im
        jRan = (jRan*ia + ic) % im
        jRan = (jRan*ia + ic) % im
        jRan = (jRan*ia + ic) % im
        theRan = float(jRan) / float(im)     
        return theRan  # theRan is between 0.0 - 1.0
    
    
    def MakeOffset(self, StringAlphabet:str):
      theRan = self.Randomizer(StringAlphabet)
      theOffset = (2.0)*theRan - (1.0)
      return theOffset # theOffset lies between -1.0 and 1.0
    
    
    def MakeGaussianOffset(self, StringAlphabet:str):
        theRan1 = self.Randomizer(StringAlphabet)
        theRan2 = self.Randomizer("cdefghijklmnopqrstuvwxyzab")
        if (theRan1==0.0 or theRan1==1.0):
          theRan1 = 0.5
        if (theRan2==0.0 or theRan2==1.0):
          theRan2 = 0.5
        theOffset = np.sin(2.0*3.14159*theRan1) * np.sqrt(-2.0*np.log(theRan2))
        return theOffset # theOffset is Gaussian with mean 0, sigma 1
    
    
    def MakeSignFlip(self, StringAlphabet:str):
        theRan = self.Randomizer(StringAlphabet)
        theSignFlip = 1.0
        if (theRan>0.5):
          theSignFlip = 1.0
        else:
          theSignFlip = -1.0
        return theSignFlip  # theSignFlip is = +1 or -1
    

    def SignOfTag(self, STag):
        sTag = 0
        if (STag < 0.0):
            sTag = -1
        elif (STag > 0.0):
            sTag = 1
        else:
            sTag = 1
        return sTag


class RooUnblindUniform(object):
    def __init__(self, name:str, title:str, blindString:str, scale:float, cpasym=0):
        self._value = cpasym
        self._blindEngine = RooBlindTools(blindString, False, 0, scale)

    def evaluate(self):
        return self._blindEngine.UnHideUniform(self._value)

