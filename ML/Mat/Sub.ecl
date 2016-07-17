﻿IMPORT ML.Mat AS ML_Mat;
EXPORT Sub(DATASET(ML_Mat.Types.Element) l,DATASET(ML_Mat.Types.Element) r) := FUNCTION
StatsL := ML_Mat.Has(l).Stats;
StatsR := ML_Mat.Has(r).Stats;
SizeMatch := ~ML_Mat.Strict OR (StatsL.XMax=StatsR.XMax AND StatsL.YMax=StatsR.YMax);

// Only slight nastiness is that these matrices may be sparse - so either side could be null
ML_Mat.Types.Element Su(l le,r ri) := TRANSFORM
    SELF.x := IF ( le.x = 0, ri.x, le.x );
    SELF.y := IF ( le.y = 0, ri.y, le.y );
	  SELF.value := le.value - ri.value; // Fortuitously; 0 is the null value
  END;

assertCondition := ~(ML_Mat.Debug AND ~SizeMatch);	
checkAssert := ASSERT(assertCondition, 'Sub FAILED - Size mismatch', FAIL);	
result := IF(SizeMatch, 
				JOIN(l,r,LEFT.x=RIGHT.x AND LEFT.y=RIGHT.y,Su(LEFT,RIGHT),FULL OUTER),
				DATASET([], ML_Mat.Types.Element));
	RETURN WHEN(result, checkAssert);

END;