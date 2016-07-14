﻿IMPORT * FROM $;
IMPORT ML, ML.Types;
IMPORT ML.Tests.Deprecated AS TestML;

output(TestML.ChickWeight,named('CheckWeight'));
/*
ChickWeightRec := RECORD
  unsigned rid;
  unsigned weight;
  unsigned Time;
  unsigned Chick;
  unsigned Diet;
END;
*/

NumericField := ML.Types.NumericField;
/* THE ABOVE RECORD LAYOUT LOOKS LIKE THE FOLLOWING:
EXPORT NumericField := RECORD
  t_RecordID id;
  t_FieldNumber number;
  t_FieldReal value;
  END;
*/

ml.ToField(TestML.ChickWeight,ChickWeight2NumericFieldDS);
output(sort(ChickWeight2NumericFieldDS,id,number),named('ChickWeight2NumericFieldDS'));

SimpleRankedChickWeight2NumericFieldDS := ML.FieldAggregates(ChickWeight2NumericFieldDS).SimpleRanked;
output(SimpleRankedChickWeight2NumericFieldDS,named('SimpleRankedChickWeight2NumericFieldDS'));

/*
MR := RECORD
  SimpleRankedChickWeight2NumericFieldDS.Number;
  SimpleRankedChickWeight2NumericFieldDS.Value;
  Types.t_FieldReal Pos := AVE(GROUP,SimpleRankedChickWeight2NumericFieldDS.Pos);
  UNSIGNED valcount:=COUNT(GROUP);
END;


T := TABLE(SimpleRankedChickWeight2NumericFieldDS,MR,Number,Value);
output(T,named('T'));
*/

RankedChickWeight2NumericFieldDS := ML.FieldAggregates(ChickWeight2NumericFieldDS).RankedInput;
output(sort(RankedChickWeight2NumericFieldDS,-number,pos),named('RankedChickWeight2NumericFieldDS'));
