﻿IMPORT ML;
IMPORT ML.Types;
cTypeRecord := RECORD
  ML.Types.t_FieldReal a1 ;
  ML.Types.t_FieldReal a2 ;
  ML.Types.t_FieldReal a3 ;
  ML.Types.t_FieldReal a4 ;
  ML.Types.t_FieldReal a5 ;
  ML.Types.t_FieldReal a6 ;
  ML.Types.t_FieldReal a7 ;
  ML.Types.t_FieldReal a8 ;
  ML.Types.t_FieldReal a9 ;
  ML.Types.t_FieldReal a10 ;
  ML.Types.t_FieldReal a11 ;
  ML.Types.t_FieldReal a12 ;
  ML.Types.t_FieldReal a13 ;
  ML.Types.t_FieldReal a14 ;
  ML.Types.t_FieldReal a15 ;
  ML.Types.t_FieldReal a16 ;
  ML.Types.t_FieldReal a17 ;
  ML.Types.t_FieldReal a18 ;
  ML.Types.t_FieldReal a19 ;
  ML.Types.t_FieldReal a20 ;
  ML.Types.t_FieldReal a21 ;
  ML.Types.t_FieldReal a22 ;
  ML.Types.t_FieldReal a23 ;
  ML.Types.t_FieldReal a24 ;
  ML.Types.t_FieldReal a25 ;
  ML.Types.t_FieldReal a26 ;
  ML.Types.t_FieldReal a27 ;
  ML.Types.t_FieldReal a28 ;
  ML.Types.t_FieldReal a29 ;
  ML.Types.t_FieldReal a30 ;
  ML.Types.t_FieldReal a31 ;
  ML.Types.t_FieldReal a32 ;
  ML.Types.t_FieldReal a33 ;
  ML.Types.t_FieldReal a34 ;
  ML.Types.t_FieldReal a35 ;
  ML.Types.t_FieldReal a36 ;
  ML.Types.t_FieldReal a37 ;
  ML.Types.t_FieldReal a38 ;
  ML.Types.t_FieldReal a39 ;
  ML.Types.t_FieldReal a40 ;
  ML.Types.t_FieldReal a41 ;
  ML.Types.t_FieldReal a42 ;
  ML.Types.t_FieldReal a43 ;
  ML.Types.t_FieldReal a44 ;
  ML.Types.t_FieldReal a45 ;
  ML.Types.t_FieldReal a46 ;
  ML.Types.t_FieldReal a47 ;
  ML.Types.t_FieldReal a48 ;
  ML.Types.t_FieldReal a49 ;
  ML.Types.t_FieldReal a50 ;
  ML.Types.t_FieldReal a51 ;
  ML.Types.t_FieldReal a52 ;
  ML.Types.t_FieldReal a53 ;
  ML.Types.t_FieldReal a54 ;
  ML.Types.t_FieldReal a55 ;
  ML.Types.t_FieldReal a56 ;
  ML.Types.t_FieldReal a57 ;
  ML.Types.t_FieldReal a58 ;
  ML.Types.t_FieldReal a59 ;
  ML.Types.t_FieldReal a60 ;
  ML.Types.t_FieldReal a61 ;
  ML.Types.t_FieldReal a62 ;
  ML.Types.t_FieldReal a63 ;
  ML.Types.t_FieldReal a64 ;
  ML.Types.t_FieldReal a65 ;
  ML.Types.t_FieldReal a66 ;
  ML.Types.t_FieldReal a67 ;
  ML.Types.t_FieldReal a68 ;
  ML.Types.t_FieldReal a69 ;
  ML.Types.t_FieldReal a70 ;
  ML.Types.t_FieldReal a71 ;
  ML.Types.t_FieldReal a72 ;
  ML.Types.t_FieldReal a73 ;
  ML.Types.t_FieldReal a74 ;
  ML.Types.t_FieldReal a75 ;
  ML.Types.t_FieldReal a76 ;
  ML.Types.t_FieldReal a77 ;
  ML.Types.t_FieldReal a78 ;
  ML.Types.t_FieldReal a79 ;
  ML.Types.t_FieldReal a80 ;
  ML.Types.t_FieldReal a81 ;
  ML.Types.t_FieldReal a82 ;
  ML.Types.t_FieldReal a83 ;
  ML.Types.t_FieldReal a84 ;
  ML.Types.t_FieldReal a85 ;
  ML.Types.t_FieldReal a86 ;
  ML.Types.t_FieldReal a87 ;
  ML.Types.t_FieldReal a88 ;
  ML.Types.t_FieldReal a89 ;
  ML.Types.t_FieldReal a90 ;
  ML.Types.t_FieldReal a91 ;
  ML.Types.t_FieldReal a92 ;
  ML.Types.t_FieldReal a93 ;
  ML.Types.t_FieldReal a94 ;
  ML.Types.t_FieldReal a95 ;
  ML.Types.t_FieldReal a96 ;
  ML.Types.t_FieldReal a97 ;
  ML.Types.t_FieldReal a98 ;
  ML.Types.t_FieldReal a99 ;
  ML.Types.t_FieldReal a100 ;
  ML.Types.t_FieldReal class;
END;
DS1:= DATASET( '~vherrara::datasets::ds1_100', cTypeRecord, CSV(HEADING(1)));
ML.AppendID(DS1, id, DS1_Id);
//wc_DS1:= full_id_ds(id <= 100000);  // smaller dataset
wc_DS1:= DS1_Id;
ML.ToField(wc_DS1, full_ds);
indepData:= full_ds(number<101);
depData:= ML.Discretize.ByRounding(full_ds(number=101));
minNumObj:= 2;    maxLevel := 50;

trainer1:= ML.Classify.DecisionTree.C45Binary(minNumObj, maxLevel); 
tmod:= trainer1.LearnC(indepData, depData);
tmodel:= trainer1.Model(tmod);
OUTPUT(SORT(tmodel, node_id, new_node_id), ALL, NAMED('TreeModel'));
results1:= trainer1.ClassifyC(indepData, tmod);
OUTPUT(results1, ALL, NAMED('ClassificationResults'));
results11:= ML.Classify.Compare(PROJECT(depData, TRANSFORM(ML.Types.DiscreteField,SELF.number:=1, SELF:=LEFT)), results1);
OUTPUT(results11.CrossAssignments, NAMED('CrossAssig1'));
OUTPUT(results11.RecallByClass, NAMED('RecallByClass1'));
OUTPUT(results11.PrecisionByClass, NAMED('PrecisionByClass1'));


