﻿IMPORT * FROM ML;
IMPORT * FROM ML.Types;
cTypeRecord := RECORD
  Types.t_FieldReal a1 ;
  Types.t_FieldReal a2 ;
  Types.t_FieldReal a3 ;
  Types.t_FieldReal a4 ;
  Types.t_FieldReal a5 ;
  Types.t_FieldReal a6 ;
  Types.t_FieldReal a7 ;
  Types.t_FieldReal a8 ;
  Types.t_FieldReal a9 ;
  Types.t_FieldReal a10 ;
  Types.t_FieldReal a11 ;
  Types.t_FieldReal a12 ;
  Types.t_FieldReal a13 ;
  Types.t_FieldReal a14 ;
  Types.t_FieldReal a15 ;
  Types.t_FieldReal a16 ;
  Types.t_FieldReal a17 ;
  Types.t_FieldReal a18 ;
  Types.t_FieldReal a19 ;
  Types.t_FieldReal a20 ;
  Types.t_FieldReal a21 ;
  Types.t_FieldReal a22 ;
  Types.t_FieldReal a23 ;
  Types.t_FieldReal a24 ;
  Types.t_FieldReal a25 ;
  Types.t_FieldReal a26 ;
  Types.t_FieldReal a27 ;
  Types.t_FieldReal a28 ;
  Types.t_FieldReal a29 ;
  Types.t_FieldReal a30 ;
  Types.t_FieldReal a31 ;
  Types.t_FieldReal a32 ;
  Types.t_FieldReal a33 ;
  Types.t_FieldReal a34 ;
  Types.t_FieldReal a35 ;
  Types.t_FieldReal a36 ;
  Types.t_FieldReal a37 ;
  Types.t_FieldReal a38 ;
  Types.t_FieldReal a39 ;
  Types.t_FieldReal a40 ;
  Types.t_FieldReal a41 ;
  Types.t_FieldReal a42 ;
  Types.t_FieldReal a43 ;
  Types.t_FieldReal a44 ;
  Types.t_FieldReal a45 ;
  Types.t_FieldReal a46 ;
  Types.t_FieldReal a47 ;
  Types.t_FieldReal a48 ;
  Types.t_FieldReal a49 ;
  Types.t_FieldReal a50 ;
  Types.t_FieldReal a51 ;
  Types.t_FieldReal a52 ;
  Types.t_FieldReal a53 ;
  Types.t_FieldReal a54 ;
  Types.t_FieldReal a55 ;
  Types.t_FieldReal a56 ;
  Types.t_FieldReal a57 ;
  Types.t_FieldReal a58 ;
  Types.t_FieldReal a59 ;
  Types.t_FieldReal a60 ;
  Types.t_FieldReal a61 ;
  Types.t_FieldReal a62 ;
  Types.t_FieldReal a63 ;
  Types.t_FieldReal a64 ;
  Types.t_FieldReal a65 ;
  Types.t_FieldReal a66 ;
  Types.t_FieldReal a67 ;
  Types.t_FieldReal a68 ;
  Types.t_FieldReal a69 ;
  Types.t_FieldReal a70 ;
  Types.t_FieldReal a71 ;
  Types.t_FieldReal a72 ;
  Types.t_FieldReal a73 ;
  Types.t_FieldReal a74 ;
  Types.t_FieldReal a75 ;
  Types.t_FieldReal a76 ;
  Types.t_FieldReal a77 ;
  Types.t_FieldReal a78 ;
  Types.t_FieldReal a79 ;
  Types.t_FieldReal a80 ;
  Types.t_FieldReal a81 ;
  Types.t_FieldReal a82 ;
  Types.t_FieldReal a83 ;
  Types.t_FieldReal a84 ;
  Types.t_FieldReal a85 ;
  Types.t_FieldReal a86 ;
  Types.t_FieldReal a87 ;
  Types.t_FieldReal a88 ;
  Types.t_FieldReal a89 ;
  Types.t_FieldReal a90 ;
  Types.t_FieldReal a91 ;
  Types.t_FieldReal a92 ;
  Types.t_FieldReal a93 ;
  Types.t_FieldReal a94 ;
  Types.t_FieldReal a95 ;
  Types.t_FieldReal a96 ;
  Types.t_FieldReal a97 ;
  Types.t_FieldReal a98 ;
  Types.t_FieldReal a99 ;
  Types.t_FieldReal a100 ;
  Types.t_FieldReal class;
END;
DS1:= DATASET( '~vherrara::datasets::ds1_100', cTypeRecord, CSV(HEADING(1)));
AppendID(DS1, id, DS1_Id);
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
results11:= Classify.Compare(PROJECT(depData, TRANSFORM(Types.DiscreteField,SELF.number:=1, SELF:=LEFT)), results1);
OUTPUT(results11.CrossAssignments, NAMED('CrossAssig1'));
OUTPUT(results11.RecallByClass, NAMED('RecallByClass1'));
OUTPUT(results11.PrecisionByClass, NAMED('PrecisionByClass1'));


