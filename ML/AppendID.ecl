﻿//---------------------------------------------------------------------------
// Macro takes any structured dataset, and appends a unique 1-based record ID
// column to it 
//
//   dIn       : The name of the input dataset
//   idfield   : The name of the field to be appended containing the id 
//               for each row.  
//   dOut      : The name of the resulting dataset
//  
//  Examples:
//    ML.AppendID(dOrig, recID, dOrigWithId);
//---------------------------------------------------------------------------
IMPORT ML;
EXPORT AppendID(dIn,idfield,dOut) := MACRO
#uniquename(dInPrep)
%dInPrep%:=TABLE(dIn,{ML.Types.t_RecordID idfield:=0;dIn;});
#uniquename(tra)
TYPEOF(%dInPrep%) %tra%(dIn L, INTEGER C) := TRANSFORM
  SELF.idfield := C; 
  SELF := L;
  END;
dOut := PROJECT(dIn,%tra%(LEFT,COUNTER));
ENDMACRO;