EXPORT Types := MODULE

        EXPORT t_Index := UNSIGNED4; // Supports matrices with up to 9B as the largest dimension
        EXPORT t_value := REAL8;
        EXPORT t_mu_no := UNSIGNED2; // Allow up to 64K matrices in one universe

        EXPORT Element := RECORD
                t_Index x; // X is rows
                t_Index y; // Y is columns
                t_value value;
        END;

        EXPORT VecElement := RECORD
                t_Index x; // a vector does not necessarily lay upon any given dimension
                t_Index y; // y will always be 1
                t_value value;
          END;

        EXPORT MUElement := RECORD(Element)
                t_mu_no no; // The number of the matrix within the universe
        END;

        EXPORT t_RecordID := UNSIGNED;
        EXPORT t_FieldNumber := UNSIGNED4;
        EXPORT t_FieldReal := REAL8;
        EXPORT t_FieldSign := INTEGER1;
        EXPORT t_Discrete := INTEGER4; // The number of 'groups' a population may be divided into - negative to allow for classificaiton to 'undershoot'
        EXPORT t_Item := UNSIGNED4; // Currently allows up to 9B different elements
        EXPORT t_Count := t_RecordID; // Possible to count every record

        EXPORT NumericField := RECORD
                t_RecordID id;
                t_FieldNumber number;
                t_FieldReal value;
          END;

        EXPORT DiscreteField := RECORD
                // t_RecordID id;
                // t_FieldNumber number;
                // t_Discrete value;
          END;

        /*EXPORT l_result := RECORD(DiscreteField)
          REAL8 conf;  // Confidence - high is good
          END;

        EXPORT ItemElement := RECORD
                t_Item value;
                t_RecordId id;
          END;
                
        EXPORT ToMatrix(DATASET(NumericField) d):=FUNCTION
          RETURN PROJECT(d,TRANSFORM(Mat.Types.Element,SELF.x:=(TYPEOF(Mat.Types.Element.x))LEFT.id;SELF.y:=(TYPEOF(Mat.Types.Element.y))LEFT.number;SELF.value:=(TYPEOF(Mat.Types.Element.value))LEFT.value;));
        END;

        EXPORT FromMatrix(DATASET(Mat.Types.Element) d):=FUNCTION
          RETURN PROJECT(d,TRANSFORM(NumericField,SELF.id:=(TYPEOF(NumericField.id))LEFT.x;SELF.number:=(TYPEOF(NumericField.number))LEFT.y;SELF.value:=(TYPEOF(NumericField.value))LEFT.value;));
        END;

        // Decision Trees and Random Forest basics
        EXPORT t_node  := INTEGER4;   // Node Identifier Number in Decision Trees and Random Forest
        EXPORT t_level := UNSIGNED2;  // Tree Level Number
        EXPORT NodeID  := RECORD
          t_node  node_id;
          t_level level;
          END;*/


END;