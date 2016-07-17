﻿IMPORT ML.Mat AS ML_Mat;
EXPORT Householder(DATASET(ML_Mat.Types.VecElement) X, ML_Mat.Types.t_Index k, ML_Mat.Types.t_Index Dim=1) := MODULE

	/* 
		HTA - Householder Transformation Algorithm
			Computes the Householder reflection matrix for use with QR decomposition

			Input:  vector X, k an index < length(X)
			Output: a matrix H that annihilates entries in the product H*X below index k
	*/
  EXPORT HTA := MODULE
	   EXPORT Default := MODULE,VIRTUAL
		  EXPORT IdentityM := IF(Dim>ML_Mat.Vec.Length(X), ML_Mat.Identity(Dim), ML_Mat.Identity(ML_Mat.Vec.Length(X)));
			EXPORT DATASET(ML_Mat.Types.Element) hReflection := DATASET([],ML_Mat.Types.Element);
			
		 END;
		 
		  // Householder Vector
			HouseV(DATASET(ML_Mat.Types.VecElement) X, ML_Mat.Types.t_Index k) := FUNCTION
				xk := X(x=k)[1].value;
				alpha := IF(xk>=0, -ML_Mat.Vec.Norm(X), ML_Mat.Vec.Norm(X));
				vk := IF (alpha=0, 1, SQRT(0.5*(1-xk/alpha)));
				p := - alpha * vk;
				RETURN PROJECT(X, TRANSFORM(ML_Mat.Types.VecElement,SELF.value := IF(LEFT.x=k, vk, LEFT.value/(2*p)), SELF :=LEFT));
			END; 
			
		 // Source: Atkinson, Section 9.3, p. 611	
		 EXPORT Atkinson := MODULE(Default)
				hV := HouseV(X(x>=k),k);
				houseVec := ML_Mat.Vec.ToCol(hV, 1);
				EXPORT DATASET(ML_Mat.Types.Element) hReflection := ML_Mat.Sub(IdentityM, ML_Mat.Scale(ML_Mat.Mul(houseVec,ML_Mat.Trans(houseVec)),2));
		 END;
		 
		 // Source: Golub and Van Loan, "Matrix Computations" p. 210
		 EXPORT Golub := MODULE(Default)
				VkValue := X(x=k)[1].value;
				VkPlus := X(x>k);
				sigma := ML_Mat.Vec.Dot(VkPlus, VkPlus);
	
				mu := SQRT(VkValue*VkValue + sigma);
				newVkValue := IF(sigma=0,1,IF(VkValue<=0, VkValue-mu, -sigma/(VkValue+mu) ));
				beta := IF( sigma=0, 0, 2*(newVkValue*newVkValue)/(sigma + (newVkValue*newVkValue)));
				
				newVkElem0 := X[1];
				newVkElem := PROJECT(newVkElem0,TRANSFORM(ML_Mat.Types.Element,SELF.x:=k,SELF.y:=1,SELF.value := newVkValue));

				hV := PROJECT(newVkElem + VkPlus,TRANSFORM(ML_Mat.Types.Element,SELF.value:=LEFT.value/newVkValue, SELF := LEFT));
				EXPORT DATASET(ML_Mat.Types.Element) hReflection := ML_Mat.Sub(IdentityM, ML_Mat.Scale(ML_Mat.Mul(hV,ML_Mat.Trans(hV)),Beta));
		 END;
	
	END;

	EXPORT Reflection(HTA.Default Control = HTA.Golub) := FUNCTION
		RETURN Control.hReflection;
	END;

END;