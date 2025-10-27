def hamming_distance(hashstr_1:str,hashstr_2:str)->int:
    """
        Calculate hamming distance between two vectors based in their hash strings

        It is the number of positions at which the two strings differ

    """
    
    xor = int(hashstr_1,2) ^ int(hashstr_2,2)

    setbits=0

    while xor>0:
        setbits+= xor & 1
        xor>>=1
    
    return setbits

