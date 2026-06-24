from typing import List, Dict

class Solution:
    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        # Tokenize each number using greedy left-to-right longest match.
        # Return a list of token lists showing how each number gets split.
        ret = []
        for num in numbers:
            inret = []
            numstr = str(num)
            l=0
            r=len(numstr)
            while l!=r:
                if numstr[l:r] in vocab:
                    inret.append(numstr[l:r])
                    l=r
                    r=len(numstr)
                else:
                    r-=1

            ret.append(inret)
        return ret

    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        # Count how many tokens the text uses with greedy tokenization.
        # Use greedy left-to-right longest match.


        ret=[]
        numstr=text
        l=0
        r=len(numstr)
        while l!=r:
                if numstr[l:r] in vocab:
                    ret.append(numstr[l:r])
                    l=r
                    r=len(numstr)
                else:
                    r-=1
        return len(ret)

    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        # Compute tokens-per-word ratio (fertility).
        # Higher = more expensive and less efficient.
        # Round to 4 decimal places.
        tokens = self.count_tokens(text,vocab)
        words = len(text.split())
        return round(tokens/words,4)
