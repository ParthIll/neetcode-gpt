from typing import List


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        # 1. Split corpus into a list of individual characters
        # 2. For each merge step:
        #    a. Count frequency of all adjacent token pairs
        #    b. Find the most frequent pair (break ties lexicographically)
        #    c. Merge all non-overlapping occurrences left to right
        #    d. Record the merge as [token_a, token_b]
        # 3. Return the list of merges performed
        corpuslist = list(corpus)
        retlist=[]
        for i in range(num_merges):
            freqMap={}
            l=0
            r=1
            while r<len(corpuslist):
                freqMap[(corpuslist[l],corpuslist[r])] = freqMap.get((corpuslist[l],corpuslist[r]),0)+1
                l+=1
                r+=1
            freqMap = dict(sorted(freqMap.items(), key=lambda item: item[0],reverse=True))
            freqMap = dict(sorted(freqMap.items(), key=lambda item: item[1]))
            print(freqMap)
            first = list(freqMap.keys())[-1][0]
            second = list(freqMap.keys())[-1][1]
            for i in range(len(corpuslist)):
                try:
                    if corpuslist[i] ==  first and corpuslist[i+1]==second:
                        corpuslist[i] = first+second
                        corpuslist.pop(i+1)
                except:
                    break
            freqMap.clear()
            retlist.append([first,second])
        return retlist
        