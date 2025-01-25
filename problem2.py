import logging
import mrjob
from mrjob.job import MRJob
import logging
import re

class SharedPrefixes(MRJob):

    ### TODO: Add/Change code below
    WORD_RE = re.compile("[a-z]+") ### regex101

    def mapper( self, _ , line ):
        words = self.WORD_RE.findall( line.lower() )
        for w in words:
            yield None, w
    
    def combiner( self, _, words ):
        unique_words = set( words )

        for word in unique_words:
            yield None, word
            # yield word[0], word

    def reducer(self, key, stats):
        # first_char = key
        word_list = list( stats )
        sorted_words = sorted( set( word_list ) )


        shared_prefix_len = 0
        tot_word_len = 0
        tot_word = len( sorted_words )

        if( tot_word > 1 ):
            for i in range( tot_word - 1):
                w1,w2 = sorted_words[ i ], sorted_words[ i + 1 ]
                sh_pref = self.calc_sh_pref( w1, w2 ) 
                shared_prefix_len += sh_pref

        tot_word_len = sum( len( w ) for w in sorted_words )

        avg_pref_len = shared_prefix_len / max( tot_word - 1, 1 ) if tot_word > 1 else 0
        avg_word_len = tot_word_len / tot_word

        ### yielding final values should be as below, Do Not Change
        yield "Unique words", tot_word
        yield "Average shared Prefix Length", avg_pref_len
        yield "Average word length", avg_word_len

    
    """lambda func"""
    @staticmethod
    def calc_sh_pref( word1, word2 ):
        
        prefix_length = 0
        
        for c1, c2 in zip( word1, word2 ):
            
            if c1 == c2:
                prefix_length += 1
            else:
                break
        
        return prefix_length 
    ### TODO: Add/Change code above


if __name__ == '__main__':
    SharedPrefixes.run()




"""====== prefix length for unique words ======
No configs found; falling back on auto-configuration
No configs specified for inline runner
Creating temp directory /tmp/problem2.tusker.20250125.150207.433791
Running step 1 of 1...
job output is in /tmp/problem2.tusker.20250125.150207.433791/output
Streaming final output from /tmp/problem2.tusker.20250125.150207.433791/output...
"Unique words"  23149
"Average shared Prefix Length"  5.127138413685848
"Average word length"   7.634887036157069
Removing temp directory /tmp/problem2.tusker.20250125.150207.433791
"""