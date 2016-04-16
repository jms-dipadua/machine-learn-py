"""Spelling Corrector.

Copyright 2007 Peter Norvig. 
Open source code under MIT license: http://www.opensource.org/licenses/mit-license.php
"""

import re, collections
import sys
import numpy as np
import pandas as pd
#more supplemental spellcheck stuff
import enchant 
import enchant.checker

def initialize():
  chk_file = raw_input("What is the file to spellcheck?    ")
  s_file = raw_input("What is name of final file?    ")

  checker = enchant.checker.SpellChecker("en_US")

  file_data = pd.read_csv(chk_file)
  search_terms = file_data['search_term'].str.lower().str.split()
  search_terms_fixed = []
  for search_term in search_terms:
    correct_wrds = []
    # following will look through search terms and only trigger the bayes check if there is a suspected error via enchant
    for term in search_term:
      if (checker.check(term) == False) and (term not in BRAND_S_WORDS):
        correct_wrds.append(correct(term))
      else:
        correct_wrds.append(term)

    search_terms_fixed.append(correct_wrds)

  file_data['search_terms_fixed'] = search_terms_fixed
  final_file = file_data.to_csv(s_file,index=False)

    

def words(text): return re.findall('[a-z]+', text.lower()) 

def train(features):
    model = collections.defaultdict(lambda: 1)
    for f in features:
        model[f] += 1
    return model

#NWORDS = train(words(file('big_mod.txt').read()))
NWORDS = train(words(file('data/product_descriptions_etc.txt').read()))
BRAND_S_WORDS = list('rheem', 'kohler', 'hampton', 'rosario', 'owens', 'kingsley', 'stanley', 'melnor', 'ge', 'fuego', 'ryobi', 'andersen', 'montagna', 'westminister', 'dewalt', 'lennox', 'quikrete', 'paslode', 'closetmaid', 'prehung', 'backsplash', 'malibu', 'kobalt', 'rustoleum', 'wonderboard,gilmore', 'electrolux', 'samsung', 'jeldwen', 'milwaukee', 'pex', 'werner', 'decora', 'dpdt', 'azek', 'grafton', 'maytag', 'dremel', 'yonkers', 'swanstone', 'martha', 'stewart', 'formica', 'countertop', 'honda', 'valvoline', 'everbilt', 'bullnose', 'wonderboard', 'honeywell', 'rheem', 'riosa', 'wilsonart', 'moen', 'durock', 'rayovac', 'masonite', 'sauder', 'tv', 'maglite', 'vormax', 'bosch', 'french', 'paracord', 'wellworth', 'banbury', 'btu')

alphabet = 'abcdefghijklmnopqrstuvwxyz'

def edits1(word):
   s = [(word[:i], word[i:]) for i in range(len(word) + 1)]
   deletes    = [a + b[1:] for a, b in s if b]
   transposes = [a + b[1] + b[0] + b[2:] for a, b in s if len(b)>1]
   replaces   = [a + c + b[1:] for a, b in s for c in alphabet if b]
   inserts    = [a + c + b     for a, b in s for c in alphabet]
   return set(deletes + transposes + replaces + inserts)

def known_edits2(word):
    return set(e2 for e1 in edits1(word) for e2 in edits1(e1) if e2 in NWORDS)

def known(words): return set(w for w in words if w in NWORDS)

def correct(word):
    print "checking word: %r"  % word
    candidates = known([word]) or known(edits1(word)) or known_edits2(word) or [word]
    best_match = max(candidates, key=NWORDS.get)
    print "best match:  %r"  % best_match
    return best_match

################ Testing code from here on ################

def spelltest(tests, bias=None, verbose=False):
    import time
    n, bad, unknown, start = 0, 0, 0, time.clock()
    if bias:
        for target in tests: NWORDS[target] += bias
    for target,wrongs in tests.items():
        for wrong in wrongs.split():
            n += 1
            w = correct(wrong)
            if w!=target:
                bad += 1
                unknown += (target not in NWORDS)
                if verbose:
                    print 'correct(%r) => %r (%d); expected %r (%d)' % (
                        wrong, w, NWORDS[w], target, NWORDS[target])
    return dict(bad=bad, n=n, bias=bias, pct=int(100. - 100.*bad/n), 
                unknown=unknown, secs=int(time.clock()-start) )



if __name__ == '__main__':
  initialize()