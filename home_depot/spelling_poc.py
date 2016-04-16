# generally useful/needed
from BeautifulSoup import BeautifulSoup
import csv
import pandas as pd
#spellcheck stuff
import enchant 
import enchant.checker
from enchant.checker.CmdLineChecker import CmdLineChecker



def initialize(): 
	chk_file = raw_input("What is the file to spellcheck?    ")
	field = raw_input("What FIELD do you want to spellcheck?   ")
	s_file = raw_input("What is name of final file?    ")

	checker = enchant.checker.SpellChecker("en_US")
	cmdln = CmdLineChecker()

	file_data = pd.read_csv(chk_file)

	fields = list(file_data.apply(lambda x:'%s' % (x[field]),axis=1))

	# maybe i don't even need this...
	#fields = strip_html(fields)

	corrected_text = []
	for data_field in fields:
		checker.set_text(str(data_field))
		for err in checker:
			print err.word
			print err.suggest()
			correct = raw_input("provide 0-index int of correct word or i to ignore, e to edit ")
			if correct == 'i':
				pass
			elif correct == 'e':
				suggest = raw_input("")
				err.replace(suggest)
			else:
				correct = int(correct)
				suggest = err.suggest()[correct]
				err.replace(suggest)
		corrected_text.append(checker.get_text())

	saved_file = write_fixed_file(corrected_text, s_file)

def strip_html(raw_text):
	text = re.sub('<[^>]*>', '', raw_text)
	text = re.sub('.[^>]*}', '', text)
	return text

initialize()