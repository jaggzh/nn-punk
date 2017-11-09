#!/usr/bin/python2.7
import re

abbrs_file = "abbreviations.txt"

def load_abbreviations_re():
	with open(abbrs_file) as f:
		content = f.readlines()
	content = [x.strip() for x in content] 
	#pf(content)
	abbr_re = "(" + '|'.join(content) + ")\."
	#pf(abbr_re)
	#exit(0)
	return re.compile(abbr_re)
