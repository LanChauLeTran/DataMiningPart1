import re

"""
This script finds all association rules with one antecedent and one consequent
"""

with open("weka_apriori_output.txt") as f:
  data = f.read()
    
    
outstr = ""
expression = r'([0-9]+\. [A-Z0-9]+=[A-z]+ [0-9]+ ==> [A-Z0-9]+=[A-z]+ [0-9]+.*)' # https://regex101.com/r/VAFanj/1
for match in re.finditer(expression,data):
  print(match.group(1))
  outstr += match.group(1) + "\n"
  
with open("weka_apriori_one_implies_one.txt", 'w') as f:
  f.write(outstr)
