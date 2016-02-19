## THIS FILE HELPS YOU GET A SAMPLE OF DATA
## DOESN'T PULL A RANDOM SAMPLE BUT SHOULD (in the future). 


print "What file do you want to sample?"
open_file = raw_input('>')
print "   ***   "
print "How many lines of data do you want to sample"
sample_num = int(raw_input('>'))
print "   ***   "
print "Save sample file as?"
write_file = raw_input('>')
print "   ***   "

with open(open_file, 'r') as in_file:
	for index in range(0, sample_num):
		line = in_file.next()
		line = str(line) # just in case not seen as string
		with open (write_file , 'a+') as out_file:
			out_file.write(line)
