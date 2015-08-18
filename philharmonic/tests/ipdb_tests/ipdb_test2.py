import sys
import pdb
parents, babies = (1, 1)
while babies < 100:
	# pdb.set_trace()
	print 'This generation has {0} babies'.format(babies)
	# pdb.set_trace()
	parents, babies = (babies, parents + babies)