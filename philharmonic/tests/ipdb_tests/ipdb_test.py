import sys
parents, babies = (1, 1)
while babies < 100:
	# if 'ipdb' not in sys.modules:
	# 	import ipdb; ipdb.set_trace()
	
	print 'This generation has {0} babies'.format(babies)
	parents, babies = (babies, parents + babies)