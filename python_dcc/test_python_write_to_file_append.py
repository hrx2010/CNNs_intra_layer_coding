
for i in range(10):

	file_out = open("test_write_append.txt" , "a+")
	
	file_out.write("values %d\n" % (i+1))

	file_out.close()


