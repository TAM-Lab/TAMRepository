# merge a list of files (directory) and output specified columns, or output some columns for a single file.
# Created by Tony HAO

import W_utility.file as ufile
from W_utility.log import ext_print
import os,sys


def file_merge (fdin, fout, columns, format):
	# read input data
	if fdin is None or fdin =="":
		return False
	texts = ufile.read(fdin) # a specific file or a directory
	result = []
	print texts
	if columns =="all":
		result = texts
	else:
		cols = columns.split('|')
		for text in texts:
			if len(cols) == 1:
				result.append(text[int(cols) -1])
			else:
				for col in cols:
					result.append(text[int(col) -1])

	print ext_print ('get %d in total' % len(result))

	# get output data directory
	if fout is None:
		fout = os.path.splitext(fdin)[0] + "_merged" + format
	# output detailed result into file
	if format == "" or ".txt":
		ufile.write_file(fout, result, False)
	elif format == ".csv":
		ufile.write_csv(fout, result)
	print ext_print ('saved result into: %s' % fout)
	print ext_print ('all tasks completed\n')
	return True


# main function	

# processing the command line options
import argparse
def _process_args():
	parser = argparse.ArgumentParser(description='Merge a Collection of Documents')
	parser.add_argument('-i', default=r"E:\Simplify\ngrams\3grams", help='input directory (automatic find and read each file as a document)')
	parser.add_argument('-o', default=None, help='output file; None: get default output path')
	parser.add_argument('-c', default="all", help='The columns you want for output into result files, [all]: all columns; [1|3|4]: 1&3&4 columns; [2]: just 2 columns')
	parser.add_argument('-f', default="", help='The formats of result files [.txt] text files; [.csv] csv files')
	return parser.parse_args(sys.argv[1:])


if __name__ == '__main__' :
	print ''
	args = _process_args()
	file_merge (args.i, args.o, args.c, args.f)
	print ''
