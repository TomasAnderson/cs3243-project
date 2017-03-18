import os.path,subprocess
from subprocess import STDOUT,PIPE
import csv,time

def compile_java(java_file):
    subprocess.check_call(['javac', java_file])

def execute_java(java_file, stdin):
    java_class,ext = os.path.splitext(java_file)
    cmd = ['java', java_class]
    proc = subprocess.Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=STDOUT)
    stdout,stderr = proc.communicate(stdin)
    return stdout
def get_count(output):
	return output.split()[3]

compile_java('PlayerSkeleton.java')
results = []
for i in range(20):
	output = execute_java('PlayerSkeleton.java', 'Jon')
	results.append(get_count(output))
timestamp = time.strftime('%s')
file_dir = './result/result_'+timestamp+'.csv'


with open(file_dir,'w') as f:
	fieldnames = ['round', 'count']
	writer = csv.DictWriter(f, fieldnames=fieldnames)
	writer.writeheader()
	index = 1
	for result in results:
		writer.writerow({'round':index,'count':result})
		index = index + 1


