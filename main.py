import sys
import os
import datetime
import argparse
import subprocess


class Time_Logger:


	def __init__(self, log_path):
		self.topic = ''
		self.log_path = log_path
		self.last_action = ''


	def read_input(self, input_str, cur_time):
		x = input_str.split(' ')
		command = x[0]
		argument = (' ').join(x[1:])
		
		if command == 'start':
			self.start(argument, cur_time)
		elif command == 'stop':
			self.stop(cur_time)
		elif command == 'note':
			self.note(argument, cur_time)
		elif command == 'open':
			self.open_log()
		else:
			print('# To start a new topic:            start {TOPIC}')
			print('# To add a note on current topic:  note {NOTE}')
			print('# To stop current topic:           stop')
			print('# To open log file in notepad:     open\n')

	def start(self, topic, cur_time):
		if (not self.topic and 
				self.topic != '-  start:' and
				self.topic != '-  stop:' and 
				topic):
			print(f'# Starting new topic "{topic}" at {cur_time}')
			with open(log_path, 'a') as fp:
				fp.write(f'@{topic}:\n')
				fp.write(f'-  start:  {cur_time}\n')
			self.topic = topic
			self.last_action = 'start'
		else:
			print(f"# Unable to add new topic {topic}. Current topic is '{self.topic}'")
			
	def stop(self, cur_time):
		if self.topic:
			print(f'# Stopping current topic "{self.topic}" at {cur_time}')
			with open(log_path, 'a') as fp:
				fp.write(f'-  stop:   {cur_time}\n\n')
			self.topic = ''
			self.last_action = 'stop'
		else: 
			print('# Unable to stop. There is no current topic.')

	def note(self, note_str, cur_time):
		if self.topic:
			print(f'Writing note to topic "{self.topic}" at {cur_time}')
			with open(log_path, 'a') as fp:
				if self.last_action != 'note':
					fp.write('-  notes:\n')
				fp.write(f'-  -  {note_str}\n')
			self.last_action = 'note'
		else:
			print('# Unable to add note. There is no current topic.')
	
	def open_log(self):
		os.startfile(log_path)
		while(process_exists('notepad.exe')): # change this to YOUR default txt editor.
			print('# Pausing for editing. Press "Enter" to continue after txt editor is closed.')
			input()
		print('# Finished editing. Notepad.exe has been closed.')
	

def process_exists(process_name):
	"""Returns True if {process_name} is running. Else, False

	This works on Windows - idk about about other OS.
	"""
	call = 'TASKLIST', '/FI', 'imagename eq %s' % process_name
	# use buildin check_output right away
	output = subprocess.check_output(call).decode()
	# check in last line for process name
	last_line = output.strip().split('\r\n')[-1]
	# because Fail message could be translated
	return last_line.lower().startswith(process_name.lower())


### main
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dir', default='2021_logs', 
    help='relative path to save current day txt file')
args = parser.parse_args()
if not os.path.isdir(args.dir):
	os.makedirs(args.dir)

run_date = datetime.date.today().strftime("%m-%d-%Y")
log_path = os.path.join(args.dir, run_date+'.txt')
run_day = run_date.split('-')[1]

### check if format of log file is correct


### run time logger
time_logger = Time_Logger(log_path)
print('# Time logger is running. Type "help" to see commands.')
while(run_day == datetime.date.today().strftime("%d")):
	
	try:
		input_str = input() 
		cur_time = datetime.datetime.now().strftime("%H:%M")
		time_logger.read_input(input_str, cur_time)
	except KeyboardInterrupt:
		cur_time = datetime.datetime.now().strftime("%H:%M")
		time_logger.stop(cur_time)
		sys.exit()

print('# It is a new day now. Re-run script to generate new txt file.')