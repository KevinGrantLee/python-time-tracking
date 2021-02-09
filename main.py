"""
TODO: write option to make visualization figures
"""


import sys
import os
import datetime
import argparse
import matplotlib.pyplot as plt
import subprocess


def process_exists(process_name):
    call = 'TASKLIST', '/FI', 'imagename eq %s' % process_name
    # use buildin check_output right away
    output = subprocess.check_output(call).decode()
    # check in last line for process name
    last_line = output.strip().split('\r\n')[-1]
    # because Fail message could be translated
    return last_line.lower().startswith(process_name.lower())


class time_logger:
	def __init__(self, log_path):
		self.topic = ''
		self.log_path = log_path
		# self.last_action_time = datetime.datetime.now().strftime("%H:%M")
		# self.last_topic = ''
		self.last_action = ''

	def read_input(self, input_str):
		x = input_str.split(' ')
		command = x[0]
		argument = (' ').join(x[1:])
		if command == 'start':
			self.start(argument)
		elif command == 'stop':
			self.stop()
		elif command == 'note':
			self.note(argument)
		elif command == 'open':
			self.open_log()
		else:
			# help 
			print('# To start a new topic: start {TOPIC}')
			print('# To add a note on current topic: note {NOTE}')
			print('# To stop current topic: stop')
			print('# To open log file in notepad: open')
			print('')

	def start(self, topic):
		if topic != '' and self.topic == '':
			current_time = datetime.datetime.now().strftime("%H:%M")
			print("# Starting new topic '{}' at {}".format(topic, current_time))
			self.topic = topic
			with open(log_path, 'a') as fp:
				fp.write(f'@{topic}:\n')
				fp.write(f'-  start:  {current_time}\n')
			self.last_action = 'start'
		else:
			print(f"# Unable to add new topic. Current topic is '{self.topic}'")
			
	def stop(self):
		if self.topic != '':
			current_time = datetime.datetime.now().strftime("%H:%M")
			# self.last_action_time = current_time
			print(f"# Stopping current topic '{self.topic}' at {current_time}")
			
			self.topic = ''
			with open(log_path, 'a') as fp:
				fp.write(f'-  stop:   {current_time}\n\n')
			self.last_action = 'stop'
		else: 
			print('# Unable to stop. There is no current topic.')

	def note(self, note_str):
		current_time = datetime.datetime.now().strftime("%H:%M")
		if self.topic != '':
			print(f"Writing note to topic '{self.topic}' at {current_time}")
			with open(log_path, 'a') as fp:
				if self.last_action != 'note':
					fp.write('-  notes:\n')
				fp.write(f'-  -  {note_str}\n')
			self.last_action = 'note'
		else:
			print('# Unable to add note when there is no current topic.')
	def open_log(self):
		print('# Pausing for editing. All commands in prompt will be ignored.')
		os.startfile(log_path)
		while(process_exists('notepad.exe')):
			input()
		print('# Finished editing. Notepad.exe has been closed.')
	

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dir', default='logs', 
    help="relative path to save current day txt file")
args = parser.parse_args()

if not os.path.isdir(args.dir):
	os.makedirs(args.dir)
log_path = os.path.join(args.dir, datetime.date.today().strftime("%m-%d-%Y")+'.txt')
cur_day =  datetime.date.today().strftime("%d")


m_time_logger = time_logger(log_path)
print('# Starting time logger. Type help to see commands.')
while(cur_day == datetime.date.today().strftime("%d")):
	try:
		input_str = input("") 
		m_time_logger.read_input(input_str)
	except KeyboardInterrupt:
		m_time_logger.stop()
		sys.exit()

print('# It is a new day now. Re-run script to generate new txt file.')










