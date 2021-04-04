"""Time Tracker

This script is a simple plain-text time tracker. 

This tool generates .txt files in a directory of your choice.

The default directory can be set.
"""


import sys
import os
import datetime
import argparse
import subprocess

from generate_figures import run as make_figures


class Time_Logger:


    def __init__(self, log_path):
        self.topic = ''
        self.log_path = log_path
        self.last_action = ''

    # def set_log_path(self, log_path):
    #     self.log_path = log_path

    def read_input(self, input_str):
        # time = datetime.datetime.now().strftime("%H:%M")
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
            print('# To start a new topic:            start {TOPIC}')
            print('# To add a note on current topic:  note {NOTE}')
            print('# To stop current topic:           stop')
            print('# To open log file in notepad:     open\n')

    def start(self, topic, time=datetime.datetime.now().strftime("%H:%M")):
        if (not self.topic and 
                self.topic != '-  start:' and
                self.topic != '-  stop:' and 
                topic):
            print(f'# START: "{topic}": {time}')
            with open(self.log_path, 'a') as fp:
                fp.write(f'@{topic}:\n')
                fp.write(f'-  start:  {time}\n')
            self.topic = topic
            self.last_action = 'start'
        else:
            print(f"# Unable to add new topic {topic}. Current topic is '{self.topic}'")
            
    def stop(self, time=datetime.datetime.now().strftime("%H:%M")):
        if self.topic:
            print(f'# STOP: "{self.topic}": {time}')
            with open(self.log_path, 'a') as fp:
                fp.write(f'-  stop:   {time}\n\n')
            self.topic = ''
            self.last_action = 'stop'
        else: 
            print('# Unable to stop. There is no current topic.')

    def note(self, note_str):
        time = datetime.datetime.now().strftime("%H:%M")
        if self.topic:
            print(f'NOTE: "{note_str}": "{self.topic}": {time}')
            with open(self.log_path, 'a') as fp:
                if self.last_action != 'note':
                    fp.write('-  notes:\n')
                fp.write(f'-  -  {note_str}\n')
            self.last_action = 'note'
        else:
            print('# Unable to add note. There is no current topic.')
    
    def open_log(self):
        os.startfile(self.log_path)
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

def check_times(start_time, stop_time):
    """Returns True if stop_time >= start_time. Else, False

    start_time and stop_time are each string tuples of form '{HOURS}:{MINUTES}')
    """
    start_h, start_m = start_time.split(':')
    stop_h, stop_m = stop_time.split(':')

    if stop_h > start_h:
        return True
    elif stop_h == start_h and stop_m >= start_m:
        return True
    else:
        return False


def run(args):
    run_date = datetime.date.today().strftime("%m-%d-%Y")
    log_path = os.path.join(args.dir, run_date+'.txt')
    run_day = run_date.split('-')[1]


    ### generate figures
    print('# Generating figures')
    make_figures(args)
    print('# Done generating figures\n')

    ### check if format of log file is correct

    ### run time logger
    time_logger = Time_Logger(log_path)
    print('# Time logger is running. Type "help" to see commands.')
    while(run_day == datetime.date.today().strftime("%d")):
        try:
            input_str = input() 
            time_logger.read_input(input_str)
        except KeyboardInterrupt:
            time = datetime.datetime.now().strftime("%H:%M")
            time_logger.stop()
            sys.exit()

    time_logger.stop()
    print('# It is a new day now. Re-run script to generate new txt file.')


### main
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dir', default='2021_logs', 
    help='relative path to save current day txt file')
parser.add_argument('-f', '--figdir', default='figures', 
    help="folder to save figures in")
parser.add_argument('-s', '--startdate', default='2021-01-01', 
    help='Start date to make figures from')
args = parser.parse_args()
if not os.path.isdir(args.dir):
    os.makedirs(args.dir)
if args.figdir != '':
    if not os.path.isdir(args.figdir):
        os.makedirs(args.figdir)

if __name__=='__main__':
    run(args)