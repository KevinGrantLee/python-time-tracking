import sys
import os
import datetime
import argparse
import subprocess
import configparser
from argparse import Namespace
from stat import S_IREAD, S_IROTH, S_IWUSR

import PySimpleGUI as sg
import time_logger as tl


def reset_config(config):
    config['DEFAULT'] = {
        'Topics': 'ml, nonproj, japanese',
        'LogFolder': r'C:\Users\kevin\OneDrive\Documents\GitHub\python-time-tracking\2021_logs',
        'StartDate': '2021-01-01',
        }
    with open('settings.ini', 'w') as configfile:
        config.write(configfile)

# def read_config():
#     config = configparser.ConfigParser()
#     config.read('settings.ini')

#     try:
#         LogFolder = config['DEFAULT']['LogFolder']
#     except:
#         LogFolder = ''
#     try:
#         Topics = config['DEFAULT']['Topics'].split(', ')
#     except: 
#         Topics = []
#     try:
#         StartDate = config['DEFAULT']['StartDate']
#     except:
#         StartDate = ''

def set_window_status(window, start):
    """
   start == True means we have just clicked "START"
   start == False means we have just clicked "STOP"
    """
    window['-FOLDER-'](disabled=not start)
    window['-BROWSE-'](disabled=not start)
    window['-ADDNOTE-'](disabled=start)
    window['-START-'](disabled=not start)
    window['-STOP-'](disabled=start)

    if start:
        window['-TOPIC-']('', disabled=not start)
        window['-STARTH-']('', disabled=not start)
        window['-STARTM-']('', disabled=not start)
        window['-STOPH-']('', disabled=start)
        window['-STOPM-']('', disabled=start)
    else:
        window['-TOPIC-'](disabled=not start)
        window['-STARTH-'](disabled=not start)
        window['-STARTM-'](disabled=not start)
        window['-STOPH-'](disabled=start)
        window['-STOPM-'](disabled=start)

config = configparser.ConfigParser()
config.read('settings.ini')

try:
    LogFolder = config['DEFAULT']['LogFolder']
except:
    LogFolder = ''
try:
    Topics = config['DEFAULT']['Topics'].split(', ')
except: 
    Topics = []
try:
    StartDate = config['DEFAULT']['StartDate']
except:
    StartDate = ''

sg.theme('DarkAmber')   # Add a touch of color

file_list_column = [
    [
        sg.Text("Log folder"),
        sg.In(LogFolder, \
            size=(40, 1), enable_events=True, key="-FOLDER-"),
        sg.FolderBrowse(key='-BROWSE-'), 
    ],
]


hours_list = [f'{i}'.zfill(2) for i in range(24)]
minutes_list = [str(i).zfill(2) for i in range(60)]
hours_list.insert(0, '')
minutes_list.insert(0, '')

layout = [  
            [
                sg.Column(file_list_column), 
                sg.Text('Start date'), sg.Input(StartDate, key='-IN-', size=(10,1)), 
                sg.CalendarButton('Open calendar', close_when_date_chosen=True,  target='-IN-', location=(0,0), no_titlebar=False, ),
            ],  
            [
                sg.Text('Topic'), 
                    sg.Combo(Topics, size=(10, 1), key='-TOPIC-'), 
                sg.Text('Start'), 
                    sg.Combo(hours_list, size=(3, 1), key='-STARTH-'), 
                    sg.Text(':'), 
                    sg.Combo(minutes_list, size=(3, 1), key='-STARTM-'), 
                sg.Text('Stop'), 
                    sg.Combo(hours_list, size=(3, 1), disabled=True, key='-STOPH-'), 
                    sg.Text(':'), 
                    sg.Combo(minutes_list, size=(3, 1), disabled=True, key='-STOPM-'), 
            ],
            [
                sg.Text('Note'), sg.InputText(key='-NOTE-'), 
                sg.Button('Add note', key='-ADDNOTE-', disabled=True),
                sg.Button('START', key='-START-'),
                sg.Button('STOP', disabled=True, key='-STOP-'),
            ],
            [
                sg.Button('Update figures', key='-UPDATE-'), 
                sg.Button('Open log', key='-LOG-'),
                sg.Button('Reset config', key='-CONFIG-'),
                sg.Button('Help', key='-HELP-'),
            ],
        ]


# initialzie time logger
run_date = datetime.date.today().strftime("%m-%d-%Y")
m_time_logger = tl.Time_Logger('')    


# Create the Window
window = sg.Window(f'Time Logger : {run_date}', layout, icon='icon.ico', resizable=False)

# Event Loop to process "events" and get the "values" of the inputs
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED: # if user closes window or clicks cancel
        m_time_logger.stop()
        break

    cur_time = datetime.datetime.now().strftime("%H:%M")
    if run_date != datetime.date.today().strftime("%m-%d-%Y"):
        sg.Popup('It is a new day now. Application will exit.', title='Info', keep_on_top=True)
        m_time_logger.stop(time='23:59')
        break

    os.makedirs('figures', exist_ok=True)
    args = Namespace(dir=values['-FOLDER-'], figdir='figures', startdate=values['-IN-'].split(' ')[0])
    log_path = os.path.join(args.dir, run_date+'.txt')
    m_time_logger.log_path = log_path


    if event == '-START-':
        if values['-FOLDER-'] == '':
            sg.Popup('Please choose a valid folder.', title='Error', keep_on_top=True)
            continue
        elif values['-TOPIC-'] == '':
            sg.Popup('Please enter a valid topic.', title='Error', keep_on_top=True)
            continue

        topic = values['-TOPIC-']
        if values['-STARTH-'] == '' and values['-STARTM-'] == '':
            start_time = cur_time

            start_h, start_m = start_time.split(':')
            window['-STARTH-'](start_h)
            window['-STARTM-'](start_m)
        elif values['-STARTH-'] != '' and values['-STARTM-'] != '':
            start_time = ':'.join([values['-STARTH-'], values['-STARTM-']])
        else:
            sg.Popup('Please choose a valid start time.', title='Error', keep_on_top=True)
            continue

        m_time_logger.start(topic, time=start_time)
        set_window_status(window, False)

    elif event == '-STOP-':
        if values['-STOPH-'] == '' and values['-STOPM-'] == '':
            stop_time = cur_time
        elif values['-STOPH-'] != '' and values['-STOPM-'] != '':
            stop_time = ':'.join([values['-STOPH-'], values['-STOPM-']])
        else:
            sg.Popup('Please enter a valid stop time.', title='Error', keep_on_top=True)
            continue
        if not tl.check_times(start_time, stop_time):
            sg.Popup('Stop time is before start time.', title='Error', keep_on_top=True)
            continue

        m_time_logger.stop(time=stop_time)
        set_window_status(window, True)


    elif event == '-ADDNOTE-':
        if values['-NOTE-'] == '':
            sg.Popup('Please enter a nonempty note.', title='Error', keep_on_top=True)
            continue
        m_time_logger.note(values['-NOTE-'])
        window['-NOTE-']('')


    elif event == '-UPDATE-':
        # check if year is valid
        if not args.startdate.isdigit():
            pass
        elif values['-FOLDER-'] == '':
            sg.Popup('Please choose a valid folder.', title='Error', keep_on_top=True)
            continue
        tl.make_figures(args)
        # have a drop down figure to open figures?

    elif event == '-LOG-':
        if os.path.isfile(m_time_logger.log_path):
            os.chmod(m_time_logger.log_path, S_IREAD|S_IROTH) # disable file writing
            os.startfile(m_time_logger.log_path)
            print('[INFO]: Close notepad.exe to continue.')
            while(tl.process_exists('notepad.exe')):
                pass
            os.chmod(m_time_logger.log_path, S_IWUSR|S_IREAD) # enable file writing 
            print('[INFO]: Resuming')


    elif event == '-CONFIG-':
        reset_config(config)
        print('[INFO]: Restart the program to update the fields.')

    elif event == '-HELP-':
        sg.Popup('This program tracks your time with plain text files.'\
            '\n\nStart by choosing a folder to save the log file in "Log folder". '\
            '\n\nThen, type in your topic in the "Topic" field.'\
            'You can select a start time, or leave it blank. If you leave it blank, it defaults to '\
            'the current time. Likewise for the stop time. '\
            'After you have selected the time, click the "START" or "STOP" buttons.'\
            '\n\nTo generate some summary figures, first choose a start date in the "Start date" field, then click "Update figures".'\
            '\n\n You can view the current log file with "Open log" in read-only mode.'\
            '\n\nThe config file "settings.ini" has defaults for the "Log folder", "Start date", and "Topic" fields. '\
            'You can manually edit the fields in "settings.ini". '\
            'Press "Reset config" to factory-reset "settings.ini".', \
            title='Help')


window.close()