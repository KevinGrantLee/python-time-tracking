# code to get used to pysimplegui

import sys
import os
import datetime
import argparse
import subprocess
from argparse import Namespace

import PySimpleGUI as sg

from generate_figures import run
import time_logger as tl


try:
    with open('topics.txt', 'r') as fp:
        topics = fp.readlines()
    topics = [topics[i].strip() for i in range(len(topics))]
except:
    topics = []


sg.theme('DarkAmber')   # Add a touch of color

file_list_column = [
    [
        sg.Text("Log folder"),
        sg.In('C:/Users/kevin/OneDrive/Documents/GitHub/python-time-tracking/2021_logs', \
            size=(40, 1), enable_events=True, key="-FOLDER-"),
        sg.FolderBrowse(key='-BROWSE-'), 
        # sg.Button('Select Folder', key='-SELECT-')
    ],
]


hours_list = [f'{i}'.zfill(2) for i in range(24)]
minutes_list = [str(i).zfill(2) for i in range(60)]
hours_list.insert(0, '')
minutes_list.insert(0, '')

# All the stuff inside your window.
layout = [  
            [
                sg.Column(file_list_column), sg.Text('Year'), sg.InputText('2021', size=(10, 1), key='-YEAR-')
            ],  
            [
                sg.Text('Topic'), 
                    sg.Combo(topics, size=(10, 1), key='-TOPIC-'), 
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
    cur_time = datetime.datetime.now().strftime("%H:%M")
    if run_date != datetime.date.today().strftime("%m-%d-%Y"):
        sg.Popup('It is a new day now. Application will exit.', keep_on_top=True)
        m_time_logger.stop()
        break

    event, values = window.read()
    # print('event=',event)
    if event == sg.WIN_CLOSED: # if user closes window or clicks cancel
        m_time_logger.stop()
        break


    # if values['-FOLDER-'] == '':
    #     sg.Popup('Please choose a folder to save logs in.', keep_on_top=True)
    #     continue

    os.makedirs('figures', exist_ok=True)
    args = Namespace(dir=values['-FOLDER-'], figdir='figures', year=values['-YEAR-'])
    log_path = os.path.join(args.dir, run_date+'.txt')
    m_time_logger.log_path = log_path


    if event == '-START-':
        if values['-FOLDER-'] == '':
            sg.Popup('Please choose a valid folder.', keep_on_top=True)
            continue
        elif values['-TOPIC-'] == '':
            sg.Popup('Please enter a valid topic.', keep_on_top=True)
            continue

        window['-TOPIC-'](disabled=True)
        window['-FOLDER-'](disabled=True)
        window['-BROWSE-'](disabled=True)
        window['-ADDNOTE-'](disabled=False)

        window['-START-'](disabled=True)
        window['-STARTH-'](disabled=True)
        window['-STARTM-'](disabled=True)

        window['-STOP-'](disabled=False)
        window['-STOPH-'](disabled=False)
        window['-STOPM-'](disabled=False)

        # processing data
        topic = values['-TOPIC-']
        if values['-STARTH-'] == '' or values['-STARTM-'] == '':
            start_time = cur_time
        else:
            start_time = ':'.join([values['-STARTH-'], values['-STARTM-']])
        m_time_logger.start(topic, time=start_time)
        

    elif event == '-STOP-':
        # print('stop', cur_time)
        window['-TOPIC-']('', disabled=False)
        window['-FOLDER-'](disabled=False)
        window['-BROWSE-'](disabled=False)
        window['-ADDNOTE-'](disabled=True)

        window['-START-'](disabled=False)
        window['-STARTH-']('', disabled=False)
        window['-STARTM-']('', disabled=False)

        window['-STOP-'](disabled=True)
        window['-STOPH-']('', disabled=True)
        window['-STOPM-']('', disabled=True)

        # processing data
        if values['-STOPH-'] == '' or values['-STOPM-'] == '':
            stop_time = cur_time
        else:
            stop_time = ':'.join([values['-STOPH-'], values['-STOPM-']])
        m_time_logger.stop(time=stop_time)

    elif event == '-ADDNOTE-':
        if values['-NOTE-'] == '':
            sg.Popup('Please enter a nonempty note.', keep_on_top=True)
            continue
        m_time_logger.note(values['-NOTE-'])
        window['-NOTE-']('')


    elif event == '-UPDATE-':
        # check if year is valid
        if not values['-YEAR-'].isdigit():
            sg.Popup('Please enter a valid year.', keep_on_top=True)
            continue
        elif values['-FOLDER-'] == '':
            sg.Popup('Please choose a valid folder.', keep_on_top=True)
            continue
        # run generate_figures.py
        tl.make_figures(args)
        # have a drop down figure to open figures?

    elif event == '-LOG-':
        if os.path.isfile(m_time_logger.log_path):
            os.startfile(m_time_logger.log_path)
            print('[INFO]: Close notepad.exe to continue.')
            while(tl.process_exists('notepad.exe')):
                pass
            print('[INFO]: Resuming...')


    elif event == '-HELP-':
        sg.Popup('This program tracks your time with plain text files.'\
            '\n\nStart by choosing a folder to save the log file in "Log folder". '\
            '\n\nThen, type in your topic in the "Topic" field.'\
            'You can select a start time, or leave it blank. If you leave it blank, it defaults to '\
            'the current time. Likewise for the stop time. '\
            'After you have selected the time, click the "START" or "STOP" buttons.'\
            '\n\nYou can also click the "Update figures" button to generate some summary figures.')


window.close()