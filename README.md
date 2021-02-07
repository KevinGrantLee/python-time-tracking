# python-time-tracking

This is a simple plain text time tracking tool. main.py creates a txt file for each day and generate_figures.py can be used to save figures. 

## Usage
''' python main.py '''

Once the script is running, there are 4 commands:

To start a new topic: ''' start {TOPIC} '''
To add a note on current topic: ''' note {NOTE} '''
To stop current topic: ''' stop '''
To open log file in notepad: ''' open '''

## Making figures
After you have created a couple of log files, you can run generate_figures.py to generate figures.
''' python generate_figures.py '''
Currently the code generates a histogram of daily time per topic, a pie chart of total time per topic, and a weekly time histogram. 

