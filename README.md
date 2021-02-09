# python-time-tracking

This is a simple plain text time tracking tool. 

```main.py``` creates a txt file for each day and ```generate_figures.py``` creates and saves figures.

## Usage
```
python main.py [-d dir]
```

Once the script is running, there are 4 commands:

To start a new topic: ``` start {TOPIC} ```

To add a note on current topic: ``` note {NOTE} ```

To stop current topic: ``` stop ```

To open log file in notepad: ``` open ```


Here is an example of an entry in the txt file:

```
@ml:
-  start:  15:02
-  notes:
-  -  studying semi-supervised learning
-  stop:   15:08
```

## Making figures
After you have created a couple of log files, you can run ```generate_figures.py``` to generate figures. ```generate_figures.py``` parses all log files (in the selected folder) and summarizes the content in figures.
```
python generate_figures.py [-d dir]
```
Currently the code generates a histogram of daily time per topic, a pie chart of total time per topic, a weekly histogram, and a calendar heatmap (looks like the github contribution graph). 

