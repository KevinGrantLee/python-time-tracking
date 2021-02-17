# python-time-tracking

This is a simple plain text time tracking tool. 

```time_logger.py``` creates a txt file for each day and ```generate_figures.py``` creates and saves figures.

```gui_time_logger.py``` has a GUI created with PySimpleGUI.
Here is a picture of the GUI:


![alt text](https://github.com/KevinGrantLee/python-time-tracking/blob/main/gui.JPG?raw=true)


## Usage
```
python time_logger.py [-d dir]
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

Here is an example figure, showing at what times I am most productive:
![alt text](https://github.com/KevinGrantLee/python-time-tracking/blob/main/example_figure.png?raw=true)

