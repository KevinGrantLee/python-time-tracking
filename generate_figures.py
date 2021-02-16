"""Generates figures

This script generates figures from the .txt format used by main.py.

Figures are generated in the current directory.

Be sure to set the correct directory of the log files and the year of the earliest log.
"""


import os 
import argparse
import datetime

import math

import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import calplot


def convert2date(date):
	"""converts m-d-y string to datetime.date object"""
	m,d,y = date.split('-')
	return datetime.date(int(y), int(m), int(d))


def time_elapsed(start, end):
	"""Returns time difference as hours with decimals

	start and end times are given in {HOUR}:{MIN} format as strings.
	"""
	def convert_time(time):
		hour, minutes = time.split(':')
		return int(hour)+int(minutes)/60

	start_time = convert_time(start)
	end_time = convert_time(end)
	assert end_time >= start_time, 'Error: end time is earlier than start time'
	return end_time - start_time


def days_elapsed(start, end):
	"""Return days elapsed between start and end

	Start and end dates are given in %m-%d-%y format.
	"""
	s_m, s_d, s_y = start.split('-')
	e_m, e_d, e_y = end.split('-')
	start_date = datetime.date(int(s_y), int(s_m), int(s_d))
	end_date = datetime.date(int(e_y), int(e_m), int(e_d))
	delta = end_date - start_date
	return delta.days




def run(args):

	time_per_topics = {}
	time_per_day_per_topics = {}

	week_labels = []
	biweek_labels = []
	start_date = convert2date('1-1-'+args.year)
	end_date = convert2date(os.listdir(args.dir)[-1].replace('.txt','')) # gets the most recent txt file in directory
	delta = datetime.timedelta(days=1)


	cnt = 0
	minutes_log = np.zeros(24*60)
	while start_date <= end_date:
		date_str = start_date.strftime("%m-%d-%Y")
		if cnt % 7 == 0:
			week_labels.append(date_str)
		if cnt % 14 == 0:
			biweek_labels.append(date_str)
		time_per_day_per_topics.setdefault(date_str, {}) # we assign missing days a total time of 0 hours
		start_date += delta
		cnt += 1

	## insert elements into biweek_labels
	ind = np.arange(1, 2*len(biweek_labels)-1, 2)
	for i in ind:
		biweek_labels.insert(i, '')
	# print(biweek_labels)


	### parsing txt files
	for mfile in os.listdir(args.dir):

		day_key = mfile.replace('.txt','')
		cur_day_data = time_per_day_per_topics[day_key]

		num_starts = 0
		num_stops = 0
		start_time = ''
		stop_time = ''
		last_topic = ''
		with open(os.path.join(args.dir, mfile)) as fp:
			lines = fp.readlines()
			for line in lines:
				if line[0] == '@': # '@" is marker char for topic line
					topic = line[1:].split(':')[0].lower().replace(' ','') # lower case, no spaces
					last_topic = topic
					time_per_topics.setdefault(last_topic, 0.0)
					cur_day_data.setdefault(last_topic, 0.0)
				elif '-  start:' in line:
					start_time = line.split(' ')[-1]
					num_starts += 1
					assert num_starts == num_stops + 1 # check that every 'stop' has an associated 'start'
				elif '-  stop:' in line:
					stop_time = line.split(' ')[-1]
					elapsed_time = time_elapsed(start_time, stop_time)
					time_per_topics[last_topic] += elapsed_time
					cur_day_data[last_topic] += elapsed_time
					num_stops += 1
					assert num_stops == num_starts

					hour, minutes = start_time.split(':')
					start_minute = 60*int(hour)+int(minutes)
					hour, minutes = stop_time.split(':')
					end_minute = 60*int(hour)+int(minutes)
					minutes_log[start_minute:end_minute] += np.ones(end_minute-start_minute) 

				# ignoring 'notes' lines

	time_per_topics = {mkey: round(time_per_topics[mkey], 3) for mkey in time_per_topics.keys()}
	time_per_day = {mkey: round(sum(time_per_day_per_topics[mkey].values()),3) \
		for mkey in time_per_day_per_topics.keys()}
	print(f'Topics are {list(time_per_topics.keys())}')
		

	### constructing separate daily counts for each topic
	print('TODO: verify separated_time_per_day is accurate. im like 80% sure')
	separated_time_per_day = {}
	for topic in time_per_topics.keys():
		separated_time_per_day.setdefault(topic, np.zeros(len(time_per_day_per_topics.keys())))
		for idx, (_, val1) in enumerate(time_per_day_per_topics.items()):
			for (mtopic, val2) in val1.items():
				if topic == mtopic:
					separated_time_per_day[topic][idx] += val2

	X = np.zeros(365)
	for cnt, (key, val) in enumerate(time_per_day.items()):
		# convert key to index
		idx = days_elapsed('1-1-'+args.year, key)
		X[idx] = val
		last_idx = idx

	X = np.resize(X, last_idx+1) # only log up to the current day
	avg_X_arr = X.mean()*np.ones(len(X))


	### scatter plot of daily time
	plt.figure()
	plt.plot(X, '.', label='Daily Time')
	# plt.bar(np.arange(len(X)), X, label='Daily Time')
	plt.plot(avg_X_arr, label=f'Average Time = {round(avg_X_arr[0],2)}')
	plt.xlabel('Days')
	plt.ylabel('Hours')
	plt.xticks(np.arange(0, len(X), 7), biweek_labels, rotation=70)
	plt.tight_layout() # make space for rotated xtick labels
	plt.legend()
	plt.savefig('Daily time.png')


	### pie chart of time per topic
	def make_autopct(values):
	    def my_autopct(pct):
	        total = sum(values)
	        # val = int(round(pct*total/100.0))
	        val = round(pct*total/100.0, 1)
	        # return '{p:.2f}%  ({v})'.format(p=pct,v=val) # both % and values
	        return f'{val}'
	    return my_autopct

	values = time_per_topics.values()
	plt.figure()
	plt.pie(values, labels=time_per_topics.keys(), autopct=make_autopct(values))
	plt.savefig('Pie chart.png')


	### histogram of total daily time, stacked
	plt.figure()
	plt.hist(separated_time_per_day.values(), bins=10, density=True, label=separated_time_per_day.keys(), stacked=True)

	xmin, xmax = plt.xlim()
	x = np.linspace(xmin, xmax, 100)
	mu, std = norm.fit(X)
	p = norm.pdf(x, mu, std)
	plt.plot(x, p, 'k', linewidth=2, label=f'N{round(mu,2), round(std**2,2)}')

	plt.xlabel('Hours')
	plt.ylabel('Frequency')
	plt.legend()
	plt.title('daily time histogram')
	plt.savefig('Daily time histogram.png')
	print('TODO: something wrong with daily time histogram - density sum >= 1 for some hours. is this ok for stacked?')


	### weekly time, separated by topic
	ind = np.arange(math.ceil(len(time_per_day.keys())/7))
	bottom_sum = np.zeros(len(ind))

	plt.figure()
	for cnt, (mtopic, val) in enumerate(separated_time_per_day.items()):
		every_fourteen_days_sum = np.add.reduceat(val, np.arange(0, len(val), 7))
		if cnt ==  0:
			plt.bar(ind, every_fourteen_days_sum, label=mtopic)
		else:
			plt.bar(ind, every_fourteen_days_sum, label=mtopic, bottom=bottom_sum)
		bottom_sum += every_fourteen_days_sum
	plt.xticks(ind, biweek_labels, rotation=70)
	plt.xlabel('Weeks')
	plt.ylabel('Hours')
	plt.legend()
	plt.tight_layout() # make space for rotated xtick labels
	plt.savefig('weekly time stacked.png')


	### calendar heatmap
	# https://pythonawesome.com/calendar-heatmaps-from-pandas-time-series-data/
	all_days = pd.date_range('1/1/'+args.year, periods=days_elapsed('1-1-'+args.year, end_date.strftime("%m-%d-%Y"))+1)
	events = pd.Series(X, index=all_days)
	plt.figure()
	calplot.calplot(events, dropzero=True, cmap='YlGn')
	plt.savefig('calendar heatmap.png')


	### clock figure showing what times i usually work
	N = 24
	bottom = 60

	# width of each bin on the plot
	width = (2*np.pi) / N

	# create theta for 24 hours
	theta = np.arange(width/2, 2*np.pi, width)

	# create the data
	hour_log = minutes_log.reshape((24,60)).sum(axis=1)

	# make a polar plot
	plt.figure(figsize = (12, 8))
	ax = plt.subplot(111, polar=True)
	bars = ax.bar(theta, hour_log, width=width, bottom=bottom)

	# set the lable go clockwise and start from the top
	ax.set_theta_zero_location("N")

	# clockwise
	ax.set_theta_direction(-1)

	# set labels
	ticks = [f'{hour}:00' for hour in range(24)]
	ax.set_xticks(np.linspace(0, 2*np.pi, 24, endpoint=False))
	ax.set_xticklabels(ticks)
	ax.set_yticklabels([])

	plt.savefig('productive time clock figure.png')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--dir', default='2021_logs', 
	    help="relative path to save current day txt file")
	parser.add_argument('-y', '--year', default='2021', 
	    help='Year of data')
	args = parser.parse_args()
	run(args)