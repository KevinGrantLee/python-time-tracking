"""
calmap looks nicer than scatter plot. Maybe try this again later. 
"""

import matplotlib.pyplot as plt
import os 
import argparse
import datetime
import numpy as np
import pandas as pd
from scipy.stats import norm
import math
import calplot
# from calendarheatmap import calmap

def convert2date(date):
	# converts m-d-y string to datetime date object
	m,d,y = date.split('-')
	return datetime.date(int(y), int(m), int(d))


def time_elapsed(start, end):
	"""
	start and end are times given in {HOUR}:{MIN} format as strings
	returns time difference as hours with decimals
	"""
	def convert_time(time):
		hour, minutes = time.split(':')
		return int(hour)+int(minutes)/60

	start_time = convert_time(start)
	end_time = convert_time(end)

	assert end_time >= start_time, 'End time is earlier than start time'
	return end_time - start_time

def days_elapsed(start, end):
	"""
	start and end dates are given in %m-%d-%y format or %m/%d/%y
	"""
	s_m, s_d, s_y = start.split('-')
	e_m, e_d, e_y = end.split('-')
	start_date = datetime.date(int(s_y), int(s_m), int(s_d))
	end_date = datetime.date(int(e_y), int(e_m), int(e_d))
	delta = end_date - start_date
	return delta.days



parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dir', default='logs', 
    help="relative path to save current day txt file")
args = parser.parse_args()

# num_days = 0
time_per_topics = {}
time_per_day_per_topics = {}


### handling missing txt file days by assigning it 0
week_labels = []
start_date = convert2date('1-1-2021')
end_date = convert2date(os.listdir(args.dir)[-1].replace('.txt',''))
delta = datetime.timedelta(days=1)
cnt = 0
while start_date <= end_date:
    if cnt % 7 == 0:
    	week_labels.append(start_date.strftime("%m-%d-%Y"))

    time_per_day_per_topics.setdefault(start_date.strftime("%m-%d-%Y"), {})
    start_date += delta
    cnt += 1


### parsing txt files
for mfile in os.listdir(args.dir):
	# should check if is valid log file
	day_key = mfile.replace('.txt','')
	# time_per_day_per_topics.setdefault(day_key, {})

	num_starts = 0
	num_stops = 0
	start_time = ''
	stop_time = ''
	last_topic = ''
	with open(os.path.join(args.dir, mfile)) as fp:
		lines = fp.readlines()
		for line in lines:
			if line[0] == '@': # marker char for topic line
				mTopic = line[1:].split(':')[0].lower()
				
				last_topic = mTopic
				time_per_topics.setdefault(last_topic, 0)
				time_per_day_per_topics[day_key].setdefault(last_topic, 0.0)

			if '-  start:' in line:
				start_time = line.split(' ')[-1]

				num_starts += 1
				assert num_starts == num_stops + 1 # check that every 'stop' has an associated 'start'
			if '-  stop:' in line:
				stop_time = line.split(' ')[-1]
				elapsed_time = time_elapsed(start_time, stop_time)

				time_per_topics[last_topic] += elapsed_time
				time_per_day_per_topics[day_key][last_topic] += elapsed_time

				num_stops += 1
				assert num_stops == num_starts

				
time_per_topics = {mkey: round(time_per_topics[mkey], 3) for mkey in time_per_topics.keys()}
# time_per_day = {mkey: round(time_per_day[mkey], 1) for mkey in time_per_day.keys()}

time_per_day = {mkey: round(sum(time_per_day_per_topics[mkey].values()),3) \
	for mkey in time_per_day_per_topics.keys()}

### constructing separate daily counts for each topic
print('TODO: verify separated_time_per_day is accurate. im like 80% sure')
separated_time_per_day = {}
for Mtopic in time_per_topics.keys():
	separated_time_per_day.setdefault(Mtopic, np.zeros(len(time_per_day_per_topics.keys())))
	for idx, (mday, val1) in enumerate(time_per_day_per_topics.items()):
		for (mtopic, val2) in val1.items():
			if Mtopic == mtopic:
				separated_time_per_day[Mtopic][idx] += val2


print('TODO: code is harcoded to date from 1-1-2021')
X = np.zeros(365)

for cnt, (key, val) in enumerate(time_per_day.items()):
	# convert key to index
	idx = days_elapsed('1-1-2021', key)
	X[idx] = val
	last_idx = idx

X = np.resize(X, last_idx+1) # only log up to the current day
avg_X = X.mean()*np.ones(len(X))



# calendar heatmap - ignore for now, need to add colorbar
# how to set ax here?
# calmap(ax, 2021, X.reshape(53,7).T)



### scatter plot of daily time
plt.figure()
plt.plot(X, '.', label='Daily Time')
# plt.bar(np.arange(len(X)), X, label='Daily Time')
plt.plot(avg_X, label=f'Average Time = {round(avg_X[0],2)}')
plt.xlabel('Days')
plt.ylabel('Hours')
plt.title('Daily time')
plt.xticks(np.arange(0, len(X), 7), week_labels, rotation=70)
plt.tight_layout() # makes space for rotated xtick labels
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
mu, std = norm.fit(X)

plt.figure()
plt.hist(separated_time_per_day.values(), bins=10, density=True, label=separated_time_per_day.keys(), stacked=True)

xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2, label=f'N{round(mu,2), round(std**2,2)}')

plt.xlabel('Hours')
plt.ylabel('Frequency')
plt.legend()
plt.title('daily time histogram')
plt.savefig('Daily time histogram.png')
print('TODO: something wrong with daily time histogram')


### weekly time, separated by topic. 
ind = np.arange(math.ceil(len(time_per_day.keys())/7))
bottom_sum = np.zeros(len(ind))
plt.figure()

for cnt, (mtopic, val) in enumerate(separated_time_per_day.items()):
	aaa = np.add.reduceat(val, np.arange(0, len(val), 7))
	if cnt ==  0:
		plt.bar(ind, aaa, label=mtopic)
	else:
		plt.bar(ind, aaa, label=mtopic, bottom=bottom_sum)
	bottom_sum += aaa
plt.xticks(ind, week_labels, rotation=70)
plt.xlabel('Weeks')
plt.ylabel('Hours')
plt.legend()
plt.tight_layout() # makes space for rotated xtick labels
plt.savefig('weekly time stacked.png')



### calendar heatmap
# https://pythonawesome.com/calendar-heatmaps-from-pandas-time-series-data/
all_days = pd.date_range('1/1/2021', periods=days_elapsed('1-1-2021', end_date.strftime("%m-%d-%Y"))+1)
events = pd.Series(X, index=all_days)
plt.figure()
calplot.calplot(events, dropzero=True, cmap='YlGn')
plt.savefig('calendar heatmap.png')


# ### daily time hexbin
# df = pd.DataFrame( np.vstack((np.arange(len(X)), X)).T, columns=["a", "b"])
# plt.figure()
# df.plot.hexbin(x="a", y="b", gridsize=35)
# plt.savefig('Daily time hexbin.png')