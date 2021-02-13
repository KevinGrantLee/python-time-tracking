import os
import csv
import datetime

import pandas as pd


def col_index_2_time_range(c):
    """

    needs new name
    """
    spacing = 30    # in minutes
    lst = [str(i*datetime.timedelta(minutes=spacing)) for i in range(24*60//spacing)]
    lst = [i[:-3] for  i in lst]
    lst.append('23:59')
    return (lst[c-1], lst[c])

def temp_name(topic_str):
    """

    needs new name
    """
    if topic_str == 'HRL':
       output = 'job'
    elif topic_str == 'VMG':
       output = 'research'
    elif topic_str == 'Nonproj':
       output = 'nonproj'
    elif topic_str in ['ECE 241A', 'ECE 239AS', 'ECE 205A']:
       output = 'class'
    else:
        output = ''
    return output

def convert_date_string(date_str, year='2020'):
    month_dict = {
        'Jan': '01',
        'Feb': '02',
        'Mar': '03',
        'Apr': '04',
        'May': '05',
        'Jun': '06',
        'Jul': '07',
        'Aug': '08',
        'Sep': '09',
        'Oct': '10',
        'Nov': '11',
        'Dec': '12',
        }
    d, m = date_str.split('-')
    d = d.zfill(2)
    m = month_dict[m]
    return '-'.join([m,d,year])

if __name__ == '__main__':

    save_dir = '2020_logs'
    csv_file =  'fall2020 - time.csv'
    df = pd.read_csv(csv_file)

    rows, cols = df.shape
    for r in range(rows):
        date_str = convert_date_string(df.iloc[r,0])
        filename = os.path.join(save_dir, date_str+'.txt')
        with open(filename, 'w') as fp:
            for c in range(1,cols):       
                topic = temp_name(df.iloc[r,c])
                if topic != '':
                    start_time, stop_time = col_index_2_time_range(c)
                    fp.write(f'@{topic}:\n')
                    fp.write(f'-  start:  {start_time}\n')
                    fp.write(f'-  stop:   {stop_time}\n\n')