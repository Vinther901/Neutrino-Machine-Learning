import sqlite3
import pandas as pd

db_file_list =  ['C:\\Users\\jv97\\Desktop\\github\\Neutrino-Machine-Learning\\raw_data\\120000_00.db',
            'C:\\Users\\jv97\\Desktop\\github\\Neutrino-Machine-Learning\\raw_data\\140000_00.db',
            'C:\\Users\\jv97\\Desktop\\github\\Neutrino-Machine-Learning\\raw_data\\160000_00.db']                    #

scalar = pd.DataFrame()                                                             #
sequential = pd.DataFrame()                                                         #
for db_file in db_file_list:                                                        #
    with sqlite3.connect(db_file) as con:                                           #
        query = 'select * from sequential'                                          # MERGES ALL .db FILES TO TWO .csv FILES:
        sequential = sequential.append(pd.read_sql(query, con))                     # scalar.csv , sequential.csv   
        query = 'select * from scalar'                                              # THESE ARE THEN WRITTEN TO DRIVE
        scalar = scalar.append(pd.read_sql(query, con))                             #
        cursor = con.cursor()                                                       #
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")

sequential.to_csv(r'C:\\Users\\jv97\\Desktop\\github\\Neutrino-Machine-Learning\\data\\sequential.csv')
scalar.to_csv(r'C:\\Users\\jv97\\Desktop\\github\\Neutrino-Machine-Learning\\data\\scalar.csv')