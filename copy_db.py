import sqlite3
import pandas as pd
from sqlite3 import Error
import os

def Print(statement):
    from time import localtime, strftime
    print("{} - {}".format(strftime("%H:%M:%S", localtime()),statement))

new_db = 'rasmus_classification_muon_1500k.db'
new_path = r'C:\Users\jv97\Desktop\github\Neutrino-Machine-Learning\raw_data'

old_db = 'rasmus_classification_muon_3neutrino_3mio.db'
old_path = r'C:\Users\jv97\Desktop\github\Neutrino-Machine-Learning\raw_data'

event_nos_query = "SELECT event_no FROM truth WHERE pid IN (-13,13)"

old_con_path = 'file:' + os.path.join(old_path,old_db) + '?mode=ro'
new_con_path = os.path.join(new_path,new_db)

with sqlite3.connect(old_con_path,uri=True) as old_db:
    
    old_cursor = old_db.cursor()
    
    Print("Selecting all from SQLite_master..")
    old_cursor.execute("select * from SQLite_master")
    
    Print("Calling fetchall..")
    tables = old_cursor.fetchall()
    
    with sqlite3.connect(new_con_path) as new_db:
        new_cursor = new_db.cursor()
        
        for table in tables:
            
            SQL_statement = table[4]
            
            Print("Creating table: {}..".format(table[1]))
            new_cursor.execute(SQL_statement)
    
    Print("Getting event_nos for events to be copied..")
    event_nos = pd.read_sql(event_nos_query,old_db)
    event_nos = event_nos.values.reshape(-1)
    
    old_cursor.execute("ATTACH DATABASE ? AS new_db", (new_con_path,))
    
    Print("Copying features..")
    query = f"INSERT INTO new_db.features SELECT * FROM features WHERE event_no IN {tuple(event_nos)}"
    old_cursor.execute(query)
    
    Print("Copying truths..")
    query = f"INSERT INTO new_db.truth SELECT * FROM truth WHERE event_no IN {tuple(event_nos)}"
    old_cursor.execute(query)
    old_db.commit()
    
Print("DONE!")