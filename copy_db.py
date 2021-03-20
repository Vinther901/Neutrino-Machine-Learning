import sqlite3
import pandas as pd
from sqlite3 import Error
import os

new_db = 'muon_500k_scaled2.db' #'oscNext_MC_2000k.db'

new_path = r'C:\Users\jv97\Desktop\github\Neutrino-Machine-Learning\raw_data'

old_db = 'rasmus_classification_muon_3neutrino_3mio.db' #rasmus_classification_muon_3neutrino_3mio.db #dev_level7_mu_e_tau_oscweight_000_unscaled.db

old_path = r'C:\Users\jv97\Desktop\github\Neutrino-Machine-Learning\raw_data' #\dev_level7_mu_e_tau_oscweight_000\data

con_path = 'file:' + os.path.join(old_path,old_db) + '?mode=ro'

event_nos_query = None#'SELECT event_no FROM truth WHERE pid in (-13,13)'
subsample = 5

event_nos = pd.read_pickle(r'C:\Users\jv97\Desktop\github\Neutrino-Machine-Learning\datasets\event_nos_500k_muon_set1.pkl').values.reshape(-1)

def create_connection(db_file):
    """ create a database connection to a SQLite database """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        
        query =""" CREATE TABLE IF NOT EXISTS features (
                        event_no INTEGER NOT NULL,
                        charge_log10 REAL NOT NULL,
                        time REAL NOT NULL,
                        dom_x REAL NOT NULL,
                        dom_y REAL NOT NULL,
                        dom_z REAL NOT NULL,
                        lc INTEGER,
                        pulse_width INTEGER,
                        SRTInIcePulses INTEGER
                    ) """
        
        cursor.execute(query)
        
        query = """ CREATE TABLE IF NOT EXISTS truth (
                        event_no INTEGER PRIMARY KEY NOT NULL,
                        energy_log10 REAL NOT NULL,
                        direction_x REAL NOT NULL,
                        direction_y REAL NOT NULL,
                        direction_z REAL NOT NULL,
                        azimuth REAL NOT NULL,
                        zenith REAL NOT NULL,
                        pid INTEGER NOT NULL,
                        stopped_muon INTEGER NOT NULL
                    ) """
        
        cursor.execute(query)
        
        query = "CREATE INDEX index_features_event_no ON features(event_no)"
        
        cursor.execute(query)
        
        conn.close()
    except Error as e:
        print(e)
    finally:
        if conn:
            conn.close()


create_connection(os.path.join(new_path,new_db))

with sqlite3.connect(con_path,uri=True) as old_db:
    cursor = old_db.cursor()
    
    if event_nos_query != None:
        event_nos = pd.read_sql(event_nos_query,old_db).sample(subsample).values.reshape(-1)

    query = "ATTACH DATABASE ? AS new_db"
    cursor.execute(query, (os.path.join(new_path,new_db),))
    
    query = f"INSERT INTO new_db.features SELECT event_no, charge_log10, time, dom_x, dom_y, dom_z, lc, pulse_width, SRTInIcePulses FROM features WHERE event_no in {tuple(event_nos)}"
    cursor.execute(query)
    
    query = f"INSERT INTO new_db.truth SELECT event_no, energy_log10, direction_x, direction_y, direction_z, azimuth, zenith, pid, stopped_muon FROM truth WHERE event_no in {tuple(event_nos)}"
    cursor.execute(query)
    old_db.commit()
old_db.close()