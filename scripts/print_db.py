from FunctionCollection import Print
from sqlite3 import connect
from os import path
from pandas import read_sql


# filename = r"dev_lvl7_mu_nu_e_classification_v003_unscaled.db"
filename = r"dev_level2_mu_tau_e_muongun_classification_wnoise_unscaled.db"

filepath = r"/groups/hep/pcs557/databases/dev_level2_mu_tau_e_muongun_classification_wnoise/data"

# query = "SELECT COUNT(*), pid FROM truth GROUP BY pid"

Print("Connecting to DB: {} ..".format(filename))
with connect(path.join(filepath,filename)) as con:
    cursor = con.cursor()

    # Execute custom query:
    try:
        Print("Trying to execute query: {} ..".format(query))
        df = read_sql(query,con)
        print(df)
    except:
        pass

    # Print the tables and indices present in the SQLite main database
    Print("Selecting all from SQLite_master..")
    cursor.execute("select * from SQLite_master")

    Print("Calling fetchall..")
    tables = cursor.fetchall()

    Print("Listing tables and indices from main database:")
    print("="*50)
    for table in tables:

            print("Type of database object: %s"%(table[0]))

            print("Name of the database object: %s"%(table[1]))

            print("Table Name: %s"%(table[2]))

            print("Root page: %s"%(table[3]))

            print("SQL statement: %s"%(table[4]))

            print("="*50)

    # con.close()