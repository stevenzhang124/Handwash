import sqlite3

conn = sqlite3.connect('handwash.db')
print("Opened database successfully")
c = conn.cursor()
c.execute("DROP TABLE IF EXISTS HANDEMO")
print("Delete Table successfully")

c.execute('''CREATE TABLE IF NOT EXISTS HANDEMO
	   (PERSON TEXT     NOT NULL,
       CTIME           INT    NOT NULL,
       HLOC    TEXT ,
       PLOC    TEXT ,
       HAND            INT ,
       PATIENT        INT,
       JUDGE INT);''')
print("Table created successfully")
conn.commit()

conn.close()