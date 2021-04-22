import sqlite3

conn = sqlite3.connect('handwash.db', isolation_level=None)
print("Opened database successfully")
c = conn.cursor()



c.execute("select * from HANDEMO")

hands = c.fetchall()
for hand in hands:
	print(hand)

c.execute("DROP TABLE HANDEMO")
conn.close()	