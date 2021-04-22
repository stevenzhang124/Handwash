import sqlite3


conn = sqlite3.connect('handwash.db', isolation_level=None)
print("Opened database successfully")
cur = conn.cursor()



tmp_time = 0
case_1 = 0
case_2 = 0
case_3 = 0
case_4 = 0

#@app.route("/data")
def getdata():
	global tmp_time
	
	if tmp_time > 0:
		sql_1 = "select * from HANDEMO where CTIME > %s and PERSON='Person_1" %(tmp_time)
		sql_2 = "select * from HANDEMO where CTIME > %s and PERSON='Person_2" %(tmp_time)
	else:
		sql_1 = "select * from HANDEMO where PERSON='Person_1'"
		sql_2 = "select * from HANDEMO where PERSON='Person_2'"

	cur.execute(sql_1)
	records_1 = cur.fetchall()
	cur.execute(sql_2)
	records_2 = cur.fetchall()

	if records_1[-1][2] > records_2[-1][2]:
		tmp_time = records_1[-1][2]
	else:
		tmp_time = records_2[-1][2]

	records_1_case = judge(records_1)
	records_2_case = judge(records_2)

	results = [records_1_case, records_2_case]

	return results 

	
def judge(records):
	global case_1
	global case_2
	global case_3
	global case_4

	if records[-1][-1] == 0 and records[-1][-3] == 1:
		case_1+=1
	if records[-1][-1] == 2 and records[-1][-3] == 1:
		case_2+=1
	if records[-1][-1] == 1 and records[-1][-3] == 1:
		case_3+=1
	if records[-1][-1] == 2 and records[-1][-3] == 0:
		case_4+=1

	case = [case_1, case_2, case_3, case_4]

	return case

result = getdata()
print(result)
