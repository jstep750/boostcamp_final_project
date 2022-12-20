import pandas as pd
import sqlite3


df = pd.read_csv('test3.csv', index_col=0)
df = df[['일자','제목','context']]
df['ticker'] = '000660'

conn = sqlite3.connect('test_database')
c = conn.cursor()

c.execute('CREATE TABLE IF NOT EXISTS newsdataset (date, title, contents)')
conn.commit()

df.to_sql('newsdataset', conn, if_exists='replace', index=False)

c.execute(''' SELECT * FROM newsdataset WHERE ticker='000660' ''')

for row in c.fetchall():
    print(row)

