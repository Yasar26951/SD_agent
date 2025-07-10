import os

import psycopg2 as pg2

from dotenv import load_dotenv
load_dotenv()
class database:
    def __init__(self):

        self.con=pg2.connect(database=os.getenv("database"),user='postgres',password=os.getenv("password"))
        self.cur=self.con.cursor()
    def store(self,lis):
        lis=[lis[0]]+[int(i) for i in lis[1:-1]]+[lis[-1]]
        lis=tuple(lis)
        print(lis)
        self.cur.execute(f"insert into  buffer (x,comment,email,message,tweet,spam) values {lis}")
        self.con.commit()
        self.cur.execute(f"select * from buffer")
        return self.cur.fetchall()
    def show(self):
        self.cur.execute(f"select  DISTINCT * from buffer")
        return self.cur.fetchall()
    def coppy(self):
        self.cur.execute("INSERT INTO histroy (x, comment, email, message, tweet, spam)SELECT DISTINCT x, comment, email, message, tweet, spam FROM buffer;")
        self.con.commit()
        print("transfer")

    def trun(self):
        self.cur.execute("truncate table buffer")
        self.con.commit()
        print("truncate")







