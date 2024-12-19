import sqlite3
from datetime import datetime
import pandas as pd

class DBHandler:
    
    def __init__(self,ruta_bd):
        self.con = sqlite3.connect(ruta_bd)
        self.cursor = self.con.cursor()
        self.db_creation()
        
    def experiment_insert(self, dl_mcs, ul_mcs, dl_rb, ul_rb, 
                        iperf_duration, iperf_mode, iperf_transport, iperf_type, description,name):
        
        start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        query = """
        INSERT INTO experiment (
            start_time, experiment_description, name_experiment,mcs_downlink, mcs_uplink, end_rb_downlink, end_rb_uplink, 
            iperf_duration, iperf_mode, iperf_transport, iperf_type
        )
        VALUES (?,?,?,?, ?, ?, ?, ?, ?, ?);
        """
        values = (start_time, description,name, dl_mcs, ul_mcs, dl_rb, ul_rb, iperf_duration, iperf_mode, iperf_transport, iperf_type)
        self.cursor.execute(query, values)
        return self.cursor.lastrowid
    
    def add_experiment_end_time(self, id):
        end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        query = """
            UPDATE experiment
            SET end_time = ?
            WHERE id = ?;
            """
        self.cursor.execute(query, (end_time, id))
        self.con.commit()
    
    def result_insert(self,dl_rate,uplink_rate, snr, cqi, timestamp,experiment_id):
        query = f"""
        INSERT INTO results (downlink_rate, uplink_rate, snr, cqi, timestamp,experiment_id)
        VALUES (?, ?, ?, ?, ?,{experiment_id});
        """
        values = list(zip(dl_rate, uplink_rate, snr, cqi, timestamp))
        self.cursor.executemany(query,values)
        self.con.commit()
    
    def result_collect(self, experiment_id):
        query = """SELECT * FROM results WHERE experiment_id = ?;"""
        self.cursor.execute(query, (experiment_id,))
        
        columns = [col[0] for col in self.cursor.description]
        
        results = [dict(zip(columns, row)) for row in self.cursor.fetchall()]
        return results
    
    def get_all_experiments(self):
        query = "SELECT * FROM experiment;"
        self.cursor.execute(query)
        
        rows = self.cursor.fetchall()  
        columns = [description[0] for description in self.cursor.description]
        
        return pd.DataFrame(rows, columns=columns)
    
    def get_all_results(self):
        query = "SELECT * FROM results;"
        self.cursor.execute(query)
        
        rows = self.cursor.fetchall()  
        columns = [description[0] for description in self.cursor.description]
        
        return pd.DataFrame(rows, columns=columns)
    
    def experiment_collect(self, experiment_id):
        query = """
            SELECT *
            FROM experiment
            WHERE id = ?;"""
        self.cursor.execute(query, (experiment_id,))
        
        columns = [col[0] for col in self.cursor.description]
        
        results = [dict(zip(columns, row)) for row in self.cursor.fetchall()]
        return results
    
    def avg_dl_throughput(self, id):
        query = f"""
            SELECT AVG(downlink_rate) AS average_downlink_rate
            FROM results
            WHERE experiment_id = {id};
        """
        self.cursor.execute(query)
        result = self.cursor.fetchone() 
        if result and result[0] is not None:
            dl_rate = result[0]
        else:
            dl_rate = 0 
        
        return dl_rate
    
    def avg_ul_throughput(self, id):
        query = f"""
            SELECT AVG(uplink_rate) AS average_uplink_rate
            FROM results
            WHERE experiment_id = {id};
        """
        self.cursor.execute(query)
        result = self.cursor.fetchone()  
        if result and result[0] is not None:
            ul_rate = result[0]
        else:
            ul_rate = 0 
        
        return ul_rate
    
    def delete_experiment(self, experiment_id):
        try:
            query_delete_resultado = "DELETE FROM resultado WHERE experiment_id = ?;"
            self.cursor.execute(query_delete_resultado, (experiment_id,))
            
            query_delete_experiment = "DELETE FROM experiment WHERE id = ?;"
            self.cursor.execute(query_delete_experiment, (experiment_id,))
            
            self.con.commit()
            return self.cursor.rowcount  
        except Exception as e:
            self.con.rollback() 
            print(f"Error deleting experiment: {e}")
            return 0
            
    def db_creation(self,db_path):
        db_schema = ''
        try:
            with open(db_path, "r") as file:
                db_schema = file.read()
        except Exception as e:
            return f'ERROR {e}'
        
        self.cursor.execute(db_schema)
        self.con.commit()