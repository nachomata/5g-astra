import sqlite3
from datetime import datetime
import pandas as pd
import os

class DBHandler:
    
    def __init__(self, ruta_bd):
        self.db_name = ruta_bd
        self.db_creation(os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.sql"))

    def get_conn(self):
        return sqlite3.connect(self.db_name, timeout=100, check_same_thread=False)
        
    def experiment_insert(self, dl_mcs, ul_mcs, dl_rb, ul_rb, 
                        iperf_duration, iperf_mode, iperf_transport, iperf_bitrate, description,name):
        
        start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        query = """
        INSERT INTO experiment (
            start_time, experiment_description, name_experiment,mcs_downlink, mcs_uplink, end_rb_downlink, end_rb_uplink, 
            iperf_duration, iperf_mode, iperf_transport, iperf_bitrate
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        values = (start_time, description,name, dl_mcs, ul_mcs, dl_rb, ul_rb, iperf_duration, iperf_mode, iperf_transport, iperf_bitrate)
        con = self.get_conn()
        cursor = con.cursor()
        cursor.execute(query, values)
        experiment_id = cursor.lastrowid
        con.commit()
        con.close()
        return experiment_id
    
    def add_experiment_end_time(self, id):
        end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        query = """
            UPDATE experiment
            SET end_time = ?
            WHERE id = ?;
            """
        con = self.get_conn()
        con.execute(query, (end_time, id))
        con.commit()
        con.close()
        
    def result_insert(self, dl_rate, uplink_rate, snr, cqi, experiment_id, timestamp):
        query = f"""
        INSERT INTO results (downlink_rate, uplink_rate, snr, cqi, experiment_id, timestamp)
        VALUES (?, ?, ?, ?, ?, ?);
        """
        values = list(zip(dl_rate, uplink_rate, snr, cqi, [experiment_id] * len(dl_rate), timestamp))       
        
        con = self.get_conn()
        cursor = con.cursor()
        cursor.executemany(query, values)
        
        con.commit()
        con.close()
    
    def result_collect(self, experiment_id):
        query = "SELECT * FROM results WHERE experiment_id = ?;"
        con = self.get_conn()
        cursor = con.cursor()
        cursor.execute(query,(experiment_id,))
            
        results = cursor.fetchall()
        con.close()
        
        return results
    
    def result_collect_ml(self, experiment_id):
        query = "SELECT * FROM results WHERE experiment_id = ?;"
        con = self.get_conn()
        cursor = con.cursor()
        cursor.execute(query, (experiment_id,))
            
        results = cursor.fetchall()
        con.close()
        
        # Inicializa listas para cada columna
        ids = []
        timestamps = []
        col3 = []
        col4 = []
        col5 = []
        col6 = []
        col7 = []
        
        # Itera por cada fila y extrae valores a las listas
        for row in results:
            ids.append(row[0])
            timestamps.append(row[1])
            col3.append(row[2])
            col4.append(row[3])
            col5.append(row[4])
            col6.append(row[5])
            col7.append(row[6])
        
        # Retorna un diccionario con las listas
        return {
            "ids": ids,
            "timestamps": timestamps,
            "dl_rate": col3,
            "ul_rate": col4,
            "snr": col5,
            "cqi": col6,
            "id_experiment": col7,
        }

    def get_all_experiments(self):
        query = "SELECT * FROM experiment;"
        con = self.get_conn()
        cursor = con.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [description[0] for description in cursor.description]
        con.close()
        results = [dict(zip(columns, row)) for row in rows]
        return results
    
    def get_all_results(self):
        query = "SELECT * FROM results;"
        con = self.get_conn()
        cursor = con.cursor()
        cursor.execute(query)
        
        rows = cursor.fetchall()  
        columns = [description[0] for description in cursor.description]
        
        con.close()
        return pd.DataFrame(rows, columns=columns)
    
    def experiment_collect(self, experiment_id):
        query = """
            SELECT *
            FROM experiment
            WHERE id = ?;"""
        con = self.get_conn()
        cursor = con.cursor()
        cursor.execute(query, (experiment_id,))
        
        columns = [col[0] for col in cursor.description]
        
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        con.close()
        return results
    
    def avg_dl_throughput(self, id):
        query = f"""
            SELECT AVG(downlink_rate) AS average_downlink_rate
            FROM results
            WHERE experiment_id = {id};
        """
        con = self.get_conn()
        cursor = con.cursor()
        cursor.execute(query)
        result = cursor.fetchone() 
        if result and result[0] is not None:
            dl_rate = result[0]
        else:
            dl_rate = 0 
        con.close()
        return dl_rate
    
    def avg_ul_throughput(self, id):
        query = f"""
            SELECT AVG(uplink_rate) AS average_uplink_rate
            FROM results
            WHERE experiment_id = {id};
        """
        con = self.get_conn()
        cursor = con.cursor()
        cursor.execute(query)
        result = cursor.fetchone()  
        if result and result[0] is not None:
            ul_rate = result[0]
        else:
            ul_rate = 0 
        con.close()
        return ul_rate
    
    def delete_experiment(self, experiment_id):
        try:
            query_delete_resultado = "DELETE FROM results WHERE experiment_id = ?;"
            con = self.get_conn()
            cursor = con.cursor()
            cursor.execute(query_delete_resultado, (experiment_id,))
            
            query_delete_experiment = "DELETE FROM experiment WHERE id = ?;"
            cursor.execute(query_delete_experiment, (experiment_id,))
            
            con.commit()
            con.close()
            return cursor.rowcount  
        except Exception as e:
            con.rollback() 
            con.close()
            print(f"Error deleting experiment: {e}")
            return 0
            
    def db_creation(self, schema_path):
        db_schema = ''
        try:
            with open(schema_path, "r") as file:
                db_schema = file.read()
        except Exception as e:
            return f'ERROR {e}'
        
        con = self.get_conn()
        cursor = con.cursor()
        cursor.executescript(db_schema)
        con.commit()
        con.close()