import sqlite3

class DBHandler:
    
    def __init__(self,ruta_bd):
        self.con = sqlite3.connect(ruta_bd)
        self.cursor = self.con.cursor()
        self.db_creation()
        
    def experiment_insert(self, dl_mcs, ul_mcs, dl_rb, ul_rb, 
                        iperf_duration, iperf_mode, iperf_transport, iperf_type):
        query = """
        INSERT INTO experiment (
            mcs_downlink, mcs_uplink, end_rb_downlink, end_rb_uplink, 
            iperf_duration, iperf_mode, iperf_transport, iperf_type
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """
        values = (dl_mcs, ul_mcs, dl_rb, ul_rb, iperf_duration, iperf_mode, iperf_transport, iperf_type)
        self.cursor.execute(query, values)

    
    def result_insert(self,dl_rate,uplink_rate, snr, cqi, experiment_id):
        query = f"""
        INSERT INTO results (downlink_rate, uplink_rate, snr, cqi, experiment_id)
        VALUES (?, ?, ?, ?, {experiment_id});
        """
        values = list(zip(dl_rate, uplink_rate, snr, cqi))
        self.cursor.executemany(query,values)
        self.con.commit()
    
    def result_collect(self, experiment_id):
        query = """SELECT * FROM results WHERE experiment_id = ?;"""
        self.cursor.execute(query, (experiment_id,))
        
        columns = [col[0] for col in self.cursor.description]
        
        results = [dict(zip(columns, row)) for row in self.cursor.fetchall()]
        return results
    
    def experiment_collect(self, experiment_id):
        query = """
            SELECT mcs_downlink, mcs_uplink, end_rb_downlink, end_rb_uplink
            FROM experiment
            WHERE id = ?;"""
        self.cursor.execute(query, (experiment_id,))
        
        columns = [col[0] for col in self.cursor.description]
        
        results = [dict(zip(columns, row)) for row in self.cursor.fetchall()]
        return results
    
    def id_max_collect(self):
        query = "SELECT MAX(id) AS max_id FROM experiment;"
        self.cursor.execute(query)
        result = self.cursor.fetchone()  
        
        if result and result[0] is not None:
            max_id = result[0]
        else:
            max_id = 0  
        
        return max_id
    
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
    
    def db_creation(self,db_path):
        db_schema = ''
        try:
            with open(db_path, "r") as file:
                db_schema = file.read()
        except Exception as e:
            return f'ERROR {e}'
        
        self.cursor.execute(db_schema)
        self.con.commit()
