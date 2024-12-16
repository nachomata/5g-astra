import sqlite3

class DBHandler:
    
    TABLE_EXPERIMENTS = """
    CREATE TABLE IF NOT EXISTS experiment(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        mcs_downlink INTEGER NOT NULL,
        mcs_uplink INTEGER NOT NULL,
        end_rb_downlink INTEGER NOT NULL,
        end_rb_uplink INTEGER NOT NULL
        );
    """
    TABLE_RESULTS = """
    CREATE TABLE IF NOT EXISTS results(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        downlink_rate FLOAT,
        uplink_rate FLOAT,
        snr FLOAT,
        cqi INTEGER,
        experiment_id integer,
        FOREIGN KEY (experiment_id) REFERENCES experiment(id)
        );
    """
    
    def __init__(self,ruta_bd):
        self.con = sqlite3.connect(ruta_bd)
        self.cursor = self.con.cursor()
        self.db_creation()
        
    def experiment_insert(self,dl_mcs, ul_mcs, dl_rb,ul_rb):
        query = f"""
        INSERT INTO experiment (mcs_downlink, mcs_uplink, end_rb_downlink, end_rb_uplink)
        VALUES ({dl_mcs}, {ul_mcs}, {dl_rb}, {ul_rb});
        """
        self.cursor.execute(query)
    
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
        # Nombres de las columnas en la tabla
        columns = [col[0] for col in self.cursor.description]
        # Obtener todos los resultados y convertirlos a diccionarios
        results = [dict(zip(columns, row)) for row in self.cursor.fetchall()]
        return results
    
    def experiment_collect(self, experiment_id):
        query = """
            SELECT mcs_downlink, mcs_uplink, end_rb_downlink, end_rb_uplink
            FROM experiment
            WHERE id = ?;"""
        self.cursor.execute(query, (experiment_id,))
        # Nombres de las columnas en la tabla
        columns = [col[0] for col in self.cursor.description]
        # Obtener todos los resultados y convertirlos a diccionarios
        results = [dict(zip(columns, row)) for row in self.cursor.fetchall()]
        return results
    
    def id_max_collect(self):
        query = "SELECT MAX(id) AS max_id FROM experiment;"
        self.cursor.execute(query)
        result = self.cursor.fetchone()  # Recupera el primer resultado de la consulta
        
        if result and result[0] is not None:
            max_id = result[0]
        else:
            max_id = 0  # Si la tabla está vacía o no hay IDs
        
        return max_id
    
    def avg_dl_throughput(self, id):
        query = f"""
            SELECT AVG(downlink_rate) AS average_downlink_rate
            FROM results
            WHERE experiment_id = {id};
        """
        self.cursor.execute(query)
        result = self.cursor.fetchone()  # Recupera el primer resultado de la consulta
        if result and result[0] is not None:
            dl_rate = result[0]
        else:
            dl_rate = 0  # Si la tabla está vacía o no hay IDs
        
        return dl_rate
    
    def avg_ul_throughput(self, id):
        query = f"""
            SELECT AVG(uplink_rate) AS average_uplink_rate
            FROM results
            WHERE experiment_id = {id};
        """
        self.cursor.execute(query)
        result = self.cursor.fetchone()  # Recupera el primer resultado de la consulta
        if result and result[0] is not None:
            ul_rate = result[0]
        else:
            ul_rate = 0  # Si la tabla está vacía o no hay IDs
        
        return ul_rate
    
    def db_creation(self):
        self.cursor.execute(DBHandler.TABLE_EXPERIMENTS)
        self.cursor.execute(DBHandler.TABLE_RESULTS)
        self.con.commit()