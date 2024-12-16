import subprocess
import sys
import shutil
import time
import yaml
from Backend.DBhandler_class import DBHandler
from Backend.influxdb_client_class import INFLUXDB_CLIENT

class SIMULATION:
    
    GNB_ZMQ_FILE_PATH = r"C:\Users\juanj\OneDrive\Escritorio\UNIVERSIDAD\Master\ARM\nuevo_5gs\docker_open5gs\srsran\gnb_zmq.yml"
    EXPERIMENT_DURATION = 1
    DB_PATH = r"C:\Users\juanj\OneDrive\Escritorio\UNIVERSIDAD\Master\ARM\nuevo_5gs\docker_open5gs\experiments.db"
    URL = "http://localhost:8086"
    TOKEN = '605bc59413b7d5457d181ccf20f9fda15693f81b068d70396cc183081b264f3b'
    
    
    def __init__(self, dl_mcs, ul_mcs, dl_rb, ul_rb):
        
        self.dl_mcs = dl_mcs
        self.ul_mcs = ul_mcs
        self.dl_rb = dl_rb
        self.ul_rb = ul_rb
        
        self.db_handler = DBHandler(SIMULATION.DB_PATH)
        self.influxdb_handler = INFLUXDB_CLIENT(SIMULATION.TOKEN,SIMULATION.URL)
        
        self.db_handler.experiment_insert(
            dl_mcs = self.dl_mcs,
            ul_mcs = self.ul_mcs,
            dl_rb = self.dl_rb,
            ul_rb = self.ul_rb
        )
                
    ## SCRIPT PARA LANZAR LOS DOCKERS  
    def launch_network(self):
        # Comand 1
        self.run_command('docker compose -f "C:/Users/juanj/OneDrive/Escritorio/UNIVERSIDAD/Master/ARM/nuevo_5gs/docker_open5gs/sa-deploy.yaml" up -d')
        time.sleep(4)
        
        # Comand 2
        self.run_command('docker compose -f "C:/Users/juanj/OneDrive/Escritorio/UNIVERSIDAD/Master/ARM/nuevo_5gs/docker_open5gs/srsgnb_zmq.yaml" up -d')
        subprocess.Popen("docker container attach srsgnb_zmq", shell=True)
        time.sleep(4)

        # Comand 3
        self.run_command('docker compose -f "C:/Users/juanj/OneDrive/Escritorio/UNIVERSIDAD/Master/ARM/nuevo_5gs/docker_open5gs/srsue_5g_zmq.yaml" up -d')
        subprocess.Popen("docker container attach srsue_5g_zmq", shell=True)   
    
    #GNB MODIFIER
    def gnb_modifier(self,file_path, pdsch_mcs, pdsch_rb, pusch_mcs, pusch_rb):
        """
        Actualiza los valores de max_ue_mcs y end_rb en las secciones pdsch y pusch del archivo YAML.

        :param file_path: Ruta al archivo YAML.
        :param pdsch_mcs: Nuevo valor para pdsch > max_ue_mcs.
        :param pdsch_rb: Nuevo valor para pdsch > end_rb.
        :param pusch_mcs: Nuevo valor para pusch > max_ue_mcs.
        :param pusch_rb: Nuevo valor para pusch > end_rb.
        """
        try:
            # Leer el archivo YAML
            with open(file_path, 'r') as file:
                data = yaml.safe_load(file)

            # Actualizar los valores
            if 'cell_cfg' in data:
                if 'pdsch' in data['cell_cfg']:
                    data['cell_cfg']['pdsch']['max_ue_mcs'] = pdsch_mcs
                    data['cell_cfg']['pdsch']['end_rb'] = pdsch_rb
                if 'pusch' in data['cell_cfg']:
                    data['cell_cfg']['pusch']['max_ue_mcs'] = pusch_mcs
                    data['cell_cfg']['pusch']['end_rb'] = pusch_rb

            # Guardar los cambios en el archivo
            with open(file_path, 'w') as file:
                yaml.dump(data, file, default_flow_style=False)

            print("Archivo actualizado correctamente.")

        except Exception as e:
            print(f"Error al actualizar el archivo: {e}")
    
    
    def run_command(self,command):
        try:
            print(f"\nEjecutando: {command}")
            result = subprocess.run(command, shell=True, check=True)
            return result.returncode
        except subprocess.CalledProcessError as e:
            print(f"Error al ejecutar el comando: {e}")
            sys.exit(e.returncode)
    
    def run(self):
        
        ##MODIFY THE GNB_ZMQ_FILE
        self.gnb_modifier(
            file_path = SIMULATION.GNB_ZMQ_FILE_PATH,
            pdsch_mcs = self.dl_mcs,
            pdsch_rb = self.dl_rb,
            pusch_mcs = self.ul_mcs,
            pusch_rb = self.ul_rb
        )
        
        ## LAUNCH THE NETWORK --> DOCKER
        print('srsRAN  --> initialization')
        self.launch_network()     
        ## EXECUTE IPERFS
        time.sleep(4)
        print('NETWORK WORKING FINE')
        self.launch_iperfs()
        print('IPERFS DONE...')
        ## WAIT THE EXPERIMENT ENDS
        time.sleep(SIMULATION.EXPERIMENT_DURATION * 70)
        ## OBTAIN RESULTS
        print('IPERFS ENDING...')
        print('COLLECTING EXPERIMENT DATA')
        results = self.influxdb_handler.collect_experiment_data()
        #SAVE RESULTS IN DB
        print('SAVING DATA')
        self.save_results(results)
        #TURNING OFF THE NETWORK
        print('CLOSING THE NETWORK...')
        self.close_network()
        
        #CALCULATING THE MOS
        #mos = self.MOS_calculator()
        
        
    ## GUARDAR LOS DATOS EN LA BASE DE DATOS
    def save_results(self,results):        
        self.db_handler.result_insert(
            dl_rate = results['dl_rate'],
            uplink_rate = results['ul_rate'],
            snr = results['snr'],
            cqi = results['cqi'],
            experiment_id = self.db_handler.id_max_collect()
        )
    ## SCRIPT QUE CIERRE LOS DOCKERS PROPLAYERS 
    def close_network(self):
        # Comand 1
        self.run_command('docker compose -f "C:/Users/juanj/OneDrive/Escritorio/UNIVERSIDAD/Master/ARM/nuevo_5gs/docker_open5gs/sa-deploy.yaml" down')
        # Comand 2
        self.run_command('docker compose -f "C:/Users/juanj/OneDrive/Escritorio/UNIVERSIDAD/Master/ARM/nuevo_5gs/docker_open5gs/srsgnb_zmq.yaml" down')
        # Comand 3
        self.run_command('docker compose -f "C:/Users/juanj/OneDrive/Escritorio/UNIVERSIDAD/Master/ARM/nuevo_5gs/docker_open5gs/srsue_5g_zmq.yaml" down')
            
    def launch_iperfs(self):
        # Iniciar el servidor iperf3 en el contenedor 'upf'
        subprocess.Popen('docker exec -it upf /bin/bash -c "iperf -s" -d', shell=True)
        # Conectar el cliente iperf3 en el contenedor 'srsue_5g_zmq' al servidor iperf3 en 192.168.100.1
        subprocess.Popen('docker exec -it srsue_5g_zmq /bin/bash -c "iperf -c 192.168.100.1 -t 60" -d', shell=True)
        
    #ML MODEL QUE CALCULA LA MOS CON LOS DATOS EXPERIMENTOS
    def MOS_calculator(self):
        print("MANU BOBO ESPABILA CON EL ML")
        