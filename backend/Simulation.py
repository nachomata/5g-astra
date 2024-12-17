import subprocess
import sys
import shutil
import time
import yaml
import os
from DBhandler import DBHandler
from MyInfluxdb import InfluxDBCollector

class Simulation:
    
    GNB_ZMQ_FILE_PATH = os.path.join("..", "docker_open5gs", "srsran", "gnb_zmq.yml")
    EXPERIMENT_DURATION = 1
    DB_PATH = os.path.join(".", "AstraSQLite.db")
    URL = "http://localhost:8086"
    TOKEN = '605bc59413b7d5457d181ccf20f9fda15693f81b068d70396cc183081b264f3b'
    
    
    def __init__(self, dl_mcs, ul_mcs, dl_rb, ul_rb, 
                 iperf_duration, iperf_mode, iperf_transport, iperf_type):
        
        self.dl_mcs = dl_mcs
        self.ul_mcs = ul_mcs
        self.dl_rb = dl_rb
        self.ul_rb = ul_rb
        
        self.iperf_duration = iperf_duration
        self.iperf_mode = iperf_mode
        self.iperf_transport = iperf_transport
        self.iperf_type = iperf_type
        
        self.db_handler = DBHandler(Simulation.DB_PATH)
        self.influxdb_handler = InfluxDBCollector(Simulation.TOKEN,Simulation.URL)
        
        self.db_handler.experiment_insert(
            dl_mcs=self.dl_mcs,
            ul_mcs=self.ul_mcs,
            dl_rb=self.dl_rb,
            ul_rb=self.ul_rb,
            iperf_duration=self.iperf_duration,
            iperf_mode=self.iperf_mode,
            iperf_transport=self.iperf_transport,
            iperf_type=self.iperf_type
        )
                
      
    def launch_network(self):
        base_path = os.path.join(os.path.dirname(__file__), "docker_open5gs")
        sa_deploy_path = os.path.join(base_path, "sa-deploy.yaml")
        srsgnb_zmq_path = os.path.join(base_path, "srsgnb_zmq.yaml")
        srsue_5g_zmq_path = os.path.join(base_path, "srsue_5g_zmq.yaml")

        # Core
        self.run_command(f'docker compose -f "{sa_deploy_path}" up -d')
        time.sleep(4)
        # GNB
        self.run_command(f'docker compose -f "{srsgnb_zmq_path}" up -d')
        time.sleep(4)
        # UE
        self.run_command(f'docker compose -f "{srsue_5g_zmq_path}" up -d')  
    
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
            file_path = Simulation.GNB_ZMQ_FILE_PATH,
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
        self.launch_iperfs(
            direction = self.iperf_mode,
            protocol = self.iperf_transport,
            duration = self.iperf_duration,
            mode = self.iperf_type

        )
        print('IPERFS DONE...')
        ## WAIT THE EXPERIMENT ENDS
        time.sleep(self.iperf_duration + 10)
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
        base_path = os.path.join(os.path.dirname(__file__), "docker_open5gs")
        sa_deploy_path = os.path.join(base_path, "sa-deploy.yaml")
        srsgnb_zmq_path = os.path.join(base_path, "srsgnb_zmq.yaml")
        srsue_5g_zmq_path = os.path.join(base_path, "srsue_5g_zmq.yaml")

        # Core
        self.run_command(f'docker compose -f "{sa_deploy_path}" down')
        # GNB
        self.run_command(f'docker compose -f "{srsgnb_zmq_path}" down')
        # UE
        self.run_command(f'docker compose -f "{srsue_5g_zmq_path}" down')  
            
    def launch_iperfs(self, direction="uplink", protocol="tcp", duration=60, mode="fixed"):
        """
        Lanza iperf en la dirección, protocolo, duración y modo especificados.

        Parámetros:
            direction (str): Dirección del tráfico.
                            "uplink"   -> Cliente en srsue_5g_zmq, servidor en upf.
                            "downlink" -> Cliente en upf, servidor en srsue_5g_zmq.
                            "both"     -> Tráfico bidireccional usando flag -r.
            protocol (str): Protocolo a usar: "tcp" o "udp". Por defecto es "tcp".
            duration (int): Duración del tráfico en segundos. Por defecto es 60.
            mode (str): Modo de tráfico: "fixed" o "variable". Por defecto es "fixed".
        """
        # Validar parámetros 'protocol' y 'mode'
        if protocol not in ["tcp", "udp"]:
            print("Protocolo no válido. Usa 'tcp' o 'udp'.")
            return
        if mode not in ["fixed", "variable"]:
            print("Modo no válido. Usa 'fixed' o 'variable'.")
            return

        # Configurar opciones para iperf
        protocol_flag = "-u" if protocol == "udp" else ""  # Modo UDP si se selecciona
        reverse_flag = "-r" if direction == "both" else ""  # Tráfico bidireccional
        bandwidth_flag = ""  # Opciones de ancho de banda

        # Configurar el modo fijo o variable
        if mode == "variable":
            bandwidth_flag = "-b 1M:100M" if protocol == "udp" else "-b 10M"  # Ejemplo de tráfico variable
        elif mode == "fixed":
            bandwidth_flag = "-b 100M" if protocol == "udp" else ""  # Fijo a 10 Mbps en UDP

        if direction == "uplink":
            # Servidor iperf en 'upf'
            subprocess.Popen('docker exec -it upf /bin/bash -c "iperf -s {flag}" -d'.format(flag=protocol_flag), shell=True)
            # Cliente iperf en 'srsue_5g_zmq'
            subprocess.Popen(
                'docker exec -it srsue_5g_zmq /bin/bash -c "iperf {flag} {reverse} {bandwidth} -c 192.168.100.1 -t {duration}" -d'.format(
                    flag=protocol_flag, reverse=reverse_flag, duration=duration, bandwidth=bandwidth_flag
                ),
                shell=True
            )
        elif direction == "downlink":
            # Servidor iperf en 'srsue_5g_zmq'
            subprocess.Popen('docker exec -it srsue_5g_zmq /bin/bash -c "iperf -s {flag}" -d'.format(flag=protocol_flag), shell=True)
            # Cliente iperf en 'upf'
            subprocess.Popen(
                'docker exec -it upf /bin/bash -c "iperf {flag} {reverse} {bandwidth} -c 192.168.100.2 -t {duration}" -d'.format(
                    flag=protocol_flag, reverse=reverse_flag, duration=duration, bandwidth=bandwidth_flag
                ),
                shell=True
            )
        elif direction == "both":
            # Servidor iperf en 'upf'
            subprocess.Popen('docker exec -it upf /bin/bash -c "iperf -s {flag}" -d'.format(flag=protocol_flag), shell=True)
            # Cliente iperf en 'srsue_5g_zmq' con tráfico bidireccional
            subprocess.Popen(
                'docker exec -it srsue_5g_zmq /bin/bash -c "iperf {flag} -r {bandwidth} -c 192.168.100.1 -t {duration}" -d'.format(
                    flag=protocol_flag, duration=duration, bandwidth=bandwidth_flag
                ),
                shell=True
            )
        else:
            print("Dirección no válida. Usa 'uplink', 'downlink' o 'both'.")
        
    #ML MODEL QUE CALCULA LA MOS CON LOS DATOS EXPERIMENTOS
    def MOS_calculator(self):
        print("MANU BOBO ESPABILA CON EL ML")
        