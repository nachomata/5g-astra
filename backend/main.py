from simulator import InfluxDBCollector, DBHandler, represent
import csv
from io import BytesIO, StringIO
from multiprocessing import Process, Queue, Value
import secrets
from time import sleep
from zipfile import ZIP_DEFLATED, ZipFile
from flask import Flask, request, jsonify, send_file
from rich import print
from rich.traceback import install
from pyexcelerate import Workbook
import json
import matplotlib.pyplot as plt
import mpld3
import os
from flask_cors import CORS, cross_origin



#install(show_locals=True)

plt.switch_backend('Agg')
app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'
db = DBHandler()

running_experiments = {}

# Experiments
@app.route('/api/v1/experiment', methods=['GET']) #Este te da todos los experimentos completos (incluyendo ID)
@app.route('/api/v1/experiment/<int:id>', methods=['GET']) #eSTI TE DA EL EXPERIMENTO CONCRETO
@app.route('/api/v1/experiment', methods=['POST']) #Para añadir experimento
@app.route('/api/v1/experiment/<int:id>', methods=['DELETE']) # PARA eliminar experimento
@app.route('/api/v1/experiment/<int:id>/results', methods=['GET']) # Obtener resultados de un escenario





def get_all_experiments():
    experiments = db.get_experiments()
    for e in experiments:
        e['num_of_graphs'] = db.get_num_of_graphs(e['id'])
    return jsonify(experiments)

@app.route('/api/v1/experiment/<int:id>/<string:mode>', methods=['GET'])
def get_experiment(id: int, mode: str):
    if not db.check_experiment(id):
        return jsonify({"error": "Experiment not found"}), 404

    if mode == 'status':
        matching_nonce = None
        for nonce, experiment in running_experiments.items():
            if experiment["id"].value == id:
                matching_nonce = nonce
                break
        if matching_nonce:
            return jsonify({"status": Simulation.STATUS_MSG[running_experiments[matching_nonce]["status"].value],
                           "progress": running_experiments[matching_nonce]["progress"].value}), 200
        elif db.check_experiment(id):
            return jsonify({"status": "Completed"}), 200
        return jsonify({"status": "Not found"}), 404
    if mode == 'raw':
        reports = db.get_reports(id)
        reports_io = BytesIO()
        reports_io.write(json.dumps(reports, indent=4).encode('utf-8'))
        reports_io.seek(0)
        
        events = db.get_events(id)
        events_io = BytesIO()
        wb = Workbook()
        wb.new_sheet('Sheet1', data=[
            ["simulation_time", "event_type", "AP", "total_interfaces_throughput", "percentage_channels_change", "channels"]
        ] + [
            [e["simulation_time"], e["event_type"], e["AP"], e["total_interfaces_throughput"], e["percentage_channels_change"], e["channels"]]
            for e in events
        ])
        wb.save(events_io)
        events_io.seek(0)
        
        pollution = db.get_pollution(id)
        pollution_io = StringIO()
        writer = csv.DictWriter(pollution_io, fieldnames=["ID", "RSSI_channels"])
        writer.writeheader()
        for pollution_id, rssi_channels in pollution.items():
            rssi_channels_str = ' '.join(map(str, rssi_channels))
            writer.writerow({"ID": pollution_id, "RSSI_channels": rssi_channels_str})
        pollution_io.seek(0)
        
        zip_io = BytesIO()
        with ZipFile(zip_io, 'w', ZIP_DEFLATED) as zip_file:
            zip_file.writestr('reports.json', reports_io.read())
            zip_file.writestr('events.xlsx', events_io.read())
            zip_file.writestr('pollution.csv', pollution_io.read())
        zip_io.seek(0)
        return send_file(zip_io, mimetype='application/zip', as_attachment=True, download_name=f'experiment_{id}.zip')
    elif mode == 'graph':
        return jsonify(db.get_graphs(id))
    else:
        return jsonify({"error": "Invalid mode"}), 400

def graph_experiment(q, id):
    plt.close('all')
    represent.experiment_from_db(id)
    graphs = []
    for i in plt.get_fignums():
        fig = plt.figure(i)
        graphs.append(mpld3.fig_to_dict(fig))
        plt.close(fig)
    q.put(graphs)
    return graphs

@app.route('/api/v1/experiment/<int:id>/<string:ap_or_cluster>/<int:ap_or_cluster_id>', methods=['GET'])
def get_experiment_ap_or_cluster(id, ap_or_cluster, ap_or_cluster_id):
    if not db.check_experiment(id):
        return jsonify({"error": "Experiment not found"}), 404
    if ap_or_cluster == 'ap':
        if not db.check_ap(id, ap_or_cluster_id):
            return jsonify({"error": "AP not found"}), 404
        q = Queue()
        p = Process(target=graph_ap, args=(q, id, ap_or_cluster_id))
        p.start()
        graphs = q.get()
        p.join()
        return jsonify(graphs)
        
def graph_ap(q, experiment_id, ap_id):
    plt.close('all')
    represent.ap_from_db(experiment_id, ap_id)
    graphs = []
    for i in plt.get_fignums():
        fig = plt.figure(i)
        graphs.append(mpld3.fig_to_dict(fig))
        plt.close(fig)
    q.put(graphs)
    return graphs

def validate_experiment_data(data):
    if not isinstance(data, dict):
        return {'error': 'Data must be a dictionary'}, 400
    
    required_fields = ['scenario_id', 'simulations_hours', 'weights']
    if not all(field in data for field in required_fields):
        return {'error': 'Missing required fields: scenario_id, simulations_hours, weights'}, 400
    if not isinstance(data['weights'], list) or len(data['weights']) != 4:
        return {'error': 'Weights must be a list of length 4'}, 400
    if sum(data['weights']) != 1:
        return {'error': 'Weights must sum up to 1'}, 400
    if not isinstance(data['simulations_hours'], int) or data['simulations_hours'] <= 0:
        return {'error': 'Simulations hours must be a positive integer'}, 400
    if not db.check_scenario(data['scenario_id']):
        return {'error': 'Scenario ID does not exist'}, 404

    return None, 0

def run_experiment(data, status):
    # sys.stdout = open(str(secrets.token_bytes(16).hex()) + ".out", "w")
    simulation = Simulation()
    simulation.silent = True
    simulation.status = status
    simulation.scenario_id = data['scenario_id']
    simulation.simulation_hours = data['simulations_hours']
    simulation.weights = data['weights']
    try:
        simulation.start()
        status["status"].value = Simulation.STATUS_COMPLETED
        status["progress"].value = 100
    except Exception as e:
        print(e)
        status["status"].value = Simulation.STATUS_ERROR

@app.route('/api/v1/experiment', methods=['POST'])
def create_experiment():
    data = request.json
    validation_result, status_code = validate_experiment_data(data)
    if validation_result:
        return jsonify(validation_result), status_code
    
    global running_experiments
    nonce = secrets.token_hex(16)
    while nonce in running_experiments:
        nonce = secrets.token_hex(16)
    running_experiments[nonce] = {"id": Value('i', -1),
                                  "status": Value('i', Simulation.STATUS_RUNNING),
                                  "progress": Value('f', 0),
                                  }
    p = Process(target=run_experiment, args=(data, running_experiments[nonce]))
    p.start()
    
    while running_experiments[nonce]["id"].value == -1:
        sleep(1)
    
    return jsonify({"id": running_experiments[nonce]["id"].value,
                    "status": Simulation.STATUS_MSG[running_experiments[nonce]["status"].value]}), 201
    

@app.route('/api/v1/experiment/<int:id>', methods=['DELETE'])
def delete_experiment(id):
    if not db.check_experiment(id):
        return jsonify({"error": "Experiment not found"}), 404
    db.delete_experiment(id)
    return jsonify({"message": "Experiment deleted"}), 200

# Scenarios
@app.route('/api/v1/scenario', methods=['GET'])
def get_all_scenarios():
    return jsonify(db.get_scenarios())

def render_scenario(q, id, transparent=False):
    plt.close('all')
    represent.scenario2D_from_db(id)
    img = plt.figure(1)
    img_io = BytesIO()
    img.savefig(img_io, format='png', transparent=transparent)
    img_io.seek(0)
    plt.close(img)
    q.put(img_io)
    return img_io
    
@cross_origin()
@app.route('/api/v1/scenario/<int:id>/<string:mode>', methods=['GET'])
def get_scenario(id, mode):
    if not db.check_scenario(id):
        return jsonify({"error": "Scenario not found"}), 404
    if mode == 'raw':
        AP, clients = db.get_scenario(id)
        AP_dt = [ap for ap in AP if ap[9] == 0]
        ap_io = StringIO()
        writer = csv.writer(ap_io, delimiter=';')
        for line in AP:
            writer.writerow(line)
        ap_io.seek(0)
        
        ap_dt_io = StringIO()
        writer = csv.writer(ap_dt_io, delimiter=';')
        for line in AP_dt:
            writer.writerow(line)
        ap_dt_io.seek(0)
            
        clients_io = StringIO()
        writer = csv.writer(clients_io, delimiter=';')
        for line in clients:
            writer.writerow(line)
        clients_io.seek(0)
        
        zip_io = BytesIO()
        with ZipFile(zip_io, 'w', ZIP_DEFLATED) as zip_file:
            zip_file.writestr('AP.csv', ap_io.getvalue())
            zip_file.writestr('AP_dt.csv', ap_dt_io.getvalue())
            zip_file.writestr('clients.csv', clients_io.getvalue())
        zip_io.seek(0)
        return send_file(zip_io, mimetype='application/zip', as_attachment=True, download_name=f'scenario_{id}.zip'), 200
    elif mode == 'graph':
        represent.scenario2D_from_db(id)
        graphs = []
        for i in plt.get_fignums():
            fig = plt.figure(i)
            graphs.append(mpld3.fig_to_dict(fig))
            plt.close(fig)
        return jsonify(graphs[0]), 200
    elif mode == 'img':
        transparent_str = request.args.get('transparent', 'False')
        transparent = True if transparent_str.lower() == 'true' else False
        q = Queue()
        p = Process(target=render_scenario, args=(q, id, transparent))
        p.start()
        img_io = q.get()
        p.join()
        return send_file(img_io, mimetype='image/png', download_name=f'scenario_{id}.png'), 200
    else:
        return jsonify({"error": "Invalid mode"}), 400

@app.route('/api/v1/scenario', methods=['POST'])
def create_scenario():
    if 'file_ap' not in request.files or 'file_client' not in request.files:
        return jsonify({"error": "Files are missing: file_ap, file_client"}), 400

    file_ap = request.files['file_ap']
    file_client = request.files['file_client']
    name = request.form.get('name')
    description = request.form.get('description')

    if not name or not description:
        return jsonify({"error": "Name and description are required"}), 400

    # Save the files temporarily with a unique name
    tmp_dir = os.path.join(os.getcwd(), 'tmp')
    os.makedirs(tmp_dir, exist_ok=True)
    file_ap_path = os.path.join(tmp_dir, f"{secrets.token_hex(16)}.csv")
    file_client_path = os.path.join(tmp_dir, f"{secrets.token_hex(16)}.csv")
    file_ap.save(file_ap_path)
    file_client.save(file_client_path)

    try:
        scenario_id = db.add_scenario_from_csv(file_ap_path, file_client_path, name, description)
        return jsonify({"id": scenario_id}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            if os.path.exists(file_ap_path):
                os.remove(file_ap_path)
            if os.path.exists(file_client_path):
                os.remove(file_client_path)
        except Exception as e:
            return jsonify({"error": f"Error cleaning up files: {str(e)}"}), 500

@app.route('/api/v1/scenario/<int:id>', methods=['DELETE'])
def delete_scenario(id):
    if not db.check_scenario(id):
        return jsonify({"error": "Scenario not found"}), 404
    db.delete_scenario(id)
    return jsonify({"message": "Scenario deleted"}), 200

@app.route('/')
def hello():
    return "Backend de MELODIC simulator, don't kill me!"

if __name__ == '__main__':
    app.run(debug=True)
