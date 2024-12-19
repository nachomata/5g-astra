from threading import Thread
from simulator import DBHandler, Simulation
from time import sleep, time
import pandas as pd
from flask import Flask, request, jsonify, send_file

DB_PATH = 'AstraSQLite.db'
db = DBHandler(DB_PATH)

app = Flask(__name__)

simulation_list = {}

@app.route('/api/v1/experiments', methods=['GET'])
def get_all_experiments():
    global db
    return jsonify(db.get_all_experiments())
@app.route('/api/v1/experiment/<int:id>', methods=['GET'])
def get_experiment_id(id:int):
    global db
    return jsonify(db.experiment_collect(id))

@app.route('/api/v1/experiment/<int:id>/results', methods=['GET'])
def get_experiment_results(id:int):
    global db
    return jsonify(db.result_collect(id))

def run_experiment(simulation):
    global simulation_list
    try:
        simulation.run()
        simulation_list[simulation.id] = {"status": Simulation.STATUS_COMPLETED,
                                    "progress": 100,
                                  }
    except Exception as e:
        print(e)
        simulation_list[simulation.id]["status"].value = Simulation.STATUS_ERROR

@app.route('/api/v1/experiment', methods=['POST'])
def create_experiment():
    data = request.json    
    global simulation_list
    
    simulation = Simulation(
        name = data['experiment_name'],
        dl_mcs = data['mcs_downlink'],
        ul_mcs = data['mcs_uplink'],
        dl_rb = data['end_rb_downlink'],
        ul_rb = data['end_rb_uplink'],
        iperf_duration = data['iperf_duration'],
        iperf_mode = data['iperf_direction'],
        iperf_transport = data['iperf_protocol'],
        iperf_bitrate = data['iperf_bitrate'],
        description = data['experiment_description']
    )
    
    simulation_list[simulation.id] = {"status": Simulation.STATUS_RUNNING,
                                  "progress": 0,
                                  }
    
    thread = Thread(target=run_experiment, args=(simulation,))
    thread.start()
    
    return jsonify({"id": simulation.id,
                    "status": Simulation.STATUS_MSG[simulation_list[simulation.id]["status"]]}), 201

@app.route('/api/v1/experiment/<int:id>', methods=['DELETE'])
def delete_experiment(id:int):
    global db
    print(db.delete_experiment(id))
    return jsonify({"message": "Experiment deleted"}), 200

@app.route('/api/v1/experiment/<int:id>/status', methods=['GET'])
def get_status(id:int):
    global db
    time_sleeps = 25

    exp = db.experiment_collect(id)
    start_time = pd.to_datetime(exp["start_time"],unit='s')
    end_time = start_time + exp["iperf_duration"] + time_sleeps
    actual_time = time()
    progress = (actual_time - start_time) / (end_time-start_time)

    return jsonify(progress)

if __name__ == '__main__':
    app.run(debug=True)
