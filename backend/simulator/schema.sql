CREATE TABLE IF NOT EXISTS experiment(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        start_time TIMESTAMP NOT NULL,
        name_experiment TEXT NOT NULL,
        end_time TIMESTAMP,
        experiment_description TEXT NOT NULL,
        mcs_downlink INTEGER NOT NULL,
        mcs_uplink INTEGER NOT NULL,
        end_rb_downlink INTEGER NOT NULL,
        end_rb_uplink INTEGER NOT NULL,
        iperf_duration INTEGER NOT NULL,
        iperf_mode TEXT CHECK(iperf_mode IN ('downlink', 'uplink', 'both')),
        iperf_transport TEXT CHECK(iperf_transport IN ('tcp', 'udp')),
        iperf_bitrate TEXT CHECK(iperf_bitrate IN ('fixed', 'variable'))
);

CREATE TABLE IF NOT EXISTS results(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TIMESTAMP,
        downlink_rate FLOAT,
        uplink_rate FLOAT,
        snr FLOAT,
        cqi INTEGER,
        experiment_id integer,
        FOREIGN KEY (experiment_id) REFERENCES experiment(id)
);
