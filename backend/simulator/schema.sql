CREATE TABLE IF NOT EXISTS experiment(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        mcs_downlink INTEGER NOT NULL,
        mcs_uplink INTEGER NOT NULL,
        end_rb_downlink INTEGER NOT NULL,
        end_rb_uplink INTEGER NOT NULL,
        iperf_duration INTEGER NOT NULL,
        iperf_mode TEXT CHECK(iperf_mode IN ('downlink', 'uplink', 'both')),
        iperf_transport TEXT CHECK(iperf_transport IN ('tcp', 'udp')),
        iperf_type TEXT CHECK(iperf_type IN ('fixed', 'variable'))
);

CREATE TABLE IF NOT EXISTS results(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        downlink_rate FLOAT,
        uplink_rate FLOAT,
        snr FLOAT,
        cqi INTEGER,
        experiment_id integer,
        FOREIGN KEY (experiment_id) REFERENCES experiment(id)
);
