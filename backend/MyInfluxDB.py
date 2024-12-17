from influxdb_client import InfluxDBClient, Point, WritePrecision
import pandas as pd
import numpy as np
from itertools import zip_longest

class InfluxDBCollector():
    
    NUM_CELLS_ACTIVE_UE_QUERY = """
        from(bucket: \"srsran\")
        |> range(start: -1m, stop: now())
        |> filter(fn: (r) => r[\"_measurement\"] == \"ue_info\")
        |> filter(fn: (r) => r[\"testbed\"] == \"default\")
        |> filter(fn: (r) => r[\"_field\"] == \"dl_brate\")
        |> window(every: 2s)\n  |> group(columns: [\"_stop\"])
        |> unique(column: \"pci\")\n  |> count(column: \"pci\")
        |> map(fn: (r) => ({ r with _value: r[\"pci\"] }))
        |> drop(columns: [\"pci\"])
        |> group()
    """

    ACTIVES_UE_QUERY = """
        from(bucket: \"srsran\")
        |> range(start: -1m, stop: now())
        |> filter(fn: (r) => r[\"_measurement\"] == \"ue_info\")
        |> filter(fn: (r) => r[\"testbed\"] == \"default\")
        |> filter(fn: (r) => r[\"_field\"] == \"dl_brate\")
        |> map(fn: (r) => ({ r with ue_id: r[\"pci\"]+\".\"+r[\"rnti\"]}))
        |> window(every: 1s)
        |> group(columns: [\"_stop\"])
        |> unique(column: \"ue_id\")
        |> count(column: \"ue_id\")
        |> map(fn: (r) => ({ r with _value: r[\"ue_id\"] }))
        |> drop(columns: [\"ue_id\"])
        |> group()
    """

    CURRENT_TOTAL_DL_RATE_QUERY = """
        from(bucket: \"srsran\")
        |> range(start: -1m, stop: now())
        |> filter(fn: (r) => r[\"_measurement\"] == \"ue_info\")
        |> filter(fn: (r) => r[\"testbed\"] == \"default\")
        |> filter(fn: (r) => r[\"_field\"] == \"dl_brate\")
        |> window(every: 1s)
        |> group(columns: [\"_stop\"])
        |> sum(column: \"_value\")
        |> group()
        |> movingAverage(n: 2) 
    """

    MAXIMUM_TOTAL_DL_RATE_QUERY = """
        from(bucket: \"srsran\")
        |> range(start: -1m, stop: now())
        |> filter(fn: (r) => r[\"_measurement\"] == \"ue_info\")
        |> filter(fn: (r) => r[\"testbed\"] == \"default\")
        |> filter(fn: (r) => r[\"_field\"] == \"dl_brate\")
        |> window(every: 1s)
        |> group(columns: [\"_stop\"])
        |> sum(column: \"_value\")
        |> group()
        |> movingAverage(n: 2)  
    """

    DL_RATE_QUERY = """
        from(bucket: \"srsran\")
        |> range(start: -1m, stop: now())
        |> filter(fn: (r) => r[\"_measurement\"] == \"ue_info\")
        |> filter(fn: (r) => r[\"testbed\"] == \"default\")
        |> filter(fn: (r) => r[\"_field\"] == \"dl_brate\")
    """

    DL_MCS_QUERY = """
        from(bucket: \"srsran\")
        |> range(start: -1m, stop: now())
        |> filter(fn: (r) => r[\"_measurement\"] == \"ue_info\")  
        |> filter(fn: (r) => r[\"testbed\"] == \"default\")
        |> filter(fn: (r) => r[\"_field\"] == \"dl_mcs\")
    """

    UL_RATE_QUERY = """
        from(bucket: \"srsran\")
        |> range(start: -1m, stop: now())
        |> filter(fn: (r) => r[\"_measurement\"] == \"ue_info\")
        |> filter(fn: (r) => r[\"testbed\"] == \"default\")
        |> filter(fn: (r) => r[\"_field\"] == \"ul_brate\")
    """

    UL_MCS_QUERY = """
        from(bucket: \"srsran\")
        |> range(start: -1m, stop: now())
        |> filter(fn: (r) => r[\"_measurement\"] == \"ue_info\")  
        |> filter(fn: (r) => r[\"testbed\"] == \"default\")
        |> filter(fn: (r) => r[\"_field\"] == \"ul_mcs\")
    """

    UL_SNR_QUERY = """
        from(bucket: \"srsran\")
        |> range(start: -1m, stop: now())
        |> filter(fn: (r) => r[\"_measurement\"] == \"ue_info\")  
        |> filter(fn: (r) => r[\"testbed\"] == \"default\")
        |> filter(fn: (r) => r[\"_field\"] == \"pusch_snr_db\")
    """

    CQI_QUERY = """
        from(bucket: \"srsran\")
        |> range(start: -1m, stop: now())
        |> filter(fn: (r) => r[\"_measurement\"] == \"ue_info\")  
        |> filter(fn: (r) => r[\"testbed\"] == \"default\")
        |> filter(fn: (r) => r[\"_field\"] == \"cqi\")
    """
    
    def __init__(self, token, url):
        self.token = token
        self.url = url
        self.client = InfluxDBClient(url=url, token=token, org='srs')
        self.query_api = self.client.query_api()
    
    def collect_dl_rate(self):
        values = []
        tables = self.query_api.query(InfluxDBCollector.DL_RATE_QUERY, org="srs")
        for table in tables:
            for record in table.records:
                values.append(record['_value'])    
        return values
    
    def collect_ul_rate(self):
        values = []
        tables = self.query_api.query(InfluxDBCollector.UL_RATE_QUERY, org="srs")
        for table in tables:
            for record in table.records:
                values.append(record['_value'])    
        return values
    
    def collect_snr(self):
        values = []
        tables = self.query_api.query(InfluxDBCollector.UL_SNR_QUERY, org="srs")
        for table in tables:
            for record in table.records:
                values.append(record['_value'])    
        return values
        
    def collect_cqi(self):
        values = []
        tables = self.query_api.query(InfluxDBCollector.CQI_QUERY, org="srs")
        for table in tables:
            for record in table.records:
                values.append(record['_value'])    
        return values
    
    def collect_experiment_data(self):
        dl_rate = self.collect_dl_rate()
        ul_rate = self.collect_ul_rate()
        snr = self.collect_snr()
        cqi = self.collect_cqi()

        data = list(zip_longest(dl_rate, ul_rate, snr, cqi, fillvalue=np.nan))
        df = pd.DataFrame(data, columns=["dl_rate", "ul_rate", "snr", "cqi"])
        return df
        
    