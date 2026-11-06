# Model card
Purpose: estimate plant AC output if inverters are running, from current sunlight, temperatures and time.
Not for: forecasting, fault diagnosis, financial settlement, or plants other than the two trained here.
Target: median running inverter multiplied by inverter count. Split: chronological 60/20/20 by day. Choice: lowest calibration MAE; test data is only reported. Metrics, data hashes and versions are in models/metrics.json.
Limits: 34 days, two plants, units are source units, Plant 2 includes many recorded stops, and irradiance outside training range triggers a warning.
Fingerprint: SHA-256 of the model file. It proves the file and receipts are unchanged. It does not prove accuracy or input truth. Commit FINGERPRINT.txt publicly for it to mean anything.
