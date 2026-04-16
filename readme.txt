We provide a Python script to run ROC/CMC evaluation directly on ROI data:
"..\roi\session1" as gallery and "..\roi\session2" as probe.

1) Install dependencies
	pip install -r requirements.txt

2) Run evaluation
	python run_roc_cmc.py

Optional arguments:
	--gallery <path_to_session1>
	--probe <path_to_session2>
	--out <output_folder>

Outputs:
	roc_curve.png
	roc_curve_logx.png
	roc_curve_zoom.png
	cmc_curve.png
	roc_cmc_results.npz