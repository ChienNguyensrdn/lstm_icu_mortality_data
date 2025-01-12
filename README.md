# lstm_icu_mortality_data
# Nghiên cứu  dữ liệu icu_mortality_data (một dạng biến thể timeseries với các kỹ thuật xử lý 
# B1. Tạo biến môi trường cho python (3.x)
	Windows:
		py -m venv .venv
	Unix/MacOS:
		python -m venv .venv
# Bước 2. Activate  environment
	Windows:
		.venv\Scripts\activate.bat
	Unix/MacOS:
		source .venv/bin/activate
# Bước 3: Install lib 
	pip install -r requirement.txt
# Bước 4 Run code 
python data_process.py
