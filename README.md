# lstm_icu_mortality_data
	Nghiên cứu  dữ liệu icu_mortality_data (một dạng biến thể timeseries với các kỹ thuật xử lý )
# Bước 1. Tạo biến môi trường cho python (3.x)
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

# Flow chart 
	Dữ liệu đọc từ data file :

![image](https://github.com/user-attachments/assets/6d307d2b-63a1-4747-8100-e0d9997964ed)

	Convert time to step:
![image](https://github.com/user-attachments/assets/211e573f-5ef5-47ba-8592-d321c2cdcfc0)

	Filled missing values 
 ![image](https://github.com/user-attachments/assets/b063be32-cc67-4cde-ae91-85646a2cc773)

 	Predicted missing values with XGBRegressor
![image](https://github.com/user-attachments/assets/50abe892-21a4-426f-84ec-aca89d432608)
	
 	Build LSTM 
 
 ![image](https://github.com/user-attachments/assets/a6869bd1-3dbc-435e-8566-a69b36870528)
