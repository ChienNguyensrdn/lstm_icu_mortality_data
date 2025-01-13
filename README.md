
###  Nghiên cứu dữ liệu ICU Mortality Data:
1. **Flow Chart - Quy trình xử lý dữ liệu**:
```mermaid
graph TD
    A[Start] --> B[Read File]
    B --> C[Convert Time to Step]
    C --> D[Fill Missing Values]
    D --> E[Predict Missing Values]
    E --> F[Combine DataFrames]
    F --> G[Scale Data]
    G --> H[Create Dataset]
    H --> I[Reshape Data for LSTM]
    I --> J[Load LSTM Model]
    J --> K[Predict with LSTM]
    K --> L[Inverse Transform Predictions]
    L --> M[Calculate MSE]
    M --> N[Print Results]
    N --> O[End]
```
### Mathematical Model Summary:
![image](https://github.com/user-attachments/assets/5b3d4669-6c25-4748-a273-a58929c4841f)
### Next:
![image](https://github.com/user-attachments/assets/794af3db-82fc-40bd-9f00-90d5e9cdb1e5)


2. **Hướng dẫn Cài đặt và Chạy Code**:
   - Bước 1: Tạo môi trường ảo co Python (phiên bản 3.x)
     ## Windows:
     		py -m venv .venv
     ## Unix/MacOS:
     		python -m venv .venv
   - Bước 2: Kích hoạt môi trường:
     ## Windows:
     		.venv\Scripts\activate.bat
     ## Unix/MacOS:
     		source .venv/bin/activate
     
   - Bước 3: Cài đặt các thư viện cần thiết
     ## Install:
     		pip install -r requirements.txt
   - Bước 4: Chạy mã xử lý dữ liệu
     ## Run:
    		python data_process.py
    		
 **Chi tiết Quy trình**:
   # Dữ liệu đọc từ data file :
   ![image](https://github.com/user-attachments/assets/6d307d2b-63a1-4747-8100-e0d9997964ed)
    
   # Convert time to step:
![image](https://github.com/user-attachments/assets/211e573f-5ef5-47ba-8592-d321c2cdcfc0)
   # Filled missing values 
![image](https://github.com/user-attachments/assets/b063be32-cc67-4cde-ae91-85646a2cc773)
   # Predicted missing values with XGBRegressor
![image](https://github.com/user-attachments/assets/50abe892-21a4-426f-84ec-aca89d432608)
   # Build LSTM 
![image](https://github.com/user-attachments/assets/a6869bd1-3dbc-435e-8566-a69b36870528)
