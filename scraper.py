import requests
import pandas as pd
import os
from datetime import datetime

URL_API = "https://valencia.opendatasoft.com/api/explore/v2.1/catalog/datasets/parquings/records?limit=100"

try:
    response = requests.get(URL_API)
    data = response.json()
    records = data.get('results', [])
    
    extracted_data = []
    fecha_actual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    for record in records:
        extracted_data.append({
            "fecha_registro": fecha_actual,
            "nombre": record.get("nombre"),
            "plazas_libres": record.get("plazaslibres"),
            "capacidad": record.get("plazas_totales"),
            "latitud": record.get("geo_point_2d", {}).get("lat"),
            "longitud": record.get("geo_point_2d", {}).get("lon")
        })
        
    df_nuevo = pd.DataFrame(extracted_data)
    csv_filename = "historico_parking.csv"
    
    if os.path.exists(csv_filename):
        df_nuevo.to_csv(csv_filename, mode='a', header=False, index=False)
    else:
        df_nuevo.to_csv(csv_filename, index=False)
        
    print("¡Datos recolectados con éxito!")

except Exception as e:
    print(f"Error al recolectar datos: {e}")
