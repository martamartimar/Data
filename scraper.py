import requests
import pandas as pd
import os
from datetime import datetime
import sys

URL_API = "https://valencia.opendatasoft.com/api/explore/v2.1/catalog/datasets/parkings/records?limit=100"

try:
    response = requests.get(URL_API)
    response.raise_for_status() # Si la web del ayuntamiento cae, esto hace saltar la alarma
    
    data = response.json()
    records = data.get('results', [])
    
    # Si la web responde pero no hay datos, paramos el robot
    if len(records) == 0:
        print("Error: El Ayuntamiento no ha devuelto ningún parking.")
        sys.exit(1)
    
    extracted_data = []
    fecha_actual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    for record in records:
        extracted_data.append({
            "fecha_registro": fecha_actual,
            "nombre": record.get("nombre", "Desconocido"),
            # Buscamos las columnas por varios nombres por si el Ayuntamiento los cambia
            "plazas_libres": record.get("plazaslibres", record.get("plazas_libres", "")),
            "capacidad": record.get("plazastotales", record.get("plazas_totales", "")),
            "latitud": record.get("geo_point_2d", {}).get("lat") if record.get("geo_point_2d") else "",
            "longitud": record.get("geo_point_2d", {}).get("lon") if record.get("geo_point_2d") else ""
        })
        
    df_nuevo = pd.DataFrame(extracted_data)
    csv_filename = "historico_parking.csv"
    
    if os.path.exists(csv_filename):
        df_nuevo.to_csv(csv_filename, mode='a', header=False, index=False)
    else:
        df_nuevo.to_csv(csv_filename, index=False)
        
    print(f"¡Éxito! Se han guardado {len(records)} parkings en el Excel.")

except Exception as e:
    print(f"Error crítico en Python: {e}")
    sys.exit(1) # Esto le avisa a GitHub de que detenga todo
