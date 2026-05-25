import requests
import pandas as pd
import os
from datetime import datetime
import sys

URL_API = "https://geoportal.valencia.es/server/rest/services/OPENDATA/Trafico/MapServer/194/query?where=1=1&outFields=*&outSR=4326&f=json"

try:
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(URL_API, headers=headers, timeout=10)
    response.raise_for_status() 
    
    data = response.json()
    features = data.get('features', [])
    
    if len(features) == 0:
        print("Error: El servidor no ha devuelto datos.")
        sys.exit(1)
    
    extracted_data = []
    fecha_actual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    for feature in features:
        attributes = feature.get('attributes', {})
        geometry = feature.get('geometry', {})
        
        # Usamos los nombres exactos que tú has descubierto en el JSON
        extracted_data.append({
            "fecha_registro": fecha_actual,
            "nombre": attributes.get("nombre", "Desconocido"),
            "capacidad": attributes.get("plazastota", 0),
            "plazas_libres": attributes.get("plazaslibr", 0),
            "longitud": geometry.get("x", 0),
            "latitud": geometry.get("y", 0)
        })
        
    df_nuevo = pd.DataFrame(extracted_data)
    csv_filename = "historico_parking.csv"
    
    if os.path.exists(csv_filename):
        df_nuevo.to_csv(csv_filename, mode='a', header=False, index=False)
    else:
        df_nuevo.to_csv(csv_filename, index=False)
        
    print(f"¡Éxito! Parkings guardados con capacidad real y coordenadas GPS.")

except Exception as e:
    print(f"Error crítico en Python: {e}")
    sys.exit(1)
