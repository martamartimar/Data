import requests
import pandas as pd
import os
from datetime import datetime
import sys

URL_API = "https://geoportal.valencia.es/server/rest/services/OPENDATA/Trafico/MapServer/194/query?where=1=1&outFields=*&f=json"

try:
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(URL_API, headers=headers, timeout=10)
    response.raise_for_status() 
    
    data = response.json()
    
    # En ArcGIS, los datos vienen dentro de una lista llamada 'features'
    features = data.get('features', [])
    
    if len(features) == 0:
        print("Error: El servidor del Ayuntamiento no ha devuelto datos.")
        sys.exit(1)
    
    extracted_data = []
    fecha_actual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    for feature in features:
        # Los valores están dentro de 'attributes' y las coordenadas en 'geometry'
        attributes = feature.get('attributes', {})
        geometry = feature.get('geometry', {})
        
        extracted_data.append({
            "fecha_registro": fecha_actual,
            "nombre": attributes.get("nombre", attributes.get("NOMBRE", "Desconocido")),
            "plazas_libres": attributes.get("plazaslibres", attributes.get("PLAZASLIBRES", 0)),
            "capacidad": attributes.get("plazastotales", attributes.get("PLAZASTOTALES", 0)),
            "longitud": geometry.get("x", 0),
            "latitud": geometry.get("y", 0)
        })
        
    df_nuevo = pd.DataFrame(extracted_data)
    csv_filename = "historico_parking.csv"
    
    if os.path.exists(csv_filename):
        df_nuevo.to_csv(csv_filename, mode='a', header=False, index=False)
    else:
        df_nuevo.to_csv(csv_filename, index=False)
        
    print(f"¡Éxito absoluto! Guardados {len(features)} parkings desde el Geoportal oficial.")

except Exception as e:
    print(f"Error crítico en Python: {e}")
    sys.exit(1)
