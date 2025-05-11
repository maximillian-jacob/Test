import nest_asyncio
import PyUber
import warnings
import time
import asyncio
import pandas as pd
from datetime import datetime
nest_asyncio.apply() #### To allow async Pyuber code execution inside a jupyter nb
warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy connectable")

import subprocess
import sys
import importlib.util
import platform

def uninstall(package):
    """Uninstall a specified package."""
    print(f"Attempting to uninstall {package}...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'uninstall', '-y', package])
    print(f"Uninstallation of {package} complete.")

def install(package):
    """Install a package. Open a new terminal window on Windows to display progress."""
    print(f"Installing {package}...")
    subprocess.run([sys.executable, "-m", "pip", "install", package], check=True)
    print(f"Installation of {package} complete.")
    
# Informative print statement before checking Python dependencies
print("Checking Python dependencies for LAK Lot Dispo OneClick...")

# List of required packages
packages = [
    'pandas',
    'PyUber',
    'pytz',
    'matplotlib',
    'numpy'
]

# Check and install missing packages
for package in packages:
    if importlib.util.find_spec(package) is None:
        print(f"{package} is not installed. Installing now...")
        install(package)
    else:
        print(f"{package} is already installed.")

# List of packages to import for step indicator
import_modules = [
    'datetime', 'timedelta', 'pandas', 'PyUber', 'pytz', 'urllib.parse', 
    'webbrowser', 'json', 'matplotlib.pyplot', 'numpy', 
    'matplotlib.dates', 'tkinter', 'ttk', 'io.BytesIO', 'base64', 'tempfile', 'os'
]

imported_count = 0
total_imports = len(import_modules)

try:
    print("Importing Packages")
    
    from datetime import datetime, timedelta 
    imported_count += 1
    print(f"Imported datetime, timedelta({imported_count}/{total_imports})")
    
    import pandas as pd 
    imported_count += 1
    print(f"Imported pandas ({imported_count}/{total_imports})")
    
    import PyUber 
    imported_count += 1
    print(f"Imported PyUber ({imported_count}/{total_imports})")
    
    import pytz
    imported_count += 1
    print(f"Imported pytz ({imported_count}/{total_imports})")
    
    import urllib.parse 
    imported_count += 1 
    print(f"Imported urllib.parse ({imported_count}/{total_imports})")
    
    import webbrowser 
    imported_count += 1
    print(f"Imported webbrowser ({imported_count}/{total_imports})")
    
    import json 
    imported_count += 1
    print(f"Imported json ({imported_count}/{total_imports})")
    
    import matplotlib.pyplot as plt 
    imported_count += 1
    print(f"Imported matplotlib.pyplot ({imported_count}/{total_imports})")
    
    import numpy as np 
    imported_count += 1
    print(f"Imported numpy ({imported_count}/{total_imports})")
    
    from matplotlib.dates import DateFormatter, date2num, num2date
    imported_count += 1
    print(f"Imported DateFormatter, date2num, num2date ({imported_count}/{total_imports})")
    
    import tkinter as tk 
    imported_count += 1
    print(f"Imported tkinter ({imported_count}/{total_imports})")
    
    from tkinter import ttk 
    imported_count += 1
    print(f"Imported ttk ({imported_count}/{total_imports})")
    
    from io import BytesIO 
    imported_count += 1
    print(f"Imported io.BytesIO ({imported_count}/{total_imports})")
    
    import base64
    imported_count += 1
    print(f"Imported base64 ({imported_count}/{total_imports})")
    
    import tempfile 
    imported_count += 1
    print(f"Imported tempfile ({imported_count}/{total_imports})")
    
    import os 
    imported_count += 1
    print(f"Imported os ({imported_count}/{total_imports})")
    
    print("All imports successful")
    
except ImportError as e:
    print(f"Import error at step {imported_count + 1}/{total_imports}: {e}")
    
sites = ['D1D']
ds = [f'{site}_PROD_XEUS' for site in sites]

############################################
###########################################        

def run_sql(sql=None, datasource='ds'):
    """
    Connect to a database with a connection and run a query. Return data as a dataframe.
    """
    try:
        cxn = PyUber.connect(datasource)
        result = cxn.execute(sql)
        rows = result.fetchall()
        column_names = [x[0] for x in result.description]
        df = pd.DataFrame(rows, columns=column_names)
        return df
    except Exception as e:
        print(datetime.now(), '|', 'ERROR. Could not execute query:', e)
    finally:
        cxn.close()
        
####################
###################
###################



def sql_waferChamberHistory(operation, lot):
    """ 
    Construct the updated SQL query with given operation and lot
    """ 
    
    return f""" 
    SELECT 'D1D' "SITE"
      ,h.RUNKEY
      ,h.BATCH_ID
      ,h.LOTOPERKEY
      ,h.LOT
      ,h.OPERATION
      ,h.ROUTE
      ,c.WAFER "WAFER_ID"
      ---,c.SLOT
      ,CAST(c.SLOT AS INTEGER) "SLOT"
      ,c.START_TIME
      ,c.END_TIME
      ,c.STATE
      ,c.ENTITY
      ,c.CHAMBER
      ,c.ENTITY_CHAMBER
      ,c.SUBENTITY
      ,c.SUB_OPERATION
      ,c.CHAMBER_SEQUENCE
      ,c.CHAMBER_PROCESS_ORDER "PROCESS_ORDER"
      ,c.CHAMBER_PROCESS_DURATION "PROCESS_TIME"
      ,c.IN_SUITCASE_FLAG
      ,lr.RECIPE "LOT_RECIPE"
      ,wr.RECIPE "WAFER_RECIPE"
      ,cr.RECIPE "CHAMBER_RECIPE"
      ,a.ATTRIBUTE_STRING "ATTRIBUTES"
      ,h.PRODUCT
      ,h.LAST_TXN_TIME
    FROM F_LOTENTITYHIST h
    INNER JOIN F_WAFERENTITYHIST w
      ON w.RUNKEY=h.RUNKEY
      AND w.ENTITY=h.ENTITY
    INNER JOIN F_WAFERCHAMBERHIST c
      ON c.RUNKEY=h.RUNKEY
      AND c.ENTITY=h.ENTITY
      AND c.WAFER=w.WAFER
    INNER JOIN F_LOT_WAFER_RECIPE wr
      ON wr.RECIPE_ID=w.WAFER_RECIPE_ID
    INNER JOIN F_LOT_WAFER_RECIPE lr
      ON lr.RECIPE_ID=h.LOT_RECIPE_ID
    INNER JOIN F_LOT_WAFER_RECIPE cr
      ON cr.RECIPE_ID=c.WAFER_CHAMBER_RECIPE_ID
    INNER JOIN F_LOT_WAFER_ATTRIBUTE a
      ON a.ATTRIBUTE_ID=c.ATTRIBUTE_ID
    WHERE h.ENTITY IN (SELECT ENTITY FROM F_ENTITY WHERE (ENTITY LIKE 'BSP%' ESCAPE '\\'))
      AND h.LAST_TXN_TIME>=TRUNC(CURRENT_DATE)-60
      AND h.OPERATION IN ('{operation}')
      AND h.LOT='{lot}'

    """ 
    
########################################
#######################################
def plot_to_base64():
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

################################################
###############################################
def display_html_in_browser(html_content):
    file_path = os.path.expanduser("~/test_page.html")
    
    with open(file_path, "w") as f:
        f.write(html_content)
    
    webbrowser.open("file://" + file_path)
    
##############################
##############################

def export_to_html(df, plot_img_base64):
    html_table = df.to_html(classes='table', index=False, border=0)
    html_content = f'''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Wafer History Table</title>
        <style>
            /* Your CSS styles here */
        </style>
    </head>
    <body>
        <h2 style="text-align: center;">Wafer History</h2>
        {html_table}
        <div style="text-align: center;">
            <img src="data:image/png;base64,{plot_img_base64}" alt="Wafer History Plot"/>
        </div>
    </body>
    </html>
    '''
    display_html_in_browser(html_content)
    
def BSP_CT_handling(df):
    """ 
    This function add a new column called CHUCK_TABLE that represents which C/T this wafer is processed in.
    It would then filter out the C/T info from the CHAMBER column.    
    """
    mask = df['CHAMBER'].str.contains(r'^CT-.+', na=False)
    if mask.any(): #Checking if any of the row returns True
        value = df.loc[mask, 'CHAMBER'].iloc[0]
        df['CHUCK_TABLE'] = value
        
    df = df[df['CHAMBER'] != value]
    return df
  
def BSP_CT_handling(df):
    """ 
    This function add a new column called CHUCK_TABLE that represents which C/T this wafer is processed in.
    It would then filter out the C/T info from the CHAMBER column.    
    """
    mask = df['CHAMBER'].str.contains(r'^CT-.+', na=False)
 
    value = df.loc[mask, 'CHAMBER'].iloc[0]
    df['CHUCK_TABLE'] = value
        
    df = df[df['CHAMBER'] != value]
    return df
  
def clean_data(df):
    """ 
    Dropping missing values, then sort the data by start time in ascending manner
    """ 
    df.dropna(subset=['START_TIME', 'END_TIME'], inplace=True)
    df.sort_values(by='START_TIME', inplace=True)
    return df
  
def plot_BSP_wafer_history(waferChamberHistory, entity, lot):
    #Chamber sequence in order. This ensures the colormap's transition makes sense
    
    BSP_chambers = ['ROBOT', 'POSITION TABLE', 'T-ARM1', 'FRONT CHUCK TABLE',
                    'Z1', 'Z2', 'Z3', 'WAFER WASH', 'T-ARM2', 'SPINNER TABLE']

    bsp_colors = plt.cm.rainbow(np.linspace(0, 1, len(BSP_chambers)))
    bsp_color_map=dict(zip(BSP_chambers, bsp_colors))

    #Sort wafer IDs by their SLOT in descending order so that the wafer with the smallest slot (first wafer) is plotted last
    unique_wafers_df = waferChamberHistory[['WAFER_ID', 'SLOT']].drop_duplicates().sort_values('SLOT', ascending=True)
    plt.figure(figsize=(15,8))
    for wafer in unique_wafers_df['WAFER_ID']:
        #Focus only on a a single wafer. Sort by start_time in ascending order to obtain the correct sequence of chamber
        DF = waferChamberHistory[waferChamberHistory.WAFER_ID == wafer].sort_values(by='START_TIME', ascending=True)
        
        DF = BSP_CT_handling(DF)
                
        #resetting the index to allow index slicing
        DF = DF.reset_index(drop=True)
        
        
        base_time = DF.START_TIME[0]
        y_label = f"slot {DF.SLOT[0]}: {wafer}, {DF.CHUCK_TABLE.unique()[0]}"
        
        for I in range(len(DF)):
            start_time = DF.START_TIME[I]
            end_time = DF.END_TIME[I]
            chamber = DF.CHAMBER[I]
            start_sec = (start_time - base_time).total_seconds()
            duration = (end_time - start_time).total_seconds()
            
            plt.barh(y_label, duration, left=start_sec,
                    color=bsp_color_map[chamber], edgecolor='k', linewidth=0.5,
                    label=chamber)
            plt.legend(BSP_chambers, loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=3)
            plt.title(f"{entity}/{lot} Wafer Movement")
            plt.xlabel('Duration (s)')
            plt.ylabel('Wafer ID')
            plt.tight_layout()
            
def main_gui():
    
    def generate_BSP_wafer_movement_plot():
        status_label.config(text="Generating wafer movement plot...")
        root.update_idletasks()
        
        # Strip whitespace from inputs
        lot = lot_entry.get().strip()
        operation = oper_entry.get().strip()
        
        updated_query = sql_waferChamberHistory(operation, lot)
        waferChamberHistory = run_sql(updated_query, ds)
        # waferChamberHistory = BSP_CT_handling(waferChamberHistory)
        # print(waferChamberHistory.columns)
        waferChamberHistory = clean_data(waferChamberHistory)
        
        unique_entities = waferChamberHistory['ENTITY'].unique()
        # print('CHECKPOINT 22222222222222')
        plot_BSP_wafer_history(waferChamberHistory, unique_entities[0], lot)
        plot_img_base64 = plot_to_base64()
        export_to_html(pd.DataFrame(), plot_img_base64)
        
        status_label.config(text="")

    root = tk.Tk()
    root.title("LAK Lot Report Generator")
    root.geometry("450x370")  # This sets window width and height

    # Setup GUI input components
    ttk.Label(root, text="Lot ID (Last 60 days, no TWs):").grid(column=0, row=0, padx=5, pady=5, sticky=tk.W)
    lot_entry = ttk.Entry(root)
    lot_entry.grid(column=0, row=1, padx=5, pady=5, sticky=tk.W)
    lot_entry.insert(0, 'D5038270')

    ttk.Label(root, text="Operation:").grid(column=0, row=2, padx=5, pady=5, sticky=tk.W)
    oper_entry = ttk.Entry(root)
    oper_entry.grid(column=0, row=3, padx=5, pady=5, sticky=tk.W)
    oper_entry.insert(0, '260536')

    # Button for generating wafer movement plot
    tk.Button(root, text="1. Generate BSP Wafer Movement Plot", command=generate_BSP_wafer_movement_plot)\
        .grid(column=0, row=4, padx=4, pady=5, sticky=tk.W)

    # Close button
    tk.Button(root, text="Close", command=root.destroy)\
    .grid(column=0, row=8, padx=5, pady=5, sticky=tk.W)
    status_label = ttk.Label(root, text="")

    status_label.grid(column=0, row=5, padx=5, pady=5, sticky=tk.W)


    status_label = ttk.Label(root, text="")
    status_label.grid(column=0, row=11, padx=5, pady=5, sticky=tk.W)

    root.mainloop()

if __name__ == "__main__":
    main_gui()
