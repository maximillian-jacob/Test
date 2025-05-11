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
    WHERE h.ENTITY IN (
        SELECT ENTITY FROM F_ENTITY
        WHERE (ENTITY LIKE 'TBT%' ESCAPE '\\' or ENTITY LIKE 'BSP%' ESCAPE '\\' or ENTITY LIKE 'PBD%' ESCAPE '\\')
    )
      AND h.LAST_TXN_TIME>=TRUNC(CURRENT_DATE)-60
      AND h.OPERATION IN ('{operation}')
      AND h.LOT='{lot}'

    """ 
    

def clean_data(df):
    df.dropna(subset=['START_TIME', 'END_TIME'], inplace=True)
    df.sort_values(by='START_TIME', inplace=True)
    return df


def plot_wafer_history(waferChamberHistory, entity, lot):
    #Optimized code

    waferChamberHistory['base_time'] = waferChamberHistory.groupby('WAFER_ID')['START_TIME'].transform('min')
    waferChamberHistory['start_sec'] = (waferChamberHistory['START_TIME'] - waferChamberHistory['base_time']).dt.total_seconds()
    waferChamberHistory['duration'] = (waferChamberHistory['END_TIME'] - waferChamberHistory['START_TIME']).dt.total_seconds()

    unique_entities = waferChamberHistory['ENTITY'].unique()
    unique_chambers = waferChamberHistory['CHAMBER'].unique()

    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_chambers)))
    color_map=dict(zip(unique_chambers, colors))
    #Order wafers based on their SLOT in descending order
    unique_wafers_df = waferChamberHistory[['WAFER_ID', 'SLOT']].drop_duplicates().sort_values('SLOT', ascending=True)

    #Loop only over wafers groups and plt all bars for each wafer at once (vectorized per group)
    plt.figure(figsize=(15,8))
    for wafer in unique_wafers_df['WAFER_ID']:
        wafer_group = waferChamberHistory[waferChamberHistory.WAFER_ID == wafer]
        y_label = f"Slot {wafer_group.SLOT.iloc[0]}: {wafer}"
        plt.barh(
            y_label,
            wafer_group['duration'],
            left = wafer_group['start_sec'],
            color=wafer_group['CHAMBER'].map(color_map),
            edgecolor='k', linewidth=0.5,
            label=wafer_group['CHAMBER']
        )
        plt.legend(unique_chambers, loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=3)
        plt.title(f'{entity} / {lot} Wafer Movement')  # Include entity in the title
        plt.title(f"{unique_entities}/{lot}")
        plt.xlabel('Duration (s)')
        plt.ylabel('Wafer ID')
        plt.tight_layout()
        
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
  
def plot_BSP_wafer_history(waferChamberHistory, entity, lot):
    #Chamber sequence in order. This ensures the colormap's transition makes sense
    
    BSP_chambers = ['ROBOT', 'POSITION TABLE', 'T-ARM1', 'FRONT CHUCK TABLE',
                    'Z1', 'Z2', 'Z3', 'WAFER WASH', 'T-ARM2', 'SPINNER TABLE']

    bsp_colors = plt.cm.rainbow(np.linspace(0, 1, len(BSP_chambers)))
    bsp_color_map=dict(zip(BSP_chambers, bsp_colors))

    #Sort wafer IDs by their SLOT in descending order so that the wafer with the smallest slot (first wafer) is plotted last
    #Keeping asceding to False. Ascending=True only works if the entire lot is fully processed.
    #If the lot is still processing in the tool, asceding=True will not produce any images since lowest wafer hasn't processed and it will trigger the "break" from except:
    unique_wafers_df = waferChamberHistory[['WAFER_ID', 'SLOT']].drop_duplicates().sort_values('SLOT', ascending=False)
    plt.figure(figsize=(15,8))
    for wafer in unique_wafers_df['WAFER_ID']:
        #Focus only on a a single wafer. Sort by start_time in ascending order to obtain the correct sequence of chamber
        DF = waferChamberHistory[waferChamberHistory.WAFER_ID == wafer].sort_values(by='START_TIME', ascending=True)
        
        try:
            DF = BSP_CT_handling(DF)
        except:
            print('At least 1 wafer is currently processing in the tool')
            break
                
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

  
def plot_pbd_wafer_history(waferChamberHistory, entity, lot):
    #Chamber sequence in order. This ensures the colormap's transition makes sense

    pbd_sub_operation = waferChamberHistory.SUB_OPERATION.unique().tolist()
    
    pbd_colors = plt.cm.rainbow(np.linspace(0, 1, len(pbd_sub_operation)))
    pbd_color_map=dict(zip(pbd_sub_operation, pbd_colors))

    #Sort wafer IDs by their SLOT in descending order so that the wafer with the smallest slot (first wafer) is plotted last
    unique_wafers_df = waferChamberHistory[['WAFER_ID', 'SLOT']].drop_duplicates().sort_values('SLOT', ascending=True)


    plt.figure(figsize=(12,8))
    test_chamber = []
    # test_chamber = set()
    for wafer in unique_wafers_df['WAFER_ID']:
        DF = waferChamberHistory[waferChamberHistory.WAFER_ID == wafer].sort_values(by='START_TIME', ascending=True)
        
        DF = DF.reset_index(drop=True)
            
        base_time = DF.START_TIME[0]
        y_label = f"slot {DF.SLOT[0]}: {wafer}"
        
        for I in range(len(DF)):

            start_time = DF.START_TIME[I]
            end_time = DF.END_TIME[I]
            # chamber = DF.CHAMBER[I] #changing to sub_operation
            sub_operation = DF.SUB_OPERATION[I]
            start_sec = (start_time - base_time).total_seconds()
            duration = (end_time - start_time).total_seconds()
            chamber = DF.CHAMBER[I]
            # If chamber is like CLM*, then set chamber to 'CLM'
            if chamber.startswith('CLM'):
                chamber = 'CLM'
                
            if chamber.startswith('EAM'):
                chamber = 'EAM Bottom'
                
            if len(test_chamber) < (len(pbd_sub_operation)):
                test_chamber.append(chamber)
            
            plt.barh(y_label, duration, left=start_sec,
                    color=pbd_color_map[sub_operation], edgecolor='k', linewidth=0.5,
                    label=sub_operation)
            # plt.legend(BSP_chambers, loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=3)
            # plt.legend(chamber, loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=3)
            plt.legend(test_chamber, loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=3)

            plt.title(f"{entity}/{lot}")
            plt.xlabel('Duration (s)')
            plt.ylabel('Wafer ID')
            plt.tight_layout()

def sql_PBD_lot_dispo(lot, prod_group):
    """ 
    Construct the updated SQL query with given operation and lot
    """ 
    
    return f""" 
    -- First Query
WITH FirstQuery AS (
    SELECT 
        TO_CHAR(meas_set_data_collect_date, 'yyyy-mm-dd hh24:mi:ss') AS meas_set_data_collect_date,
        lot,
        foup_slot,
        waf3,
        wlt_component,
        TO_CHAR(wlt_end_time, 'yyyy-mm-dd hh24:mi:ss') AS wlt_end_time,
        raw_parameter_name,
        MAX(raw_value) AS raw_value,
        measurement_set_name,
        raw_parameter_name_1,
        spc_operation
    FROM (
        SELECT  
            a3.data_collection_time AS meas_set_data_collect_date,
            a0.lot,
            a4.foup_slot,
            a4.wafer3 AS waf3,
            wlt.component AS wlt_component,
            wlt.end_time AS wlt_end_time,
            a4.parameter_name AS raw_parameter_name,
            a4.value AS raw_value,
            a3.measurement_set_name,
            a4.parameter_name AS raw_parameter_name_1,
            a0.operation AS spc_operation
        FROM 
            P_SPC_MEASUREMENT_SET a3
        INNER JOIN 
            P_SPC_SESSION a2 ON a2.spcs_id = a3.spcs_id AND a2.data_collection_time = a3.data_collection_time
        LEFT JOIN 
            P_SPC_LOT a0 ON a0.spcs_id = a2.spcs_id
        LEFT JOIN 
            P_SPC_MEASUREMENT a4 ON a4.spcs_id = a3.spcs_id AND a4.measurement_set_name = a3.measurement_set_name
        LEFT JOIN 
            P_SPC_WLT_Component wlt ON wlt.spcs_id = a4.spcs_id AND wlt.wafer = a4.wafer
        WHERE
            a0.lot = '{lot}' 
            AND a3.data_collection_time >= TRUNC(SYSDATE) - 7 
            AND a2.test_name IN ('7PSTPBDANNEAL', '8PSTPBDANNEAL', '80PBDPSTANNEAL', '80PBDPSTANNEAL110') 
            AND a3.measurement_set_name IN ('TSV_PBD.WGT_BE.78', 'TSV_PBD.WGT_BE.77', 'TSV_PBD.WGT_BE.80') 
            AND a4.parameter_name IN ('RESXRAISMEANRMS', 'RESYRAISMEANRMS') 
    ) AS subquery
    GROUP BY 
        meas_set_data_collect_date,
        lot,
        foup_slot,
        waf3,
        wlt_component,
        wlt_end_time,
        raw_parameter_name,
        measurement_set_name,
        raw_parameter_name_1,
        spc_operation
),

-- Second Query
SecondQuery AS (
    SELECT 
        a0.product AS product_c,
        a0.lot AS lot_c,
        REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(a0.lot_title, ',', ';'), CHR(9), ' '), CHR(10), ' '), CHR(13), ' '), CHR(34), ''''), CHR(7), ' ') AS lot_title_c,
        a0.lot_owner AS lot_owner_c
    FROM 
        F_EXPERIMENT a0
    INNER JOIN
        F_EXPT_CONTEXT a1 ON a1.history_deleted_flag = 'N' AND a1.lotplan = a0.lotplan
    WHERE
        (a1.operation LIKE '156988%' OR
         a1.operation LIKE '237329%' OR
         a1.operation LIKE '157018%' OR
         a1.operation LIKE '236824%' OR
         a1.operation LIKE '216467%' OR
         a1.operation LIKE '245886%') 
        AND a0.src_erase_date IS NULL  
        AND a0.lotplan_state != 'Terminated'
),

-- Third Query
ThirdQuery AS (
    SELECT 
        a0.parameter_name AS parameter_name,
        a1.lo_control_lmt AS lcl,
        a1.up_control_lmt AS ucl
    FROM 
        P_SPC_CHART a0
    INNER JOIN 
        P_SPC_CHART_LIMIT a1 ON a1.chart_id = a0.chart_id
    WHERE
        (a0.chart_on LIKE 'TSV_PBD.IS_RES.78' OR
         a0.chart_on LIKE 'TSV_PBD.IS_RES.80') 
        AND a0.spc_chart_load_date >= TRUNC(SYSDATE) - 1 
        AND a0.spc_chart_ordered_categories LIKE '{prod_group}'
        AND a1.latest_limit_flag = 'Y'
)

-- Combined Query
SELECT 
    fq.meas_set_data_collect_date,
    fq.lot,
    fq.foup_slot,
    fq.waf3,
    fq.wlt_component,
    fq.wlt_end_time,
    fq.raw_parameter_name,
    fq.raw_value,
    fq.measurement_set_name,
    fq.raw_parameter_name_1,
    fq.spc_operation,
    sq.product_c,
    sq.lot_title_c,
    sq.lot_owner_c,
    tq.lcl,
    tq.ucl
FROM 
    FirstQuery fq
LEFT JOIN 
    SecondQuery sq ON fq.lot = sq.lot_c
LEFT JOIN 
    ThirdQuery tq ON fq.raw_parameter_name = tq.parameter_name;
    """ 


def lot_dispo_preprocessing(df):
    """
    Preprocess the lot disposition DataFrame.

    Parameters:
    df (pd.DataFrame): A DataFrame containing the raw data.

    Returns:
    pd.DataFrame: A DataFrame with preprocessed data.
    """
    # Convert WLT_END_TIME to datetime for plotting
    df['WLT_END_TIME'] = pd.to_datetime(df['WLT_END_TIME'])

    # Sort the DataFrame by WLT_END_TIME in ascending order
    df = df.sort_values(by='WLT_END_TIME', ascending=True)

    # Create a new column that represents the slot and wid
    df['SLOT_WID'] = '(s' + df['FOUP_SLOT'].astype(str) + ', w' + df['WAF3'].astype(str) +')'

    return df


def plot_dispo_chart(df):
    """ 
    The code defines a function `plot_dispo_chart(df)` that generates a chart with two subplots. 
    Each subplot displays scatter plots for different components based on the input DataFrame `df`.
    
    """
    
    # Create subplots
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 10), sharex=True)

    # First subplot for RESXRAISMEANRMS
    ax1 = axes[1]
    subset_resx = df[df['RAW_PARAMETER_NAME'] == 'RESXRAISMEANRMS']
    for component in subset_resx['WLT_COMPONENT'].unique():
        subset = subset_resx[subset_resx['WLT_COMPONENT'] == component]
        ax1.scatter(subset['WLT_END_TIME'], subset['RAW_VALUE'], label=component)
        for i, row in subset.iterrows():
            ax1.text(row['WLT_END_TIME'], row['RAW_VALUE'] - 0.05, row['SLOT_WID'], ha='center', va='top', fontsize=10)

    # Get unique LCL and UCL for RESX
    lcl_resx = subset_resx['LCL'].unique()[0]
    ucl_resx = subset_resx['UCL'].unique()[0]

    ax1.axhline(y=lcl_resx, color='blue', linestyle='--')
    ax1.axhline(y=ucl_resx, color='red', linestyle='--')
    ax1.text(1.01, lcl_resx, 'LCL', color='blue', va='center', ha='left', transform=ax1.get_yaxis_transform())
    ax1.text(1.01, ucl_resx, 'UCL', color='red', va='center', ha='left', transform=ax1.get_yaxis_transform())
    ax1.set_title('RESXRAISMEANRMS')
    ax1.set_ylabel('Raw Value')
    ax1.set_xlabel('WLT End Time')
    ax1.legend(loc='center left', bbox_to_anchor=(1.03, 0.5))

    # Second subplot for RESYRAISMEANRMS
    ax2 = axes[0]
    subset_resy = df[df['RAW_PARAMETER_NAME'] == 'RESYRAISMEANRMS']
    for component in subset_resy['WLT_COMPONENT'].unique():
        subset = subset_resy[subset_resy['WLT_COMPONENT'] == component]
        ax2.scatter(subset['WLT_END_TIME'], subset['RAW_VALUE'], label=component)
        for i, row in subset.iterrows():
            ax2.text(row['WLT_END_TIME'], row['RAW_VALUE'] - 0.05, row['SLOT_WID'], ha='center', va='top', fontsize=10)

    # Get unique LCL and UCL for RESY
    lcl_resy = subset_resy['LCL'].unique()[0]
    ucl_resy = subset_resy['UCL'].unique()[0]

    ax2.axhline(y=lcl_resy, color='blue', linestyle='--')
    ax2.axhline(y=ucl_resy, color='red', linestyle='--')
    ax2.text(1.01, lcl_resy, 'LCL', color='blue', va='center', ha='left', transform=ax2.get_yaxis_transform())
    ax2.text(1.01, ucl_resy, 'UCL', color='red', va='center', ha='left', transform=ax2.get_yaxis_transform())
    ax2.set_title('RESYRAISMEANRMS')
    ax2.set_ylabel('Raw Value')
    ax2.legend(loc='center left', bbox_to_anchor=(1.03, 0.5))

    # Adjust layout
    plt.tight_layout()
    plt.show()

def PBD_dispo_template(df):
    """ 
    This function outputs the result using a template proposed by the PBD Eng team.
    """
    # HTML tags for bold text
    bold_start = "<strong>"
    bold_end = "</strong>"

    # Create HTML content
    html_content = "<div style='font-family: Arial, sans-serif; font-size: 30px;'>"

    # Lot and product name
    lot = df['LOT'].iloc[0]
    product = df['PRODUCT_C'].iloc[0]
    html_content += f"<p>Lot / Product: {bold_start}{lot}{bold_end} / {bold_start}{product}{bold_end}</p>"

    # Lot title and lot owner
    lot_title = df['LOT_TITLE_C'].iloc[0]
    lot_owner = df['LOT_OWNER_C'].iloc[0]
    html_content += f"<p>Lot Title/Lot Owner: {bold_start}{lot_title}{bold_end} / {bold_start}{lot_owner}{bold_end}</p>"

    # WLT_Components
    wlt_components = ", ".join(df['WLT_COMPONENT'].unique())
    html_content += f"<p>WLT_Components: {bold_start}{wlt_components}{bold_end}</p>"

    # Wafers that are out of control
    x_above_ucl = df[(df['RAW_PARAMETER_NAME'].str.startswith('RESX')) & (df['RAW_VALUE'] > df['UCL'])]['SLOT_WID'].tolist()
    x_below_lcl = df[(df['RAW_PARAMETER_NAME'].str.startswith('RESX')) & (df['RAW_VALUE'] < df['LCL'])]['SLOT_WID'].tolist()
    y_above_ucl = df[(df['RAW_PARAMETER_NAME'].str.startswith('RESY')) & (df['RAW_VALUE'] > df['UCL'])]['SLOT_WID'].tolist()
    y_below_lcl = df[(df['RAW_PARAMETER_NAME'].str.startswith('RESY')) & (df['RAW_VALUE'] < df['LCL'])]['SLOT_WID'].tolist()

    if x_above_ucl:
        html_content += f"<p>Wafers above RESX UCL: {bold_start}{', '.join(x_above_ucl)}{bold_end}</p>"
    if x_below_lcl:
        html_content += f"<p>Wafers below RESX LCL: {bold_start}{', '.join(x_below_lcl)}{bold_end}</p>"
    if y_above_ucl:
        html_content += f"<p>Wafers above RESY UCL: {bold_start}{', '.join(y_above_ucl)}{bold_end}</p>"
    if y_below_lcl:
        html_content += f"<p>Wafers below RESY LCL: {bold_start}{', '.join(y_below_lcl)}{bold_end}</p>"

    html_content += "</div>"

    return html_content

def PBD_export_to_html(df, template_html):
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
        <h2 style="text-align: center;">PBD Dispo Template</h2>
        {html_table}
        
        {template_html}
    </body>
    </html>
    '''
    display_html_in_browser(html_content)


def main_gui():
    
    def generate_wafer_movement_plot():
        status_label.config(text="Generating wafer movement plot...")
        root.update_idletasks()
        
        # Strip whitespace from inputs
        lot = lot_entry.get().strip()
        operation = oper_entry.get().strip()
        
        updated_query = sql_waferChamberHistory(operation, lot)
        waferChamberHistory = run_sql(updated_query, ds)
        waferChamberHistory = clean_data(waferChamberHistory)
        
        unique_entities = waferChamberHistory['ENTITY'].unique()
        
        try:
            plot_wafer_history(waferChamberHistory, unique_entities[0], lot)
        except Exception as e:
            print('No data. Verify lot and operation.')
        plot_img_base64 = plot_to_base64()
        export_to_html(pd.DataFrame(), plot_img_base64)
        
        status_label.config(text="")
    
    def generate_BSP_wafer_movement_plot():
        status_label.config(text="Generating wafer movement plot...")
        root.update_idletasks()
        
        # Strip whitespace from inputs
        lot = lot_entry.get().strip()
        operation = oper_entry.get().strip()
        
        updated_query = sql_waferChamberHistory(operation, lot)
        waferChamberHistory = run_sql(updated_query, ds)
        waferChamberHistory = clean_data(waferChamberHistory)
        
        unique_entities = waferChamberHistory['ENTITY'].unique()
        try:
            plot_BSP_wafer_history(waferChamberHistory, unique_entities[0], lot)
        except Exception as e:
            print('No data. Verify lot and operations.')
            print(f"Error: {e}")
            return
        
        plot_img_base64 = plot_to_base64()
        export_to_html(pd.DataFrame(), plot_img_base64)
        
        status_label.config(text="")

    def generate_PBD_wafer_movement_plot():
        status_label.config(text="Generating wafer movement plot...")
        root.update_idletasks()
        
        # Strip whitespace from inputs
        lot = lot_entry.get().strip()
        operation = oper_entry.get().strip()
        
        updated_query = sql_waferChamberHistory(operation, lot)
        waferChamberHistory = run_sql(updated_query, ds)
        waferChamberHistory = clean_data(waferChamberHistory)
                    
        unique_entities = waferChamberHistory['ENTITY'].unique()
        try:
            plot_pbd_wafer_history(waferChamberHistory, unique_entities[0], lot)
        except Exception as e:
            print('No Data. Verify Lot and Operation')
            print(f"Error: {e}")
            return
        
        plot_img_base64 = plot_to_base64()
        export_to_html(pd.DataFrame(), plot_img_base64)
        
        status_label.config(text="")  
        
    def generate_lot_dispo_chart_plot():
        status_label.config(text='Generating lot dispo chart plot...')
        root.update_idletasks()
        
        #Strip whitespace from inputs
        lot = lot_entry.get().strip()
        prod_group = oper_entry.get().strip()
        prod_group = '%' + prod_group
        query = sql_PBD_lot_dispo(lot, prod_group)
        df = run_sql(query, ds)
        df = lot_dispo_preprocessing(df)

        plot_dispo_chart(df)
        
        template_html = PBD_dispo_template(df)
        PBD_export_to_html(pd.DataFrame(), template_html)
        
        status_label.config(text="")      


    root = tk.Tk()
    root.title("BSP/TBT Report Generator")
    root.geometry("450x370")  # This sets window width and height

    # Setup GUI input components
    ttk.Label(root, text="Lot ID (Last 60 days, no TWs):").grid(column=0, row=0, padx=5, pady=5, sticky=tk.W)
    lot_entry = ttk.Entry(root)
    lot_entry.grid(column=0, row=1, padx=5, pady=5, sticky=tk.W)
    lot_entry.insert(0, 'D511T4F0')

    ttk.Label(root, text="Operation or Prod_group (for PBD dispo):").grid(column=0, row=2, padx=5, pady=5, sticky=tk.W)
    oper_entry = ttk.Entry(root)
    oper_entry.grid(column=0, row=3, padx=5, pady=5, sticky=tk.W)
    oper_entry.insert(0, '192229')
    
    # Button for generating wafer movement plot
    tk.Button(root, text="1. Generate BSP Wafer Movement Plot", command=generate_BSP_wafer_movement_plot)\
        .grid(row=4, column=0, padx=4, pady=5, sticky=tk.W)

    # Button for generating wafer movement plot
    tk.Button(root, text="2. Generate TBT Wafer Movement Plot", command=generate_wafer_movement_plot)\
        .grid(row=5, column=0, padx=4, pady=5, sticky=tk.W)
        
    tk.Button(root, text="3. Generate PBD Wafer Movement Plot", command=generate_PBD_wafer_movement_plot)\
        .grid(row=6, column=0, padx=4, pady=5, sticky=tk.W)
        
    tk.Button(root, text="4. Generate PBD Dispo Chart Plot", command=generate_lot_dispo_chart_plot)\
        .grid(row=8, column=0, padx=4, pady=5, sticky=tk.W)

    # Close button
    tk.Button(root, text="Close", command=root.destroy)\
    .grid(column=1, row=5, padx=5, pady=5, sticky=tk.W)
    status_label = ttk.Label(root, text="")

    status_label.grid(column=0, row=5, padx=5, pady=5, sticky=tk.W)


    status_label = ttk.Label(root, text="")
    status_label.grid(column=0, row=11, padx=5, pady=5, sticky=tk.W)

    root.mainloop()

if __name__ == "__main__":
    main_gui()
