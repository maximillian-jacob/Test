# %%
import nest_asyncio
import PyUber
import warnings
import time
import asyncio
import pandas as pd
from datetime import datetime

# %%
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import smtplib

nest_asyncio.apply() #### To allow async Pyuber code execution inside a jupyter nb
warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy connectable")

# %%
import subprocess
import sys
import importlib.util
import platform

# %%
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

# %%
# List of required packages
packages = [
    'pandas',
    'PyUber',
    'pytz',
    'matplotlib',
    'numpy'
]

# %%
# Check and install missing packages
for package in packages:
    if importlib.util.find_spec(package) is None:
        print(f"{package} is not installed. Installing now...")
        install(package)
    else:
        print(f"{package} is already installed.")


# %%
# List of packages to import for step indicator
import_modules = [
    'datetime', 'timedelta', 'pandas', 'PyUber', 'pytz', 'urllib.parse', 
    'webbrowser', 'json', 'matplotlib.pyplot', 'numpy', 
    'matplotlib.dates', 'tkinter', 'ttk', 'io.BytesIO', 'base64', 'tempfile', 'os'
]

# %%
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

# %%
sites = ['D1D']
ds = [f'{site}_PROD_XEUS' for site in sites]

# %%
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

# %%
####################
###################
###################


def sql_spc():
    """ 
    Construct the updated SQL query with given operation and lot
    """ 
    
    return f"""
select *
from f_entityattribute ea
where ea.entity in ('BSP02', 'BSP05', 'BSP07', 'BSP409', \
                    'BSP411', \
                    'TBT01', 'TBT02', 'TBT03', 'TBT04', 'TBT405', 'TBT407', 'TBT409', 'TBT411', 'TBT413')
and ea.attribute_name like 'SPCChartValidation%'

    """ 

# %%
query = sql_spc()
df = run_sql(query, ds)



# %%
# Email setup
#Email setup
me = "maximillian.jacob@intel.com" 
you = "maximillian.jacob@intel.com" 

#Convert TXN_date to datetime
df['TXN_DATE'] = pd.to_datetime(df['TXN_DATE'])

#Calculate the number of days since the last transcation date
current_date = datetime.now()
df['DAYS_AGO'] = (current_date - df['TXN_DATE']).dt.total_seconds() / (60 * 60 * 24) 
df['DAYS_AGO'] = df['DAYS_AGO'].round(1) #Rounding to one decimal place

#Count the number of 'N' in SPC Validation
tools_off_count = df[df['ATTRIBUTE_VALUE'] == 'N'].shape[0]

#Creating a message container
msg = MIMEMultipart('alternative')
msg['Subject'] = f"TBT/BSP SPC Validation Flag ({tools_off_count} Tools off)"
msg['From'] = me 
msg['To'] = you

#HTML content
html = """\
    <html>
        <head>
            <meta http-equiv="Content-Type" content="text/html"; charset="UTF-8">
            <title>SPC Validation Flag</title>
            <style type="text/css" media="screen">
                table {
                    border-collapse: collapse;
                    width: 60%;
                }
                th, td {
                    border: 1px solid black;
                    padding: 8px;
                    text-align: center;
                }
                th {
                    background-color: #f2f2f2;
                    text-align: center;
                }
            </style>
        </head>
        
        <body>
            <p style="font-size: 25px; font-weight: bold;">BSP/TBT SPC Validation Flag</p>
            
            <table>
                <tr>
                    <th style='width: 20%;'>Entity</th>
                    <th style='width: 50%;'>Txn Date</th>
                    <th style='width: 10%;'>SPC Validation</th>
                    <th style='width: 20%;'>Last Transaction (days)</th>
                </tr>
    """ 

#Adding rows to the HTML able
for index, row in df.iterrows():
    color = 'green' if row['ATTRIBUTE_VALUE'] == 'Y' else 'red'
    html += f""" 
            <tr>
                <td>{row['ENTITY']}</td>
                <td>{row['TXN_DATE']}</td>
                <td style= "background-color: {color};">{row['ATTRIBUTE_VALUE']}</td>
                <td>{row['DAYS_AGO']}</td>
            </tr>
        """ 

#Closing the HTML content
html += """\
    </table>
    </body>
    </html>
    """
    
#Attach HTML part
part1 = MIMEText(html, 'html')
msg.attach(part1)

#Sending the email
s = smtplib.SMTP('smtp.intel.com')
s.sendmail(me, you, msg.as_string())
s.quit()  
        