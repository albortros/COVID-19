import pandas as pd
import sys
import glob

outputfile = 'sitetable.html'

if len(sys.argv) == 2:
    csvfile = sys.argv[1]
else:
    files = glob.glob('2020-??-??_GP.csv')
    if files:
        files.sort()
        csvfile = files[-1]
    else:
        csvfile = '2020-04-08_GP.csv'
csv = pd.read_csv(csvfile, parse_dates=['data'])

grouped = csv.groupby('codice_regione')

output = ''
for code in csv['codice_regione'].unique():
    row = grouped.get_group(code)
    
    # id for the outer div.
    region = row['denominazione_regione'].values[0]
    if code == 4:
        region = 'Trentino-Alto Adige'
    regionid = 'table' + region.replace("'", '').replace(' ','').replace('-', '').lower()
    
    today = row['data'].max()
    midnight = pd.Timestamp(year=today.year, month=today.month, day=today.day + 1, hour=0, minute=0, second=0)
    
    output += f'''
<div class="predtablediv">
    <div class="{today}">
        <table>
            <tr class="header">
                <td class="leftheader">{region}</td>
                <td>oggi (dati)</td>
                <td>domani</td>
            </tr>
            <tr>
                <td class="leftheader">casi</td>
                <td>{row['casi_oggi'].sum():+.0f}</td>
                <td>{row['casi_domani'].sum():+.0f}</td>
            </tr>
            <tr>
                <td class="leftheader">deceduti</td>
                <td>{row['morti_oggi'].sum():+.0f}</td>
                <td>{row['morti_domani'].sum():+.0f}</td>
            </tr>
        </table>
    </div>
    <div class="{midnight}">
        <table>
            <tr class="header">
                <td class="leftheader">{region}</td>
                <td>oggi (pred.)</td>
                <td>domani</td>
            </tr>
            <tr>
                <td class="leftheader">casi</td>
                <td>{row['casi_domani'].sum():+.0f}</td>
                <td>{row['casi_dopodomani'].sum():+.0f}</td>
            </tr>
            <tr>
                <td class="leftheader">deceduti</td>
                <td>{row['morti_domani'].sum():+.0f}</td>
                <td>{row['morti_dopodomani'].sum():+.0f}</td>
            </tr>
        </table>
    </div>
</div>
'''
output = output.lstrip()

print(f'Writing output in {outputfile}...')
with open(outputfile, 'w') as file:
    file.write(output)
