import pandas as pd

outputfile = 'sitetable.html'

csv = pd.read_csv('LocalExp_forecast_regioni_2020-03-23-telegram.csv', parse_dates=['data'])

grouped = csv.groupby('codice_regione')

output = ''
for code in csv['codice_regione'].unique():
    row = grouped.get_group(code)
    region = row['denominazione_regione'].values[0]
    if code == 4:
        region = 'Trentino-Alto Adige'
    regionid = 'table' + region.replace("'", '').replace(' ','').replace('-', '').lower()
    output += f"""
<div id="{regionid}" class="predtablediv">
    <table>
        <tr class="header">
            <td class="leftheader">{region}</td>
            <td>oggi</td>
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
"""
output = output.lstrip()

print(f'Writing output in {outputfile}...')
with open(outputfile, 'w') as file:
    file.write(output)
