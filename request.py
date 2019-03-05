import requests
url = 'http://localhost:5000/api'
r = requests.post(url,json={'':0,'Beat':2432,'Crime_Date_Day':3,'Crime_month':5,'Year':2016,'DayOT':0,})
print(r.json())