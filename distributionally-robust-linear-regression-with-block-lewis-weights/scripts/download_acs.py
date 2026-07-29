import os, sys, zipfile, requests
state_codes = {'AL':'01','AK':'02','AZ':'04','AR':'05','CA':'06','CO':'08','CT':'09','DE':'10','FL':'12','GA':'13','HI':'15','ID':'16','IL':'17','IN':'18','IA':'19','KS':'20','KY':'21','LA':'22','ME':'23','MD':'24','MA':'25','MI':'26','MN':'27','MS':'28','MO':'29','MT':'30','NE':'31','NV':'32','NH':'33','NJ':'34','NM':'35','NY':'36','NC':'37','ND':'38','OH':'39','OK':'40','OR':'41','PA':'42','RI':'44','SC':'45','SD':'46','TN':'47','TX':'48','UT':'49','VT':'50','VA':'51','WA':'53','WV':'54','WI':'55','WY':'56','PR':'72'}
datadir = 'data/2018/1-Year'
os.makedirs(datadir, exist_ok=True)
base = 'https://www2.census.gov/programs-surveys/acs/data/pums/2018/1-Year'
done = 0; skipped = 0; failed = []
for st, code in state_codes.items():
    fn = f'psam_p{code}.csv'
    path = os.path.join(datadir, fn)
    if os.path.isfile(path) and os.path.getsize(path) > 1000:
        skipped += 1; continue
    remote = f'csv_p{st.lower()}.zip'
    url = f'{base}/{remote}'
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        zp = os.path.join(datadir, remote)
        with open(zp,'wb') as h: h.write(r.content)
        with zipfile.ZipFile(zp) as z: z.extract(fn, path=datadir)
        os.remove(zp)
        sz = os.path.getsize(path)
        done += 1
        print(f'{st}: {sz} bytes', flush=True)
    except Exception as e:
        failed.append((st, str(e)[:80]))
        print(f'{st}: FAILED {e}', flush=True)
print(f'DONE downloaded={done} skipped={skipped} failed={failed}')
