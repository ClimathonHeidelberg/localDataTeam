import cdsapi
c = cdsapi.Client()
c.retrieve(
   'satellite-carbon-dioxide',
   {
       'format':'zip',
       'processing_level':'level_3',
       'variable':'xco2',
       'sensor_and_algorithm':'merged_obs4mips',
       'version':'3.1'
   },
   'download.zip')
