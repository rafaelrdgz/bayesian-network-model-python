from Alarm import alarm


bn = alarm()
print(bn.query('Burglary', evidence={'John calls' : True, 'Mary calls' : True}));